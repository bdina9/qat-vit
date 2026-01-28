# project/src/models/model_registry.py
"""
Cross-Platform Model Registry: Jetson-Safe ViT + Optional OWL-ViT/Pruning

Goals:
- Jetson-safe imports (no heavy deps required on-device)
- Model registry + factory helpers for teacher/student creation
- Optional QAT wrapper for training/inference scaffolding
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

# =============================================================================
# PLATFORM DETECTION (Jetson safety)
# =============================================================================
def is_jetson() -> bool:
    """
    Best-effort Jetson detection.
    Returns:
        True if Jetson likely detected, False otherwise.
    """
    # Jetson device-tree marker (often binary; decode safely)
    dt_model = Path("/proc/device-tree/model")
    if dt_model.exists():
        try:
            raw = dt_model.read_bytes()
            model = raw.decode("utf-8", errors="ignore").lower()
            return ("jetson" in model) or ("nvidia" in model)
        except Exception:
            pass

    # Heuristics: tegra libs / paths
    if any("tegra" in p.lower() for p in sys.path):
        return True
    if Path("/usr/lib/aarch64-linux-gnu/tegra").exists():
        return True

    return False


def get_platform() -> str:
    """Return 'jetson' or 'desktop'."""
    return "jetson" if is_jetson() else "desktop"


PLATFORM = get_platform()
_IS_JETSON = PLATFORM == "jetson"

# Avoid print-at-import for libraries; keep as warning (non-fatal)
if _IS_JETSON:
    warnings.warn("Jetson platform detected - enabling edge-optimized defaults", RuntimeWarning)

# =============================================================================
# CORE DEPENDENCIES
# =============================================================================
_TIMM_AVAILABLE = False
try:
    import timm  # type: ignore

    _TIMM_AVAILABLE = True
except ImportError as e:
    if not _IS_JETSON:
        raise ImportError("timm required for ViT models. Install with: pip install timm") from e
    warnings.warn("timm not available on Jetson - ViT models unavailable", RuntimeWarning)

# =============================================================================
# OPTIONAL DEPENDENCIES (avoid on Jetson)
# =============================================================================
_TRANSFORMERS_AVAILABLE = False
if not _IS_JETSON:
    try:
        from transformers import AutoConfig, AutoModelForObjectDetection  # type: ignore

        _TRANSFORMERS_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        warnings.warn("transformers not available - OWL-ViT models disabled", RuntimeWarning)

# =============================================================================
# Quantization API compatibility
# =============================================================================
try:
    # PyTorch 2.x preferred
    from torch.ao.quantization import DeQuantStub, QuantStub  # type: ignore
except Exception:
    # Older PyTorch fallback
    from torch.quantization import DeQuantStub, QuantStub  # type: ignore

# =============================================================================
# QAT WRAPPER
# =============================================================================
class QATWrapper(nn.Module):
    """
    Quantization-Aware Training wrapper.
    - For classification: quant -> model -> dequant
    - For detection (transformers): quant input, dequant logits if present
    """

    def __init__(self, model: nn.Module, task: str = "classification"):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()
        self.task = task  # 'classification' or 'detection'

    def forward(self, x: torch.Tensor, **kwargs):
        xq = self.quant(x)
        if self.task == "detection" and _TRANSFORMERS_AVAILABLE:
            outputs = self.model(pixel_values=xq, **kwargs)
            if hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
                outputs.logits = self.dequant(outputs.logits)
            return outputs
        return self.dequant(self.model(xq))

    def fuse_model(self) -> None:
        """No-op for ViT; kept for parity with common quantization flows."""
        return


# =============================================================================
# MODEL REGISTRY
# =============================================================================
_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}
_MODEL_INFO: Dict[str, Dict] = {}


def register_model(name: str, task: str = "classification", input_size: int = 224):
    """Decorator to register a model constructor."""
    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        _MODEL_REGISTRY[name] = fn
        _MODEL_INFO[name] = {
            "task": task,
            "input_size": input_size,
            "description": (fn.__doc__ or "").strip(),
            "jetson_compatible": not ("owlv2" in name or "pruned" in name.lower()),
        }
        return fn

    return decorator


# =============================================================================
# ViT TEACHER
# =============================================================================
@register_model(name="vit_base_patch16_224_teacher", task="classification", input_size=224)
def _create_vit_base_teacher(
    pretrained: bool = True,
    num_classes: int = 10,
    checkpoint_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> nn.Module:
    """
    ViT-Base/16 teacher model (Jetson-compatible).
    - Uses timm backbone.
    - Optional HF checkpoint load (CIFAR-10 fine-tuned).
    """
    if not _TIMM_AVAILABLE:
        raise RuntimeError("timm not available - ViT models require timm library")

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        **kwargs,
    )

    if not pretrained:
        return model

    # Load weights
    state_dict = None
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
    else:
        # No trailing spaces; allow redirects
        url = (
            "https://huggingface.co/edadaltocg/"
            "vit_base_patch16_224_in21k_ft_cifar10/resolve/main/pytorch_model.bin"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            url,
            map_location="cpu",
            check_hash=False,
        )

    # If checkpoint is nested (common pattern), unwrap
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    # Strip DDP prefix if needed
    if isinstance(state_dict, dict) and state_dict:
        first_key = next(iter(state_dict.keys()))
        if first_key.startswith("module."):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    return model


# =============================================================================
# ViT STUDENT
# =============================================================================
@register_model(name="vit_small_patch16_224_student", task="classification", input_size=224)
def _create_vit_small_student(
    pretrained: bool = False,
    num_classes: int = 10,
    checkpoint_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> nn.Module:
    """
    ViT-Small/16 student model for QAT distillation (Jetson-friendly).
    - Uses timm backbone.
    - Optional checkpoint load with safe prefix stripping.
    """
    if not _TIMM_AVAILABLE:
        raise RuntimeError("timm not available - ViT models require timm library")

    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )

    if checkpoint_path is None:
        return model

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        warnings.warn(f"Checkpoint not found: {ckpt_path} - using current weights", RuntimeWarning)
        return model

    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    if isinstance(state_dict, dict) and state_dict:
        first_key = next(iter(state_dict.keys()))

        # DDP
        if first_key.startswith("module."):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            first_key = next(iter(state_dict.keys()))

        # If someone saved the wrapped QATWrapper state_dict, drop quant/dequant stubs
        if first_key.startswith(("quant.", "dequant.")):
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(("quant.", "dequant."))}

    # strict=False to tolerate head mismatch or minor key drift
    model.load_state_dict(state_dict, strict=False)
    return model


# =============================================================================
# OPTIONAL: OWL-ViT V2 (Desktop only)
# =============================================================================
if _TRANSFORMERS_AVAILABLE and not _IS_JETSON:

    @register_model(name="owlv2-base-patch16-ensemble", task="detection", input_size=960)
    def _create_owlv2_teacher(pretrained: bool = True, **kwargs) -> nn.Module:
        """OWL-ViT v2 Base teacher (desktop/DGX only)."""
        if pretrained:
            return AutoModelForObjectDetection.from_pretrained(
                "google/owlv2-base-patch16-ensemble",
                torch_dtype=torch.float32,
                **kwargs,
            )
        config = AutoConfig.from_pretrained("google/owlv2-base-patch16-ensemble")
        return AutoModelForObjectDetection.from_config(config)

    @register_model(name="owlv2-small-pruned", task="detection", input_size=768)
    def _create_owlv2_student(
        pretrained: bool = False,
        depth_ratio: float = 0.75,
        width_ratio: float = 0.75,
        head_ratio: float = 0.75,
        checkpoint_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> nn.Module:
        """Pruned OWL-ViT v2 student (train desktop, deploy via export)."""
        config = AutoConfig.from_pretrained("google/owlv2-base-patch16-ensemble")

        config.num_hidden_layers = max(6, int(config.num_hidden_layers * depth_ratio))
        config.hidden_size = max(384, int(config.hidden_size * width_ratio))
        config.num_attention_heads = max(6, int(config.num_attention_heads * head_ratio))

        # keep vision config aligned
        if hasattr(config, "vision_config"):
            config.vision_config.num_hidden_layers = config.num_hidden_layers
            config.vision_config.hidden_size = config.hidden_size
            config.vision_config.num_attention_heads = config.num_attention_heads
            config.vision_config.image_size = 768

        model = AutoModelForObjectDetection.from_config(config)

        if checkpoint_path is None:
            return model

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            warnings.warn(f"Checkpoint not found: {ckpt_path} - using random init", RuntimeWarning)
            return model

        state_dict = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

        if isinstance(state_dict, dict) and state_dict:
            first_key = next(iter(state_dict.keys()))
            if first_key.startswith("module."):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                first_key = next(iter(state_dict.keys()))
            if first_key.startswith("model."):
                state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        return model


# =============================================================================
# PUBLIC API
# =============================================================================
def create_model(
    name: str,
    pretrained: bool = True,
    num_classes: int = 10,
    checkpoint_path: Optional[Union[str, Path]] = None,
    qat_wrapper: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Platform-safe model factory.

    Raises:
        ValueError if model unavailable or incompatible on Jetson.
    """
    if name not in _MODEL_REGISTRY:
        available = [
            n
            for n in _MODEL_REGISTRY.keys()
            if (not _IS_JETSON) or _MODEL_INFO[n].get("jetson_compatible", True)
        ]
        raise ValueError(f"Model '{name}' not found. Available on {PLATFORM}: {', '.join(available)}")

    if _IS_JETSON and not _MODEL_INFO[name].get("jetson_compatible", True):
        raise ValueError(f"Model '{name}' not compatible with Jetson.")

    fn_kwargs = {"pretrained": pretrained, **kwargs}
    if _MODEL_INFO[name]["task"] == "classification":
        fn_kwargs["num_classes"] = num_classes

    model = _MODEL_REGISTRY[name](checkpoint_path=checkpoint_path, **fn_kwargs)

    if qat_wrapper:
        model = QATWrapper(model, task=_MODEL_INFO[name]["task"])

    return model


def create_teacher(
    model_family: str = "vit",
    num_classes: int = 10,
    checkpoint_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> nn.Module:
    if model_family == "vit":
        return create_model(
            "vit_base_patch16_224_teacher",
            pretrained=True,
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
    if model_family == "owlv2":
        if _IS_JETSON:
            raise ValueError("OWL-ViT not supported on Jetson.")
        if not _TRANSFORMERS_AVAILABLE:
            raise ValueError("OWL-ViT requires transformers. Install with: pip install transformers")
        return create_model(
            "owlv2-base-patch16-ensemble",
            pretrained=True,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
    raise ValueError(f"Unsupported teacher family: {model_family}")


def create_student(
    model_family: str = "vit",
    num_classes: int = 10,
    checkpoint_path: Optional[Union[str, Path]] = None,
    qat_wrapper: bool = False,
    **kwargs,
) -> nn.Module:
    if model_family == "vit":
        return create_model(
            "vit_small_patch16_224_student",
            pretrained=False,
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
            qat_wrapper=qat_wrapper,
            **kwargs,
        )
    if model_family == "owlv2":
        if _IS_JETSON:
            raise ValueError("OWL-ViT training not supported on Jetson.")
        if not _TRANSFORMERS_AVAILABLE:
            raise ValueError("OWL-ViT requires transformers.")
        return create_model(
            "owlv2-small-pruned",
            pretrained=False,
            checkpoint_path=checkpoint_path,
            qat_wrapper=qat_wrapper,
            **kwargs,
        )
    raise ValueError(f"Unsupported student family: {model_family}")


def list_available_models(jetson_only: bool = False) -> Dict[str, Dict]:
    models: Dict[str, Dict] = {}
    for name, info in _MODEL_INFO.items():
        if jetson_only and not info.get("jetson_compatible", True):
            continue
        models[name] = {
            "task": info["task"],
            "input_size": info["input_size"],
            "jetson_compatible": info.get("jetson_compatible", True),
            "description": (info["description"].splitlines()[0] if info["description"] else "No description"),
        }
    return models


def get_model_complexity(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
) -> Dict[str, Union[float, str]]:
    """
    Lightweight complexity estimate (no external deps).
    """
    params = sum(p.numel() for p in model.parameters())
    if params > 80e6:
        gflops = 17.6  # ViT-Base approx
    elif params > 20e6:
        gflops = 4.7  # ViT-Small approx
    else:
        gflops = 1.2  # Tiny approx
    return {"params_m": params / 1e6, "gflops": gflops, "platform": PLATFORM}


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(f"MODEL REGISTRY SELF-TEST ({PLATFORM.upper()} PLATFORM)")
    print("=" * 70)

    print("\nAvailable models:")
    for name, info in list_available_models(jetson_only=_IS_JETSON).items():
        compat = "✓ Jetson" if info["jetson_compatible"] else "✗ Desktop-only"
        print(f"  {compat:12s} | {name:40s} | {info['task']:15s}")

    print(f"\n1) Creating ViT-Base teacher...")
    try:
        teacher = create_teacher(model_family="vit", num_classes=10)
        c = get_model_complexity(teacher)
        print(f"   ✓ params={c['params_m']:.1f}M  flops~={c['gflops']:.1f}G")
    except Exception as e:
        print(f"   ✗ failed: {e}")

    print(f"\n2) Creating ViT-Small student (QAT wrapper)...")
    try:
        student = create_student(model_family="vit", num_classes=10, qat_wrapper=True)
        base = student.model if isinstance(student, QATWrapper) else student
        c = get_model_complexity(base)
        print(f"   ✓ params={c['params_m']:.1f}M  flops~={c['gflops']:.1f}G")
        print(f"   ✓ qat_wrapped={isinstance(student, QATWrapper)}")
    except Exception as e:
        print(f"   ✗ failed: {e}")

    print("\n3) Forward pass test...")
    try:
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y_t = teacher(x)
            y_s = student(x)
        print(f"   ✓ teacher out: {tuple(y_t.shape)}")
        if isinstance(y_s, torch.Tensor):
            print(f"   ✓ student out: {tuple(y_s.shape)}")
        else:
            print("   ✓ student out: (non-tensor output)")
    except Exception as e:
        print(f"   ✗ failed: {e}")

    print("\nDone.")
