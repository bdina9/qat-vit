# FasterTransformer ViT-Quantization Extraction → Standalone Repo Workflow

This repo is extracted from:

`FasterTransformer/examples/pytorch/vit/ViT-quantization`

Goal: work on KD + QAT ViT INT8 training without carrying the full FasterTransformer tree.

---

## 0) What happened / current state

We extracted the ViT-quantization subproject from FasterTransformer using `git subtree split`, and pushed it into:

- Repo: `https://github.com/bdina9/qat-vit`
- Branch: `ft-vtq-extract`

This branch contains the extracted project at repo root.

---

## 1) Extraction commands (done once)

From FasterTransformer root:

```bash
cd ~/FasterTransformer

# split subproject into standalone history
git subtree split -P examples/pytorch/vit/ViT-quantization -b vit-quantization-only

# push extracted branch to standalone repo
git remote add qatvit https://github.com/bdina9/qat-vit.git
git push -u qatvit vit-quantization-only:ft-vtq-extract

