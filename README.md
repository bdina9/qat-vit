# project/README.md

## Overview
Teacher→Student distillation with optional QAT scaffolding:
- Teacher: `vit_base_patch16_224_teacher`
- Student: `vit_small_patch16_224_student` (+ optional `QATWrapper`)

Outputs:
- `best_params.yaml` from Optuna search
- `best_qat.pth` (trained QAT/fake-quant weights)
- `best_converted.pth` (eager-mode converted quant weights; CPU-oriented)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
