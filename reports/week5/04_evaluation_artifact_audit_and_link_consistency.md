[← Back to Week 5 Index](../week5_evaluation_technical_debt.md)

# Evaluation Artifact Audit and Link Consistency

## 1. Why this audit is needed

Week 5 now has calibration and abstention experiments that generate figures and JSONL registry files. Without an audit layer, evaluation evidence can become scattered or accidentally mixed with source-controlled files.

## 2. Source-controlled files versus local artifacts

Source-controlled files include:

- source code
- tests
- experiment scripts
- configs
- reports
- README

Local-only artifacts include:

- checkpoints
- generated figures
- registry JSONL outputs
- canvas debug figures
- user-drawn canvas samples

## 3. Audit utilities implemented

Task 7D adds lightweight utilities for:

- directory artifact summaries
- required `.gitignore` pattern audit
- Markdown relative-link audit
- report markdown collection

## 4. Required ignored paths

- `results/figures/*.png`
- `results/checkpoints/*.npz`
- `results/canvas_debug/`
- `results/registry/`
- `data/user_digits/`

## 5. Link consistency

Reports should remain navigable. Week 5 audit checks relative `.md` links under README and `reports/`.

## 6. What this task intentionally does not do

This task intentionally does not include:

- no training
- no calibration correction
- no abstention policy deployment
- no MNIST/CNN
- no modification of generated artifacts
- no moving local canvas samples

## 7. How this supports later tasks

Before temperature scaling or future calibration correction, the project should keep evaluation outputs traceable and prevent accidental artifact commits.

## 8. Next step

Next possible task:

- temperature scaling on a proper validation split, or
- a broader technical-debt cleanup after the audit output is inspected

[← Back to Week 5 Index](../week5_evaluation_technical_debt.md)
