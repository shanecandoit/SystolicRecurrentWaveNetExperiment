# SRWN Baseline Comparison

Generated: 2026-05-01 12:24:17

## What was run

- Trained SRWN recurrent, SRWN ablated, and MLP baseline on parity data.
- Logged train loss and validation accuracy by epoch.
- Evaluated test accuracy and estimated MACs.

## Final comparison

| Model | Test metric | Params | Estimated MACs/sample |
|---|---|---|---|
| SRWN recurrent fixed | acc=1.0000 | 5576 | 21216.00 |
| SRWN recurrent adaptive | acc=1.0000 | 5576 | 6745.10 |
| SRWN ablated fixed | acc=0.9951 | 5576 | 21216.00 |
| MLP baseline | acc=0.9961 | 866 | 816.00 |

## How to interpret

1. SRWN adaptive mode matches or beats MLP accuracy on this task.
2. SRWN adaptive mode is still compute-heavier than MLP with this configuration.
3. Recurrent SRWN beats ablated SRWN, so recurrence contributes to final quality.

## Plot files

- outputs/train_loss.png
- outputs/val_accuracy.png
- outputs/accuracy_vs_compute.png
