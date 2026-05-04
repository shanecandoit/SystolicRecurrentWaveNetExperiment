# Fashion-MNIST v4: Wide SRWN with Dropout

Generated: 2026-05-03 09:53:07

## Goal

Test whether dropout regularization (default p=0.1) improves wide-SRWN generalization and closes the CNN accuracy gap.

## CNN baseline

- Test accuracy: 0.8906
- Estimated MACs/sample: 1218048.00

## SRWN variant table

| Name | Hidden | Dropout | Test acc | Halted acc | Mean halt wave | Adaptive MACs | Params | Beats CNN acc? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| wide_d01_r3_h64_w3 | 64 | 0.10 | 0.8598 | 0.8534 | 1.7224 | 308563.56 | 201256 | no |
| wide_d01_r3_h96_w3 | 96 | 0.10 | 0.8608 | 0.8552 | 1.5956 | 610772.89 | 412456 | no |
| wide_d01_r3_h128_w3 | 128 | 0.10 | 0.8554 | 0.8532 | 1.8124 | 1178628.51 | 697384 | no |
| wide_d01_r3_h160_w3 | 160 | 0.10 | 0.8536 | 0.8510 | 1.5082 | 1525049.60 | 1056040 | no |
| wide_d01_r3_h192_w3 | 192 | 0.10 | 0.8582 | 0.8550 | 1.5544 | 2225316.25 | 1488424 | no |

## Key findings

No SRWN variant reached CNN accuracy (0.8906).
Best SRWN: wide_d01_r3_h96_w3 at 0.8608 (gap 0.0298).

Best SRWN by accuracy: wide_d01_r3_h96_w3 (0.8608)
Best SRWN by accuracy-per-MAC: wide_d01_r3_h64_w3

## Plot files

- outputs/fashion_v4_accuracy_bar.png
- outputs/fashion_v4_compute_bar.png
- outputs/fashion_v4_acc_vs_macs.png
