# Fashion-MNIST v3: Wide SRWN Push to Match CNN Accuracy

Generated: 2026-05-01 18:35:27

## Goal

Push SRWN to maximum width (hidden_dim) to find accuracy ceiling and determine if pure accuracy can match CNN.

## CNN baseline

- Test accuracy: 0.8872
- Estimated MACs/sample: 1218048.00

## SRWN variant table (all h64→h192, rows=3, cols=4, waves=3)

| Name | Hidden | Test acc | Halted acc | Mean halt wave | Adaptive MACs | Params | Beats CNN acc? |
|---|---:|---:|---:|---:|---:|---:|---|
| wide_r3_h64_w3 | 64 | 0.8582 | 0.8582 | 1.6882 | 303433.01 | 201256 | ✗ no |
| wide_r3_h96_w3 | 96 | 0.8552 | 0.8506 | 1.8248 | 687696.08 | 412456 | ✗ no |
| wide_r3_h128_w3 | 128 | 0.8518 | 0.8512 | 1.6028 | 1053928.24 | 697384 | ✗ no |
| wide_r3_h160_w3 | 160 | 0.8540 | 0.8530 | 1.4448 | 1466214.40 | 1056040 | ✗ no |
| wide_r3_h192_w3 | 192 | 0.8624 | 0.8566 | 1.5256 | 2186874.47 | 1488424 | ✗ no |

## Key findings

✗ No SRWN variant reached CNN accuracy (0.8872).
   Best achieved: wide_r3_h192_w3 at 0.8624 (gap: 0.0248)

Best SRWN by accuracy: wide_r3_h192_w3 (0.8624)
Best SRWN by accuracy-per-MAC: wide_r3_h64_w3

## Plot files

- outputs/fashion_v3_accuracy_bar.png
- outputs/fashion_v3_compute_bar.png
- outputs/fashion_v3_acc_vs_macs.png
