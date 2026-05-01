# Fashion-MNIST v2: Deeper and Wider SRWN Sweep

Generated: 2026-05-01 13:32:01

## CNN baseline

- Test accuracy: 0.8700
- Estimated MACs/sample: 1218048.00

## SRWN variant table

| Name | Family | Rows | Cols | Hidden | Eval waves | Test acc | Halted acc | Mean halt wave | Adaptive MACs | Params |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_r3_h24_w3 | baseline | 3 | 4 | 24 | 3 | 0.8328 | 0.8258 | 2.1668 | 65826.89 | 40936 |
| deeper_r4_h24_w4 | deeper | 4 | 4 | 24 | 4 | 0.8366 | 0.8390 | 2.7530 | 97573.82 | 47944 |
| deeper_r5_h24_w4 | deeper | 5 | 4 | 24 | 4 | 0.8256 | 0.8274 | 3.1848 | 131940.10 | 54952 |
| wider_r3_h32_w3 | wider | 3 | 4 | 32 | 3 | 0.8424 | 0.8398 | 2.0462 | 103138.25 | 63784 |
| wider_r3_h48_w3 | wider | 3 | 4 | 48 | 3 | 0.8532 | 0.8452 | 1.9394 | 202217.24 | 123304 |

## Verdict: when do we beat CNN on both accuracy and MACs?

No tested deeper/wider SRWN variant beat CNN on both accuracy and MACs in this v2 sweep.

Best SRWN by accuracy: wider_r3_h48_w3 (0.8532).
Best SRWN by accuracy-per-MAC: baseline_r3_h24_w3.

## Plot files

- outputs/fashion_v2_accuracy_bar.png
- outputs/fashion_v2_compute_bar.png
- outputs/fashion_v2_acc_vs_macs.png
