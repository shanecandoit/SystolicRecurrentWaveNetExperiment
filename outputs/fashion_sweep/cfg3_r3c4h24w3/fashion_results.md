# Fashion-MNIST SRWN vs CNN Results

Generated: 2026-05-01 13:09:31

## Configuration

| Parameter | Value |
|---|---|
| seed | 7 |
| data_dir | outputs/fashion_data |
| epochs | 4 |
| batch_size | 128 |
| train_size | 8000 |
| val_size | 1500 |
| test_size | 3000 |
| rows | 3 |
| cols | 4 |
| hidden_dim | 24 |
| train_waves | 3 |
| eval_waves | 3 |
| learning_rate | 0.001 |
| conf_threshold | 0.9 |
| device | cuda |

## Final Comparison

| Model | Test accuracy | Params | Estimated MACs/sample |
|---|---:|---:|---:|
| SRWN fixed | 0.7840 | 40936 | 83904.00 |
| SRWN adaptive | 0.7803 | 40936 | 83904.00 |
| CNN baseline | 0.8423 | 206922 | 1218048.00 |

## Plot files

- outputs/fashion_train_loss.png
- outputs/fashion_val_accuracy.png
- outputs/fashion_accuracy_vs_compute.png
