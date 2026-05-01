# Systolic Recurrent Wave Network

Systolic Recurrent Wave Network (SRWN) is a grid-based recurrent model aimed at anytime inference. This repository contains runnable experiments that evaluate SRWN behavior on both synthetic and image benchmarks.

## Project files

- `srwn_experiment.py`: parity ablation (recurrent vs ablated SRWN)
- `srwn_benchmark.py`: parity benchmark with MLP baseline and plots
- `fashion_mnist_benchmark.py`: Fashion-MNIST benchmark with CNN baseline and plots
- `design.md`: architecture assumptions and validation criteria
- `experiments.md`: experiment plan and rationale
- `outputs/results.md`: consolidated analysis report

## Experiment summary

### 1) Parity (recurrent mechanism validation)

Latest key outcomes (from `outputs/results.md`):

- Recurrent SRWN early exits improve strongly across waves.
- Ablated SRWN early exits stay flat across waves.
- Recurrent wave path is validated as useful for refinement.
- Delta-based convergence criterion is not yet validated.

### 2) Parity baseline comparison (SRWN vs MLP)

Latest key outcomes:

- SRWN recurrent achieved the best test accuracy on parity.
- Adaptive SRWN significantly reduced SRWN compute vs fixed-wave SRWN.
- MLP remained much cheaper in MACs than SRWN under that configuration.

### 3) Fashion-MNIST comparison (SRWN vs CNN)

Latest run produced:

- SRWN fixed test accuracy: 0.8298
- SRWN adaptive test accuracy: 0.8212
- CNN test accuracy: 0.8706
- SRWN fixed MACs/sample: 83904
- SRWN adaptive MACs/sample: 72678
- CNN MACs/sample: 1218048

Interpretation:

- CNN currently wins on accuracy.
- SRWN currently wins on compute cost.
- This is a clear compute-accuracy tradeoff rather than a dominance result.

## Compute metric: MACs

**MAC** stands for **Multiply-Accumulate Operation** — a single multiplication followed by an addition. It is the standard hardware-level unit for measuring neural network inference cost, because every linear layer, convolution, and recurrent connection bottoms out to a count of MACs.

For a linear layer mapping `in` inputs to `out` outputs, the cost is exactly `in × out` MACs (one multiply-add per output neuron per input element). The total inference cost of a model is the sum of MACs across all its layers.

### How MACs are counted in this codebase

**SRWN per sample** (from `srwn_benchmark.py:srwn_macs` and `fashion_mnist_benchmark.py:srwn_macs`):

```
input_projection = cols × chunk_dim × hidden_dim        # one linear per column
per_node         = 3 × hidden_dim × hidden_dim           # top + left + right linears in each GridCell
node_cost        = rows × cols × per_node                # all cells in one wave sweep
head_cost        = cols × hidden_dim × N_CLASSES         # one classifier head per column
per_wave         = node_cost + head_cost

fixed_MACs    = input_projection + eval_waves × per_wave
adaptive_MACs = input_projection + mean_halt_wave × per_wave
```

`fixed_MACs` is the cost when every wave always runs to completion. `adaptive_MACs` is the expected cost when the confidence-threshold halting policy is applied — it uses `mean_halt_wave`, the average wave at which a sample exited, measured on the test set.

**CNN baseline per sample** (from `fashion_mnist_benchmark.py:cnn_macs`):

```
conv1 = 28 × 28 × 16 × (1 × 3 × 3)       # first conv: 16 filters, 3×3 kernel, 1 input channel
conv2 = 14 × 14 × 32 × (16 × 3 × 3)      # second conv after max-pool: 32 filters, 16 input channels
fc1   = (32 × 7 × 7) × 128
fc2   = 128 × N_CLASSES
```

**MLP baseline per sample** (from `srwn_benchmark.py:mlp_macs`):

```
macs = input_bits × h + h × h + h × N_CLASSES    # three fully-connected layers
```

MACs are an analytical count, not a profiled runtime. They measure raw multiply-add work and ignore memory bandwidth, parallelism, and hardware-specific effects. The key use here is comparing relative compute cost across architectures on the same task, particularly to evaluate SRWN's adaptive halting benefit.

## How to run experiments

### A) Core parity ablation

```bash
D:/apps/Python39/python.exe srwn_experiment.py --json
```

Generates:

- `outputs/results.json`
- `outputs/results.md`

### B) Parity baseline benchmark with plots

```bash
D:/apps/Python39/python.exe srwn_benchmark.py --epochs 80 --out-dir outputs
```

Generates:

- `outputs/benchmark_metrics.json`
- `outputs/benchmark_results.md`
- `outputs/train_loss.png`
- `outputs/val_accuracy.png`
- `outputs/accuracy_vs_compute.png`

### C) Fashion-MNIST benchmark with CNN baseline

```bash
D:/apps/Python39/python.exe fashion_mnist_benchmark.py --out-dir outputs
```

Generates:

- `outputs/fashion_metrics.json`
- `outputs/fashion_results.md`
- `outputs/fashion_train_loss.png`
- `outputs/fashion_val_accuracy.png`
- `outputs/fashion_accuracy_vs_compute.png`

## Current conclusions

1. SRWN recurrence is functionally meaningful on the synthetic parity test.
2. SRWN can reduce its own compute with confidence-based adaptive halting.
3. On Fashion-MNIST, current SRWN configuration is compute-efficient but less accurate than a conventional CNN baseline.
4. The next optimization target is to improve SRWN Fashion-MNIST accuracy while preserving its compute advantage.
