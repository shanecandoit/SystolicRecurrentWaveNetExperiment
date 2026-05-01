# Systolic Recurrent Wave Network

Systolic Recurrent Wave Network (SRWN) is a grid-based recurrent model aimed at anytime inference. This repository contains runnable experiments that evaluate SRWN behavior on both synthetic and image benchmarks.

## Project files

- `srwn_experiment.py`: parity ablation (recurrent vs ablated SRWN)
- `srwn_benchmark.py`: parity benchmark with MLP baseline and plots
- `fashion_mnist_benchmark.py`: Fashion-MNIST benchmark with CNN baseline and plots
- `fashion_mnist_v2.py` and `fashion-mnist-v2.py`: deeper/wider SRWN sweep against CNN baseline
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

Latest baseline run produced:

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

### 4) Fashion-MNIST v2 (deeper and wider SRWN)

Latest v2 sweep produced:

- CNN test accuracy: 0.8700
- Best SRWN accuracy: 0.8532 (wider_r3_h48_w3)
- SRWN adaptive MACs range: 65826.89 to 202217.24
- CNN MACs/sample: 1218048.00

Crossover answer:

- In this tested v2 sweep, no SRWN variant beat CNN on both accuracy and MACs at the same time.
- SRWN remained much cheaper; CNN remained more accurate.

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

### D) Fashion-MNIST v2 deeper/wider sweep

```bash
D:/apps/Python39/python.exe fashion-mnist-v2.py --out-dir outputs --epochs 8
```

Generates:

- `outputs/fashion_v2_metrics.json`
- `outputs/fashion_v2_verdict.json`
- `outputs/fashion_v2_results.md`
- `outputs/fashion_v2_accuracy_bar.png`
- `outputs/fashion_v2_compute_bar.png`
- `outputs/fashion_v2_acc_vs_macs.png`

## Current conclusions

### Proven (high confidence)

1. **SRWN recurrence refines early columns on synthetic parity**: Recurrent early-column accuracy improves by ~48% across waves (col0: 0.5186→1.0), while ablated SRWN stays flat (col0: 0.5186→0.5186). This validates the core recurrent mechanism works as intended on parity (but see caveats below).
2. **Adaptive halting reduces SRWN compute**: Confidence-threshold halting cut SRWN compute by 68% on parity (21216→6745 MACs) and 13% on Fashion-MNIST (83904→72678 MACs) with minimal accuracy loss. This demonstrates anytime inference is viable when inputs have mixed difficulty.
3. **SRWN is 6-18x cheaper than CNN in MACs**: Across all Fashion-MNIST experiments, SRWN variants ranged 66k-202k MACs vs CNN's 1.22M MACs. This is a real efficiency advantage. However, MACs are analytical counts; actual wall-clock latency requires hardware profiling.
4. **Wider SRWN > Deeper SRWN for accuracy**: In the v2 sweep, hidden_dim scaling improved accuracy more (+2.4% from 24→48) than row scaling (+1% from 3→5 rows). This guides future hyperparameter searches toward width before depth.

### Unproven (limitations, caveats)

5. **SRWN does not beat CNN on both accuracy and compute**: Best tested SRWN accuracy (0.8532 on Fashion-MNIST v2) trails CNN (0.8700). SRWN is cheaper but less accurate. The experiment doesn't show when SRWN would be preferable—only a compute-accuracy tradeoff, not a dominant design. See [outputs/results.md](outputs/results.md#comprehensive-conclusions) for detailed per-experiment strength/weakness analysis.

6. **Parity is too clean to validate real inference**: XOR and 3-bit parity are noise-free, nondeterministic tasks. The recurrence mechanism that works on parity may not transfer to noisy images or ambiguous labels. Early conclusions (Exp 1-2) should be weighted lightly when considering image benchmarks.

7. **CNN baselines are weak**: All CNN comparisons use a simple 2-layer CNN (reaching 0.87 on Fashion-MNIST). Modern architectures (ResNet, EfficientNet, Vision Transformer) reach 95%+ on the same task. "SRWN is cheaper than a weak CNN" is different from "SRWN is cheaper than a competitive CNN."

8. **Delta convergence criterion fails**: SRWN was designed to refine toward a fixed point, but delta (wave-to-wave refinement magnitude) stays high (~2.0) rather than decaying. This suggests SRWN may not be converging to an attractor. The theory expects convergence; the data doesn't show it.

### Recommended next steps

- **Validate wall-clock latency**: Profile SRWN and CNN on GPU/TPU to see if lower MACs translates to lower latency. Grid ops might not parallelize as well as convolutions.
- **Test halting on high variance inputs**: Current halting tests use uniform-difficulty datasets (parity, Fashion-MNIST). Test on datasets with easy and hard samples to measure the true efficiency gain.
- **Run to convergence, not fixed epochs**: Experiments ran 6-8 epochs. Run both SRWN and CNN to same validation loss threshold to see final accuracy gaps without artificial epoch limits.
- **Explore 1D recurrence**: Test whether a simpler 1D recurrent chain (depth only) would achieve 80% of SRWN's benefit at lower cost. The 2D grid topology is assumed, not validated as optimal.

Full detailed analysis with plot interpretations: [outputs/results.md](outputs/results.md#comprehensive-conclusions)
