# Systolic Recurrent Wave Network

This repository evaluates a Systolic Recurrent Wave Network (SRWN) as an anytime model: can iterative wave refinement produce useful predictions early, reduce compute with adaptive halting, and stay competitive with conventional networks?

## Story in one page

### Research questions

1. Does recurrence in SRWN actually add useful refinement?
2. Can confidence-based halting reduce SRWN compute without large quality loss?
3. On image classification, where is the SRWN vs CNN frontier?
4. Can SRWN beat CNN on both accuracy and compute in the tested budget?

### What was run

1. Parity ablation: recurrent SRWN vs ablated SRWN ([srwn_experiment.py](srwn_experiment.py)).
2. Parity benchmark: SRWN variants vs MLP with plots ([srwn_benchmark.py](srwn_benchmark.py)).
3. Fashion-MNIST baseline: SRWN vs 2-layer CNN ([fashion_mnist_benchmark.py](fashion_mnist_benchmark.py)).
4. Fashion-MNIST v2 sweep: deeper/wider SRWN variants vs CNN ([fashion_mnist_v2.py](fashion_mnist_v2.py), [fashion-mnist-v2.py](fashion-mnist-v2.py)).

### Headline findings

1. Recurrence is functionally real on parity: early columns improve with waves in recurrent SRWN and stay flat in ablated SRWN.
2. Adaptive halting reduces SRWN compute:
- parity: about 68% reduction (fixed -> adaptive)
- Fashion-MNIST baseline: about 13% reduction
3. On Fashion-MNIST, SRWN is much cheaper in MACs but less accurate than the tested CNN.
4. In the v2 deeper/wider sweep, no tested SRWN variant beats CNN on both accuracy and MACs.
5. Width improves SRWN accuracy more reliably than depth in the tested range.

## What each experiment shows

### Experiment 1: Parity ablation

What it answers:
- Whether recurrence is doing real work or the network is mostly feed-forward.

What it shows:
- Recurrent early-column gains are large.
- Ablated early-column gains are near zero.
- Core recurrence mechanism is validated on synthetic data.

Primary caveat:
- Delta convergence signal does not show clear decay, so fixed-point convergence remains unproven.

### Experiment 2: Parity benchmark (with MLP)

What it answers:
- Whether SRWN competitiveness includes compute efficiency, not only accuracy.

What it shows:
- SRWN recurrent reaches top parity accuracy.
- Adaptive halting retains quality while lowering SRWN compute.
- MLP remains substantially cheaper on this synthetic task.

Primary caveat:
- Parity is an unrealistically clean benchmark and weak evidence for image-domain competitiveness.

### Experiment 3: Fashion-MNIST baseline

What it answers:
- How SRWN compares with a standard CNN on real image data.

What it shows:
- CNN wins on accuracy.
- SRWN wins on compute (MACs).
- This is a tradeoff frontier, not a dominance result.

Primary caveat:
- Short training budget and single SRWN configuration limit conclusions.

### Experiment 4: Fashion-MNIST v2 deeper/wider sweep

What it answers:
- Whether depth/width scaling can close the SRWN-CNN gap.

What it shows:
- Wider SRWN variants improve more than deeper variants.
- Best tested SRWN remains below CNN accuracy while staying far cheaper.
- In tested settings, SRWN does not beat CNN on both metrics.

Primary caveat:
- Search is not exhaustive (rows/cols/waves/lr space remains broad).

## Current conclusions

### Supported by current evidence

1. SRWN recurrence is meaningful for refinement on synthetic parity.
2. Adaptive halting can reduce SRWN compute while preserving most accuracy.
3. SRWN currently occupies a low-compute, lower-accuracy region relative to tested CNN baselines on Fashion-MNIST.
4. Width is currently the higher-leverage SRWN scaling axis than depth.

### Not yet demonstrated

1. SRWN beating CNN on both accuracy and compute in the tested Fashion-MNIST budget.
2. True fixed-point-like convergence of SRWN wave dynamics.
3. Wall-clock latency advantage (MAC advantage is analytical, not profiled latency).
4. Robust halting benefits on highly heterogeneous real-world difficulty distributions.

## Work still required

Priority 1:
1. Run SRWN and CNN to comparable convergence criteria, not just fixed epochs.
2. Profile wall-clock latency and memory behavior, not only MACs.
3. Expand SRWN sweep across columns, waves, and learning rates.

Priority 2:
1. Add stronger baselines (deeper CNN, lightweight ViT, or efficient conv models).
2. Add halting analysis by difficulty buckets (easy/medium/hard samples).
3. Test alternate SRWN topologies (for example, 1D chain or bidirectional recurrence).

Priority 3:
1. Evaluate on a harder dataset (for example, CIFAR-10) with the same reporting structure.
2. Optimize a joint metric (accuracy at fixed compute budget) instead of accuracy alone.

## Where to read full evidence

1. Consolidated narrative and plot-backed conclusions: [outputs/results.md](outputs/results.md)
2. Parity ablation artifacts: [outputs/results.json](outputs/results.json)
3. Parity benchmark artifacts: [outputs/benchmark_results.md](outputs/benchmark_results.md), [outputs/benchmark_metrics.json](outputs/benchmark_metrics.json)
4. Fashion baseline artifacts: [outputs/fashion_results.md](outputs/fashion_results.md), [outputs/fashion_metrics.json](outputs/fashion_metrics.json)
5. Fashion v2 artifacts: [outputs/fashion_v2_results.md](outputs/fashion_v2_results.md), [outputs/fashion_v2_metrics.json](outputs/fashion_v2_metrics.json), [outputs/fashion_v2_verdict.json](outputs/fashion_v2_verdict.json)

## Plot index

1. Parity benchmark plots:
- [outputs/train_loss.png](outputs/train_loss.png)
- [outputs/val_accuracy.png](outputs/val_accuracy.png)
- [outputs/accuracy_vs_compute.png](outputs/accuracy_vs_compute.png)

2. Fashion baseline plots:
- [outputs/fashion_train_loss.png](outputs/fashion_train_loss.png)
- [outputs/fashion_val_accuracy.png](outputs/fashion_val_accuracy.png)
- [outputs/fashion_accuracy_vs_compute.png](outputs/fashion_accuracy_vs_compute.png)

3. Fashion v2 plots:
- [outputs/fashion_v2_accuracy_bar.png](outputs/fashion_v2_accuracy_bar.png)
- [outputs/fashion_v2_compute_bar.png](outputs/fashion_v2_compute_bar.png)
- [outputs/fashion_v2_acc_vs_macs.png](outputs/fashion_v2_acc_vs_macs.png)

## Run commands

1. Parity ablation:

```bash
D:/apps/Python39/python.exe srwn_experiment.py --json
```

2. Parity benchmark:

```bash
D:/apps/Python39/python.exe srwn_benchmark.py --epochs 80 --out-dir outputs
```

3. Fashion baseline:

```bash
D:/apps/Python39/python.exe fashion_mnist_benchmark.py --out-dir outputs
```

4. Fashion v2 sweep:

```bash
D:/apps/Python39/python.exe fashion-mnist-v2.py --out-dir outputs --epochs 8
```
