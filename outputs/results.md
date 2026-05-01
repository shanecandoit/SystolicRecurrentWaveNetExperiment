# SRWN Results: Consolidated Narrative

## Executive conclusion

SRWN demonstrates real recurrent refinement behavior and meaningful internal compute reduction via adaptive halting. On Fashion-MNIST, tested SRWN variants are consistently much cheaper than the tested CNN baseline in MACs but remain less accurate. In the current experiment budget, no SRWN variant beats CNN on both accuracy and compute at the same time.

## The story arc

1. First validate mechanism on synthetic data (parity).
2. Then benchmark that mechanism against a simple baseline (MLP).
3. Then move to image data (Fashion-MNIST) against a CNN baseline.
4. Then test whether scaling SRWN depth/width closes the gap.

Result: the mechanism works, the efficiency lever works, but the quality frontier still favors CNN.

## Experiment-by-experiment evidence

## Experiment 1: Parity ablation (recurrent vs ablated SRWN)

Question:
- Does recurrence provide measurable refinement beyond a feed-forward-like ablated variant?

What was run:
- [srwn_experiment.py](../srwn_experiment.py)
- Metrics output: [outputs/results.json](results.json)

Key evidence:
- Recurrent early-column gains are large:
- col0 gain: +0.4814
- col1 gain: +0.5005
- Ablated gains are flat:
- col0 gain: +0.0000
- col1 gain: +0.0000
- Rightmost final accuracy reaches 1.0000 in recurrent run.

What this experiment shows:
- Recurrence in SRWN is functionally active and contributes to refinement.

What it does not show:
- Delta convergence remains unvalidated (delta does not clearly decay across late waves).

## Experiment 2: Parity benchmark (SRWN vs MLP)

Question:
- Is SRWN not only accurate but also compute-competitive on a simple benchmark?

What was run:
- [srwn_benchmark.py](../srwn_benchmark.py)
- Metrics/report: [outputs/benchmark_metrics.json](benchmark_metrics.json), [outputs/benchmark_results.md](benchmark_results.md)
- Plots: [outputs/train_loss.png](train_loss.png), [outputs/val_accuracy.png](val_accuracy.png), [outputs/accuracy_vs_compute.png](accuracy_vs_compute.png)

Key evidence:
- SRWN recurrent fixed: 1.0000 accuracy, 21216 MACs/sample
- SRWN recurrent adaptive: 1.0000 accuracy, 6745.10 MACs/sample
- SRWN ablated fixed: 0.9951 accuracy, 21216 MACs/sample
- MLP baseline: 0.9961 accuracy, 816 MACs/sample

What this experiment shows:
- Recurrence improves SRWN quality over ablated SRWN at matched fixed compute.
- Adaptive halting can sharply reduce SRWN compute while preserving quality.

What it does not show:
- SRWN compute competitiveness on parity: MLP remains far cheaper.

## Experiment 3: Fashion-MNIST baseline (SRWN vs CNN)

Question:
- Does SRWN preserve its value proposition on real image classification?

What was run:
- [fashion_mnist_benchmark.py](../fashion_mnist_benchmark.py)
- Metrics/report: [outputs/fashion_metrics.json](fashion_metrics.json), [outputs/fashion_results.md](fashion_results.md)
- Plots: [outputs/fashion_train_loss.png](fashion_train_loss.png), [outputs/fashion_val_accuracy.png](fashion_val_accuracy.png), [outputs/fashion_accuracy_vs_compute.png](fashion_accuracy_vs_compute.png)

Key evidence:
- SRWN fixed: 0.8298 accuracy, 83904 MACs/sample
- SRWN adaptive: 0.8212 accuracy, 72678.49 MACs/sample
- CNN baseline: 0.8706 accuracy, 1218048 MACs/sample

What this experiment shows:
- CNN is more accurate in this setup.
- SRWN is far cheaper in analytical MACs.
- Adaptive halting still provides moderate compute reduction on image data.

What it does not show:
- Final best-possible gap under convergence training (run was short and single-config for SRWN).

## Experiment 4: Fashion-MNIST v2 sweep (deeper and wider SRWN)

Question:
- Can deeper/wider SRWN close the accuracy gap while staying below CNN compute?

What was run:
- [fashion_mnist_v2.py](../fashion_mnist_v2.py), [fashion-mnist-v2.py](../fashion-mnist-v2.py)
- Metrics/report: [outputs/fashion_v2_metrics.json](fashion_v2_metrics.json), [outputs/fashion_v2_results.md](fashion_v2_results.md), [outputs/fashion_v2_verdict.json](fashion_v2_verdict.json)
- Plots: [outputs/fashion_v2_accuracy_bar.png](fashion_v2_accuracy_bar.png), [outputs/fashion_v2_compute_bar.png](fashion_v2_compute_bar.png), [outputs/fashion_v2_acc_vs_macs.png](fashion_v2_acc_vs_macs.png)

Key evidence:
- Best SRWN accuracy: 0.8532 (wider_r3_h48_w3)
- CNN accuracy: 0.8700
- SRWN adaptive MAC range: 65826.89 to 202217.24
- CNN MACs/sample: 1218048

What this experiment shows:
- Width improved SRWN more consistently than depth in tested ranges.
- No tested SRWN point surpassed CNN on both axes simultaneously.

What it does not show:
- Whether untested regions of SRWN hyperparameter space can close the gap.

## Cross-experiment synthesis

### Supported conclusions

1. Mechanism claim is supported: SRWN recurrence contributes meaningful refinement.
2. Efficiency claim is supported in-model: adaptive halting reduces SRWN compute.
3. System-level frontier claim is supported: SRWN currently occupies lower-compute, lower-accuracy space relative to tested CNN.
4. Scaling-direction claim is supported in tested space: width is currently higher leverage than depth.

### Not yet supported

1. Dominance claim is not supported: SRWN does not beat CNN on both accuracy and MACs in current tests.
2. Convergence claim is not supported: fixed-point-like wave convergence remains unproven.
3. Runtime advantage claim is not supported: MAC counts are analytical and not a replacement for latency profiling.

## Plot-backed interpretation guide

1. [outputs/accuracy_vs_compute.png](accuracy_vs_compute.png): parity points separate mostly by compute, showing adaptive halting benefit and MLP compute dominance.
2. [outputs/fashion_accuracy_vs_compute.png](fashion_accuracy_vs_compute.png): baseline image tradeoff, SRWN left/lower and CNN right/higher.
3. [outputs/fashion_v2_acc_vs_macs.png](fashion_v2_acc_vs_macs.png): v2 frontier view, no SRWN point enters better-accuracy/lower-compute region relative to CNN.
4. [outputs/fashion_v2_accuracy_bar.png](fashion_v2_accuracy_bar.png): width helps SRWN more than depth in tested settings.
5. [outputs/fashion_v2_compute_bar.png](fashion_v2_compute_bar.png): all tested SRWN variants remain far below CNN MAC budget.

## What work is still required

### Priority 1: close methodological gaps

1. Train to comparable convergence criteria instead of fixed short epoch budgets.
2. Profile wall-clock latency, throughput, and memory on target hardware.
3. Expand SRWN search space (cols, waves, learning rate, regularization), not only rows/hidden size.

### Priority 2: strengthen comparisons

1. Add stronger baselines (larger CNN, efficient conv model, lightweight transformer).
2. Run matched-capacity and matched-compute comparisons, not only one baseline point.
3. Add confidence calibration and halting-by-difficulty analysis.

### Priority 3: validate generality

1. Test on a harder dataset with the same reporting schema.
2. Evaluate alternative SRWN topologies and recurrence directions.
3. Optimize a joint objective (accuracy at fixed compute budget) for model selection.

## Bottom line

SRWN is promising as an anytime, compute-aware architecture, but current evidence supports it as a tradeoff option, not a replacement for CNNs on Fashion-MNIST. The next phase should focus on convergence-controlled training, runtime profiling, and broader architecture/baseline sweeps to determine whether that tradeoff frontier can move.