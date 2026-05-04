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

## Experiment 5: Fashion-MNIST v3 wide push (SRWN accuracy ceiling)

Question:
- Can very wide SRWN (hidden_dim h64→h192) match or exceed CNN accuracy (0.8872)?
- What is the maximum accuracy SRWN can achieve in this configuration?

What was run:
- [fashion_mnist_v3.py](../fashion_mnist_v3.py), [fashion-mnist-v3.py](../fashion-mnist-v3.py)
- Metrics/report: [outputs/fashion_v3_metrics.json](fashion_v3_metrics.json), [outputs/fashion_v3_results.md](fashion_v3_results.md), [outputs/fashion_v3_verdict.json](fashion_v3_verdict.json)
- Plots: [outputs/fashion_v3_accuracy_bar.png](fashion_v3_accuracy_bar.png), [outputs/fashion_v3_compute_bar.png](fashion_v3_compute_bar.png), [outputs/fashion_v3_acc_vs_macs.png](fashion_v3_acc_vs_macs.png)
- Training: 25 epochs, all variants use rows=3, cols=4, waves=3, only hidden_dim varies

Key evidence:

| Variant | Hidden dim | Test acc | Adaptive MACs | Params | Beats CNN? |
|---|---:|---:|---:|---:|---|
| wide_r3_h64_w3 | 64 | 0.8582 | 303433 | 201256 | ✗ no |
| wide_r3_h96_w3 | 96 | 0.8552 | 687696 | 412456 | ✗ no |
| wide_r3_h128_w3 | 128 | 0.8518 | 1053928 | 697384 | ✗ no |
| wide_r3_h160_w3 | 160 | 0.8540 | 1466214 | 1056040 | ✗ no |
| wide_r3_h192_w3 | 192 | 0.8624 | 2186874 | 1488424 | ✗ no |
| CNN baseline | — | 0.8872 | 1218048 | 206922 | reference |

Best SRWN accuracy: wide_r3_h192_w3 at 0.8624 — gap to CNN is 0.0248.

What this experiment shows:
- Even extremely wide SRWN (h192, 1.5M parameters) fails to close the accuracy gap to CNN.
- Accuracy trend is non-monotonic with width: h64 (0.8582) outperforms h96 (0.8552) and h128 (0.8518) before recovering at h160 and h192. This is a classic sign of underfitting/overfitting competition — medium widths have enough capacity to overfit but not enough to generalize better than smaller models at 25 epochs.
- At h128 and above, SRWN MACs are comparable to or higher than CNN, meaning the compute-efficiency argument disappears at the widths needed to approach CNN accuracy.

What it does not show:
- Whether dropout or other regularization could stabilize the mid-width accuracy dip.
- Whether extended training or learning-rate scheduling can close the remaining 0.025 gap.
- Wider SRWN (h256+) behavior, though h192 already exceeds CNN in MACs so this would abandon the efficiency motivation entirely.

## Experiment 6: Fashion-MNIST v4 wide push with dropout

Question:
- Does dropout regularization (p=0.1) improve SRWN generalization at medium and high widths?
- Can v4 close the v3 gap to CNN accuracy?

What was run:
- [fashion_mnist_v4.py](../fashion_mnist_v4.py), [fashion-mnist-v4.py](../fashion-mnist-v4.py)
- Core SRWN path uses optional dropout via [fashion_mnist_benchmark.py](../fashion_mnist_benchmark.py)
- Metrics/report: [outputs/fashion_v4_metrics.json](fashion_v4_metrics.json), [outputs/fashion_v4_verdict.json](fashion_v4_verdict.json), [outputs/fashion_v4_results.md](fashion_v4_results.md)
- Plots: [outputs/fashion_v4_accuracy_bar.png](fashion_v4_accuracy_bar.png), [outputs/fashion_v4_compute_bar.png](fashion_v4_compute_bar.png), [outputs/fashion_v4_acc_vs_macs.png](fashion_v4_acc_vs_macs.png)
- Training: 25 epochs, rows=3, cols=4, waves=3, dropout=0.1, hidden_dim sweep h64→h192

Key evidence:

| Variant | Hidden dim | Test acc | Adaptive MACs | Params | Beats CNN? |
|---|---:|---:|---:|---:|---|
| wide_d01_r3_h64_w3 | 64 | 0.8598 | 308563.56 | 201256 | ✗ no |
| wide_d01_r3_h96_w3 | 96 | 0.8608 | 610772.89 | 412456 | ✗ no |
| wide_d01_r3_h128_w3 | 128 | 0.8554 | 1178628.51 | 697384 | ✗ no |
| wide_d01_r3_h160_w3 | 160 | 0.8536 | 1525049.60 | 1056040 | ✗ no |
| wide_d01_r3_h192_w3 | 192 | 0.8582 | 2225316.25 | 1488424 | ✗ no |
| CNN baseline | — | 0.8906 | 1218048.00 | 206922 | reference |

Best SRWN accuracy: `wide_d01_r3_h96_w3` at 0.8608 — gap to CNN is 0.0298.

What this experiment shows:
- Dropout 0.1 did not close the CNN gap and did not produce any SRWN accuracy win.
- v4 best (0.8608) is slightly below v3 best (0.8624), so dropout did not raise the observed accuracy ceiling.
- Non-monotonic width behavior remains: accuracy peaks at h96 and declines at larger widths.
- At h128 and above, SRWN compute is comparable to or above CNN while still lower in accuracy.

What it does not show:
- Whether different dropout rates (0.05, 0.2) or scheduling would change the ceiling.
- Whether learning-rate scheduling with dropout can improve wide-model stability.

## Cross-experiment synthesis

### Supported conclusions

1. Mechanism claim is supported: SRWN recurrence contributes meaningful refinement on parity.
2. Efficiency claim is supported in-model: adaptive halting reduces SRWN compute.
3. System-level frontier claim is supported: SRWN occupies lower-compute, lower-accuracy space relative to tested CNN across all image experiments.
4. Scaling-direction claim is partially supported: width helps more than depth, but accuracy improvement from additional width is non-monotonic and levels off below CNN.
5. Accuracy ceiling claim confirmed: even at h192 (1.5M params, more compute than CNN), SRWN best accuracy is 0.8624 vs CNN 0.8872. Matching CNN accuracy on this architecture and training setup is not supported by evidence.

### Not yet supported

1. Dominance claim: SRWN has not beaten CNN on either accuracy or the accuracy+compute combination.
2. Convergence claim: fixed-point-like wave convergence remains unproven.
3. Runtime advantage: MAC counts are analytical and not a substitute for latency profiling.
4. Regularization benefit: dropout 0.1 was tested; it did not close the CNN gap and did not improve the best v3 accuracy.

## Plot-backed interpretation guide

1. [outputs/accuracy_vs_compute.png](accuracy_vs_compute.png): parity benchmark, adaptive halting benefit and MLP compute dominance.
2. [outputs/fashion_accuracy_vs_compute.png](fashion_accuracy_vs_compute.png): baseline image tradeoff, SRWN left/lower and CNN right/higher.
3. [outputs/fashion_v2_acc_vs_macs.png](fashion_v2_acc_vs_macs.png): v2 frontier, no SRWN point beats CNN on either axis.
4. [outputs/fashion_v2_accuracy_bar.png](fashion_v2_accuracy_bar.png): width helps more than depth in v2 range.
5. [outputs/fashion_v3_accuracy_bar.png](fashion_v3_accuracy_bar.png): v3 width push — accuracy plateau, non-monotonic curve, CNN remains out of reach.
6. [outputs/fashion_v3_acc_vs_macs.png](fashion_v3_acc_vs_macs.png): at h128+ SRWN MACs exceed CNN MACs, yet accuracy stays lower. The efficiency argument disappears before accuracy matches.
7. [outputs/fashion_v4_accuracy_bar.png](fashion_v4_accuracy_bar.png): dropout run still peaks below CNN and below v3 best.
8. [outputs/fashion_v4_acc_vs_macs.png](fashion_v4_acc_vs_macs.png): dropout-adjusted frontier remains below CNN; high-width SRWN becomes compute-heavier than CNN.

## What work is still required

### Priority 1: improve optimization after dropout result

1. Keep dropout support, but sweep rates (0.0, 0.05, 0.1, 0.2) at h96 and h128 to locate the best regularization point.
2. Add learning-rate schedule (cosine or step decay) because fixed LR with 25 epochs may trap wide variants below their ceiling.
3. Constrain search to SRWN configs below CNN MAC budget while maximizing accuracy.

### Priority 2: close methodological gaps

1. Train SRWN and CNN to the same validation loss plateau, not a fixed epoch count.
2. Profile actual latency and memory, not only MACs.
3. Add LR scheduling (step decay or cosine annealing) to all long training runs.

### Priority 3: strengthen comparisons and architecture

1. Add a stronger CNN baseline (e.g., deeper CNN or lightweight conv model at same parameter budget as best SRWN).
2. Test whether adding a spatial-aware input encoding (e.g., 2D positional embedding) to SRWN chunks improves image accuracy.
3. Test halting-by-difficulty: does early halting correlate with genuinely easy samples, or is it arbitrary?

## Bottom line

Across six experiments, SRWN still sits below CNN on Fashion-MNIST accuracy. The v3 extreme-width run reached 0.8624, and v4 with dropout 0.1 reached 0.8608, while CNN was 0.8906 in the v4 run. Dropout made training more regularized but did not deliver a breakthrough in accuracy or restore a compute-quality advantage at high widths. The next step is targeted optimization (dropout-rate and LR-schedule sweeps) rather than further naive width scaling.