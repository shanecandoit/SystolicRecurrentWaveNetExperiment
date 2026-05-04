# Fashion-MNIST v3 → v4 Experiment Tracker

---

## Experiment 5 (v3): Wide SRWN Push — DONE

Goal: Push SRWN to max width (h64→h192) to find accuracy ceiling.

### Results (25 epochs, rows=3, cols=4, waves=3)

| Variant | Test acc | Adaptive MACs | Beats CNN? |
|---|---:|---:|---|
| wide_r3_h64_w3 | 0.8582 | 303433 | ✗ |
| wide_r3_h96_w3 | 0.8552 | 687696 | ✗ |
| wide_r3_h128_w3 | 0.8518 | 1053928 | ✗ |
| wide_r3_h160_w3 | 0.8540 | 1466214 | ✗ |
| wide_r3_h192_w3 | 0.8624 | 2186874 | ✗ |
| CNN baseline | 0.8872 | 1218048 | — |

Key finding: accuracy is non-monotonic (h64 > h96 > h128, then recovers). Gap to CNN = 0.0248. At h128+ SRWN MACs exceed CNN MACs with lower accuracy.

### Tasks
- [x] Create fashion_mnist_v3.py
- [x] Run experiment (25 epochs)
- [x] Update outputs/results.md with honest findings
- [x] Update README.md (experiment list, artifact refs, run commands)
- [x] Update cross-experiment synthesis and bottom line in results.md

---

## Experiment 6 (v4): Dropout Regularization — DONE

Goal: Determine whether dropout (p=0.1) in GridCell stabilizes training at medium widths and raises the accuracy ceiling.

### Results (25 epochs, rows=3, cols=4, waves=3, dropout=0.1)

| Variant | Test acc | Adaptive MACs | Beats CNN? |
|---|---:|---:|---|
| wide_d01_r3_h64_w3 | 0.8598 | 308564 | ✗ |
| wide_d01_r3_h96_w3 | 0.8608 | 610773 | ✗ |
| wide_d01_r3_h128_w3 | 0.8554 | 1178629 | ✗ |
| wide_d01_r3_h160_w3 | 0.8536 | 1525050 | ✗ |
| wide_d01_r3_h192_w3 | 0.8582 | 2225316 | ✗ |
| CNN baseline | 0.8906 | 1218048 | — |

Key findings:
- Dropout did not beat CNN on accuracy.
- Best SRWN was `wide_d01_r3_h96_w3` at 0.8608 (gap to CNN: 0.0298).
- Compared with v3 best (0.8624), v4 best is slightly lower.
- Non-monotonic width behavior persists (h96 best, then declines).

### Tasks
- [x] Add optional dropout support to SRWN in `fashion_mnist_benchmark.py`
- [x] Create `fashion_mnist_v4.py` and `fashion-mnist-v4.py`
- [x] Run experiment: `python fashion-mnist-v4.py --out-dir outputs --epochs 25 --dropout-rate 0.1`
- [x] Update `outputs/results.md` and `README.md` with honest v4 outcome

### Success criteria outcome
- [x] Honest report completed
- [x] Accuracy at h96 >= 0.8582 (true, but still below CNN)
- [ ] Best SRWN exceeds v3 best (0.8624)
- [ ] Any SRWN variant beats CNN accuracy (0.8906)

Conclusion: dropout 0.1 improved some medium-width stability relative to v3 dip behavior, but did not raise the SRWN accuracy ceiling or close the CNN gap.

---

## Next focus (Experiment 7 candidate)

1. Add learning-rate schedule (cosine or step decay) with dropout 0.1.
2. Sweep dropout rates (0.0, 0.05, 0.1, 0.2) for h96 and h128 only.
3. Keep compute in check by targeting models <= CNN MAC budget while maximizing accuracy.
