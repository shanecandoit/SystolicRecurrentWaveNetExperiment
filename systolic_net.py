"""
Systolic Grid Neural Network
- 2D grid of tanh nodes
- Each node reads from top, left, right-previous-wave
- Bottom row = output heads, read after each wave
- Halt: state delta (convergence) or confidence threshold
- Exports to standalone C file with baked-in weights
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
import os

# ── Config ─────────────────────────────────────────────────────────────────────
ROWS        = 4
COLS        = 4
D           = 16        # hidden dim per node
N_CLASSES   = 10
PCA_DIM     = 32        # input compressed to this before feeding columns
N_WAVES_TR  = 4         # unroll depth during training
LR          = 0.001
EPOCHS      = 10
BATCH       = 64
DELTA_THRESH = 0.001    # convergence: max state change between waves
CONF_THRESH  = 0.90     # confidence: normalized entropy drop

np.random.seed(42)

# ── Node weights ────────────────────────────────────────────────────────────────

def make_weights():
    """All weights as dicts for easy indexing and export."""
    scale = 0.05
    W = {}
    for r in range(ROWS):
        for c in range(COLS):
            key = (r, c)
            W[key] = {
                'top':   np.random.randn(D, D) * scale,
                'left':  np.random.randn(D, D) * scale,
                'right': np.random.randn(D, D) * scale,
                'b':     np.zeros(D),
                # gradient accumulators
                'dW_top':   np.zeros((D, D)),
                'dW_left':  np.zeros((D, D)),
                'dW_right': np.zeros((D, D)),
                'db':       np.zeros(D),
            }
    # Input projections: map PCA_DIM -> D for each column
    W['in'] = [np.random.randn(D, PCA_DIM) * scale for _ in range(COLS)]
    W['din'] = [np.zeros((D, PCA_DIM)) for _ in range(COLS)]
    # Output heads: D -> N_CLASSES for each column
    W['head'] = [np.random.randn(N_CLASSES, D) * scale for _ in range(COLS)]
    W['dhead'] = [np.zeros((N_CLASSES, D)) for _ in range(COLS)]
    return W


def zero_grads(W):
    for r in range(ROWS):
        for c in range(COLS):
            k = (r, c)
            W[k]['dW_top'][:]   = 0
            W[k]['dW_left'][:]  = 0
            W[k]['dW_right'][:] = 0
            W[k]['db'][:]       = 0
    for c in range(COLS):
        W['din'][c][:] = 0
        W['dhead'][c][:] = 0


# ── Forward ─────────────────────────────────────────────────────────────────────

def node_forward(W, r, c, h_top, h_left, h_right):
    """Returns (h_out, z) where z is pre-activation (needed for backward)."""
    z = (W[(r,c)]['top']   @ h_top  +
         W[(r,c)]['left']  @ h_left +
         W[(r,c)]['right'] @ h_right +
         W[(r,c)]['b'])
    return np.tanh(z), z


def forward_wave(W, x_proj, prev_state):
    """
    x_proj: (COLS, D)  — projected input, one vector per column
    prev_state: (ROWS, COLS, D)  — state from previous wave (right-reads)
    Returns: new_state (ROWS, COLS, D), activations dict
    """
    state = np.zeros((ROWS, COLS, D))
    acts  = {}   # (r,c) -> (h, z, h_top, h_left, h_right)

    zero_vec = np.zeros(D)

    for r in range(ROWS):
        for c in range(COLS):
            h_top   = x_proj[c]     if r == 0          else state[r-1, c]
            h_left  = zero_vec      if c == 0           else state[r, c-1]
            h_right = zero_vec      if c == COLS-1      else prev_state[r, c+1]

            h, z = node_forward(W, r, c, h_top, h_left, h_right)
            state[r, c] = h
            acts[(r,c)] = (h, z, h_top, h_left, h_right)

    return state, acts


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def read_outputs(W, state):
    """Returns list of (probs, conf) for each bottom column."""
    results = []
    for c in range(COLS):
        h     = state[ROWS-1, c]
        logit = W['head'][c] @ h
        probs = softmax(logit)
        # Normalized entropy: 0=uniform, 1=certain
        H     = -np.sum(probs * np.log(probs + 1e-9))
        conf  = 1.0 - H / np.log(N_CLASSES)
        results.append((probs, conf))
    return results


# ── Backward ────────────────────────────────────────────────────────────────────

def backward_wave(W, acts, d_state_in, d_prev_state_in):
    """
    d_state_in:      (ROWS, COLS, D) — upstream gradient into this wave's state
    d_prev_state_in: (ROWS, COLS, D) — upstream gradient into prev_wave state
                     (from right-reads of the next wave's backward pass)
    Returns: d_prev_state (ROWS, COLS, D) — gradient to push to prev wave
    """
    d_prev_state = d_prev_state_in.copy()
    d_input_proj = np.zeros((COLS, D))

    # Reverse sweep (mirrors forward order)
    for r in reversed(range(ROWS)):
        for c in reversed(range(COLS)):
            h, z, h_top, h_left, h_right = acts[(r,c)]

            dh = d_state_in[r, c]
            dz = dh * (1.0 - h**2)   # tanh jacobian

            # Weight gradients
            W[(r,c)]['dW_top']   += np.outer(dz, h_top)
            W[(r,c)]['dW_left']  += np.outer(dz, h_left)
            W[(r,c)]['dW_right'] += np.outer(dz, h_right)
            W[(r,c)]['db']       += dz

            # Propagate to sources
            if r == 0:
                d_input_proj[c] += W[(r,c)]['top'].T @ dz
            else:
                d_state_in[r-1, c] += W[(r,c)]['top'].T @ dz

            if c > 0:
                d_state_in[r, c-1] += W[(r,c)]['left'].T @ dz

            # Right-read gradient goes back to prev_state of right neighbor
            if c < COLS - 1:
                d_prev_state[r, c+1] += W[(r,c)]['right'].T @ dz

    return d_prev_state, d_input_proj


# ── Training ─────────────────────────────────────────────────────────────────────

def cross_entropy_grad(probs, y_onehot):
    """Returns (loss, dlogits)."""
    loss     = -np.sum(y_onehot * np.log(probs + 1e-9))
    dlogits  = probs - y_onehot           # CE + softmax combined gradient
    return loss, dlogits


def train(X_train, y_train, X_val, y_val):
    W  = make_weights()
    lb = LabelBinarizer()
    lb.fit(y_train)
    Y  = lb.transform(y_train)

    N  = X_train.shape[0]
    # Weight waves: later waves count more
    wave_weights = np.linspace(0.5, 1.0, N_WAVES_TR)
    # Weight columns: rightmost counts more (has seen more context)
    col_weights  = np.linspace(0.5, 1.0, COLS)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        idx = np.random.permutation(N)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, N - BATCH, BATCH):
            batch_idx = idx[start:start+BATCH]
            Xb = X_train[batch_idx]     # (B, PCA_DIM)
            Yb = Y[batch_idx]           # (B, N_CLASSES)

            zero_grads(W)
            batch_loss = 0.0

            for i in range(BATCH):
                x = Xb[i]
                y = Yb[i]

                # Project input to column inputs
                x_proj = np.array([W['in'][c] @ x for c in range(COLS)])  # (COLS, D)

                # ── Forward: unroll N_WAVES_TR waves ──────────────────
                all_states = []   # list of (ROWS, COLS, D)
                all_acts   = []
                state      = np.zeros((ROWS, COLS, D))

                for w in range(N_WAVES_TR):
                    state, acts = forward_wave(W, x_proj, state)
                    all_states.append(state.copy())
                    all_acts.append(acts)

                # ── Loss at each wave's bottom row ────────────────────
                # d_state at each wave from head loss
                head_d_states = [np.zeros((ROWS, COLS, D)) for _ in range(N_WAVES_TR)]

                for w in range(N_WAVES_TR):
                    wgt_w = wave_weights[w]
                    results = read_outputs(W, all_states[w])
                    for c in range(COLS):
                        probs, _ = results[c]
                        loss, dlogit = cross_entropy_grad(probs, y)
                        wgt = wgt_w * col_weights[c]
                        batch_loss += wgt * loss
                        # Head gradient
                        dh = W['head'][c].T @ (wgt * dlogit)
                        head_d_states[w][ROWS-1, c] += dh
                        W['dhead'][c] += wgt * np.outer(dlogit, all_states[w][ROWS-1, c])

                # ── Backward: BPTT through waves (reverse) ────────────
                d_prev = np.zeros((ROWS, COLS, D))
                for w in reversed(range(N_WAVES_TR)):
                    d_state = head_d_states[w] + d_prev
                    d_prev, d_in = backward_wave(W, all_acts[w], d_state, np.zeros((ROWS, COLS, D)))
                    for c in range(COLS):
                        W['din'][c] += np.outer(d_in[c], x)

            # ── SGD update ────────────────────────────────────────────
            for r in range(ROWS):
                for c in range(COLS):
                    k = (r, c)
                    W[k]['top']   -= LR * W[k]['dW_top']   / BATCH
                    W[k]['left']  -= LR * W[k]['dW_left']  / BATCH
                    W[k]['right'] -= LR * W[k]['dW_right'] / BATCH
                    W[k]['b']     -= LR * W[k]['db']        / BATCH
            for c in range(COLS):
                W['in'][c]   -= LR * W['din'][c]   / BATCH
                W['head'][c] -= LR * W['dhead'][c] / BATCH

            total_loss += batch_loss / BATCH
            n_batches  += 1

        # Validation accuracy (use max waves, best-conf column)
        val_correct = 0
        for i in range(len(X_val)):
            state = np.zeros((ROWS, COLS, D))
            x_proj = np.array([W['in'][c] @ X_val[i] for c in range(COLS)])
            for _ in range(N_WAVES_TR):
                state, _ = forward_wave(W, x_proj, state)
            results = read_outputs(W, state)
            # Pick best confidence column
            pred = max(results, key=lambda r: r[1])[0].argmax()
            val_correct += (pred == y_val[i])
        val_acc = val_correct / len(X_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_W = {k: {kk: vv.copy() if isinstance(vv, np.ndarray) else
                          [a.copy() for a in vv] if isinstance(vv, list) else vv
                          for kk, vv in v.items()}
                      if isinstance(v, dict) else
                      ([a.copy() for a in v] if isinstance(v, list) else v)
                      for k, v in W.items()}

        print(f"Epoch {epoch+1:2d} | loss {total_loss/n_batches:.4f} | val_acc {val_acc:.4f}")

    return best_W


# ── C Export ─────────────────────────────────────────────────────────────────────

def arr_to_c(name, arr, indent=0):
    """Flatten a numpy array to a C float array initializer."""
    flat  = arr.flatten()
    lines = []
    pad   = " " * indent
    lines.append(f"{pad}static const float {name}[{flat.size}] = {{")
    row   = []
    for i, v in enumerate(flat):
        row.append(f"{v:.6f}f")
        if len(row) == 8 or i == len(flat)-1:
            lines.append(f"{pad}    {', '.join(row)},")
            row = []
    lines.append(f"{pad}}};")
    return "\n".join(lines)


def export_c(W, pca_components, pca_mean, out_path="systolic_net.c"):
    lines = []
    lines.append("/* Auto-generated systolic grid network */")
    lines.append("#include <math.h>")
    lines.append("#include <string.h>")
    lines.append(f"#define ROWS        {ROWS}")
    lines.append(f"#define COLS        {COLS}")
    lines.append(f"#define D           {D}")
    lines.append(f"#define N_CLASSES   {N_CLASSES}")
    lines.append(f"#define PCA_DIM     {PCA_DIM}")
    lines.append(f"#define INPUT_DIM   {pca_mean.shape[0]}")
    lines.append(f"#define DELTA_THRESH {DELTA_THRESH}f")
    lines.append(f"#define CONF_THRESH  {CONF_THRESH}f")
    lines.append("")

    # PCA projection
    lines.append(arr_to_c("pca_mean",       pca_mean))
    lines.append(arr_to_c("pca_components", pca_components))  # (PCA_DIM, INPUT_DIM)
    lines.append("")

    # Input projections
    for c in range(COLS):
        lines.append(arr_to_c(f"W_in_{c}", W['in'][c]))
    lines.append("")

    # Node weights
    for r in range(ROWS):
        for c in range(COLS):
            k = (r, c)
            lines.append(arr_to_c(f"W_top_{r}_{c}",   W[k]['top']))
            lines.append(arr_to_c(f"W_left_{r}_{c}",  W[k]['left']))
            lines.append(arr_to_c(f"W_right_{r}_{c}", W[k]['right']))
            lines.append(arr_to_c(f"b_{r}_{c}",       W[k]['b']))
    lines.append("")

    # Head weights
    for c in range(COLS):
        lines.append(arr_to_c(f"W_head_{c}", W['head'][c]))
    lines.append("")

    # ── Inference code ────────────────────────────────────────────────────────
    lines.append(r"""
static float state[ROWS][COLS][D];
static float prev_state[ROWS][COLS][D];

static float dot(const float* a, const float* b, int n) {
    float s = 0; for (int i=0; i<n; i++) s += a[i]*b[i]; return s;
}
static float tanhf_approx(float x) { return tanhf(x); }

/* Run one wave. W_top/left/right are DxD row-major. */
static void run_wave(const float x_proj[COLS][D]) {
    float zero[D]; memset(zero, 0, sizeof(zero));
    /* Pointer tables for weights — avoids giant switch */
    const float* W_tops[ROWS][COLS]   = {
""")
    # Generate pointer tables
    for r in range(ROWS):
        row_tops  = ", ".join(f"W_top_{r}_{c}"   for c in range(COLS))
        lines.append(f"        {{{row_tops}}},")
    lines.append("    };")
    lines.append("    const float* W_lefts[ROWS][COLS]  = {")
    for r in range(ROWS):
        row = ", ".join(f"W_left_{r}_{c}"  for c in range(COLS))
        lines.append(f"        {{{row}}},")
    lines.append("    };")
    lines.append("    const float* W_rights[ROWS][COLS] = {")
    for r in range(ROWS):
        row = ", ".join(f"W_right_{r}_{c}" for c in range(COLS))
        lines.append(f"        {{{row}}},")
    lines.append("    };")
    lines.append("    const float* biases[ROWS][COLS] = {")
    for r in range(ROWS):
        row = ", ".join(f"b_{r}_{c}" for c in range(COLS))
        lines.append(f"        {{{row}}},")
    lines.append("    };")

    lines.append(r"""
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            const float* h_top   = (r == 0)      ? x_proj[c]             : state[r-1][c];
            const float* h_left  = (c == 0)      ? zero                  : state[r][c-1];
            const float* h_right = (c == COLS-1) ? zero                  : prev_state[r][c+1];
            for (int d = 0; d < D; d++) {
                float z = biases[r][c][d];
                for (int k = 0; k < D; k++) {
                    z += W_tops[r][c][d*D+k]   * h_top[k];
                    z += W_lefts[r][c][d*D+k]  * h_left[k];
                    z += W_rights[r][c][d*D+k] * h_right[k];
                }
                state[r][c][d] = tanhf_approx(z);
            }
        }
    }
}

static void softmax_inplace(float* x, int n) {
    float mx = x[0];
    for (int i=1; i<n; i++) if (x[i]>mx) mx=x[i];
    float s = 0;
    for (int i=0; i<n; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i=0; i<n; i++) x[i] /= s;
}

/* Normalized entropy: 0=uniform, 1=certain */
static float confidence(const float* probs, int n) {
    float H = 0;
    for (int i=0; i<n; i++) if (probs[i]>0) H -= probs[i]*logf(probs[i]);
    return 1.0f - H / logf((float)n);
}

/* Max state delta between prev_state and state at bottom row */
static float bottom_delta(void) {
    float mx = 0;
    for (int c=0; c<COLS; c++)
        for (int d=0; d<D; d++) {
            float diff = fabsf(state[ROWS-1][c][d] - prev_state[ROWS-1][c][d]);
            if (diff > mx) mx = diff;
        }
    return mx;
}

/*
 * systolic_predict()
 *   input:       raw feature vector (INPUT_DIM floats)
 *   max_waves:   hard cap on wave iterations
 *   out_probs:   N_CLASSES float output (confidence-weighted avg across columns)
 *   out_waves:   how many waves were actually run
 *   returns:     predicted class index
 *
 * HALT LOGIC:
 *   After each wave, check two criteria:
 *   1. Bottom-row state delta < DELTA_THRESH  -> grid converged, more waves useless
 *   2. Best-column confidence > CONF_THRESH   -> answer is certain enough
 *   Whichever triggers first wins.
 *   This lets the CALLER also enforce a time budget by passing max_waves.
 */
int systolic_predict(const float* input, int max_waves,
                     float* out_probs, int* out_waves) {
    /* PCA project: subtract mean, then project */
    float centered[INPUT_DIM];
    for (int i=0; i<INPUT_DIM; i++) centered[i] = input[i] - pca_mean[i];

    float pca_out[PCA_DIM];
    for (int p=0; p<PCA_DIM; p++) {
        pca_out[p] = 0;
        for (int i=0; i<INPUT_DIM; i++)
            pca_out[p] += pca_components[p*INPUT_DIM+i] * centered[i];
    }

    /* Project PCA output into per-column D-vectors */
    float x_proj[COLS][D];
""")
    # Per-column input projection references
    lines.append("    const float* W_ins[COLS] = {" +
                 ", ".join(f"W_in_{c}" for c in range(COLS)) + "};")
    lines.append(r"""
    for (int c=0; c<COLS; c++)
        for (int d=0; d<D; d++) {
            x_proj[c][d] = 0;
            for (int p=0; p<PCA_DIM; p++)
                x_proj[c][d] += W_ins[c][d*PCA_DIM+p] * pca_out[p];
        }

    /* Reset state */
    memset(state,      0, sizeof(state));
    memset(prev_state, 0, sizeof(prev_state));
""")
    lines.append("    const float* W_heads[COLS] = {" +
                 ", ".join(f"W_head_{c}" for c in range(COLS)) + "};")
    lines.append(r"""
    float probs[COLS][N_CLASSES];
    float confs[COLS];
    int   done = 0;

    for (int w = 0; w < max_waves && !done; w++) {
        memcpy(prev_state, state, sizeof(state));
        run_wave((const float (*)[D])x_proj);

        float best_conf = 0;
        for (int c=0; c<COLS; c++) {
            for (int k=0; k<N_CLASSES; k++) {
                probs[c][k] = 0;
                for (int d=0; d<D; d++)
                    probs[c][k] += W_heads[c][k*D+d] * state[ROWS-1][c][d];
            }
            softmax_inplace(probs[c], N_CLASSES);
            confs[c] = confidence(probs[c], N_CLASSES);
            if (confs[c] > best_conf) best_conf = confs[c];
        }

        float delta = (w > 0) ? bottom_delta() : 1.0f;

        if (out_waves) *out_waves = w + 1;

        /* Halt check */
        if (delta < DELTA_THRESH || best_conf > CONF_THRESH) done = 1;
    }

    /* Aggregate: confidence-weighted average across columns */
    float total_conf = 0;
    for (int c=0; c<COLS; c++) total_conf += confs[c];
    for (int k=0; k<N_CLASSES; k++) out_probs[k] = 0;
    for (int c=0; c<COLS; c++)
        for (int k=0; k<N_CLASSES; k++)
            out_probs[k] += (confs[c] / total_conf) * probs[c][k];

    int best = 0;
    for (int k=1; k<N_CLASSES; k++)
        if (out_probs[k] > out_probs[best]) best = k;
    return best;
}
""")

    src = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(src)
    print(f"Exported {out_path} ({os.path.getsize(out_path)//1024} KB)")


# ── Main ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading MNIST...")
    mnist  = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y   = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(int)

    X_tr, X_val = X[:55000], X[55000:60000]
    y_tr, y_val = y[:55000], y[55000:60000]

    print(f"PCA {X_tr.shape[1]} -> {PCA_DIM}...")
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(X_tr)
    X_tr_pca  = pca.transform(X_tr)
    X_val_pca = pca.transform(X_val)

    print(f"Training {ROWS}x{COLS} grid, D={D}, {N_WAVES_TR} waves...")
    W = train(X_tr_pca, y_tr, X_val_pca, y_val)

    print("Exporting C...")
    export_c(W,
             pca_components=pca.components_,   # (PCA_DIM, INPUT_DIM)
             pca_mean=pca.mean_,                # (INPUT_DIM,)
             out_path="outputs/systolic_net.c")
    print("Done.")

