import argparse
import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


N_CLASSES = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class GridCell(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.top = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.left = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.right = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, h_top, h_left, h_right):
        out = torch.tanh(self.top(h_top) + self.left(h_left) + self.right(h_right) + self.bias)
        return self.dropout(out)


class SRWNFashion(nn.Module):
    def __init__(self, rows: int, cols: int, hidden_dim: int, input_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        if input_dim % cols != 0:
            raise ValueError("input_dim must be divisible by cols")

        self.rows = rows
        self.cols = cols
        self.hidden_dim = hidden_dim
        self.chunk_dim = input_dim // cols
        self.dropout_rate = dropout_rate

        self.input_layers = nn.ModuleList([nn.Linear(self.chunk_dim, hidden_dim) for _ in range(cols)])
        self.cells = nn.ModuleList([GridCell(hidden_dim, dropout_rate=dropout_rate) for _ in range(rows * cols)])
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, N_CLASSES) for _ in range(cols)])

    def cell(self, row: int, col: int) -> GridCell:
        return self.cells[row * self.cols + col]

    def forward(self, flat_images, waves: int):
        batch_size = flat_images.shape[0]
        device = flat_images.device
        zero = torch.zeros(batch_size, self.hidden_dim, device=device)

        chunks = flat_images.view(batch_size, self.cols, self.chunk_dim)
        projected = torch.stack(
            [self.input_layers[col](chunks[:, col, :]) for col in range(self.cols)],
            dim=1,
        )

        state = torch.zeros(batch_size, self.rows, self.cols, self.hidden_dim, device=device)
        all_logits = []
        all_conf = []

        for _ in range(waves):
            prev_state = state
            next_rows = []

            for row in range(self.rows):
                next_cols = []
                for col in range(self.cols):
                    h_top = projected[:, col, :] if row == 0 else next_rows[row - 1][:, col, :]
                    h_left = zero if col == 0 else next_cols[col - 1]
                    h_right = zero if col == self.cols - 1 else prev_state[:, row, col + 1, :]
                    next_cols.append(self.cell(row, col)(h_top, h_left, h_right))
                next_rows.append(torch.stack(next_cols, dim=1))

            state = torch.stack(next_rows, dim=1)
            bottom = state[:, self.rows - 1, :, :]
            logits = torch.stack([self.heads[col](bottom[:, col, :]) for col in range(self.cols)], dim=1)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
            conf = 1.0 - entropy / torch.log(torch.tensor(float(N_CLASSES), device=device))

            all_logits.append(logits)
            all_conf.append(conf)

        return torch.stack(all_logits, dim=1), torch.stack(all_conf, dim=1)


class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, N_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def count_params(model):
    return sum(param.numel() for param in model.parameters())


def make_loaders(data_dir, batch_size, train_size, val_size, test_size, seed):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_full = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_full = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    rng = random.Random(seed)
    train_indices = list(range(len(train_full)))
    rng.shuffle(train_indices)
    train_indices = train_indices[: train_size + val_size]
    train_subset = Subset(train_full, train_indices[:train_size])
    val_subset = Subset(train_full, train_indices[train_size:train_size + val_size])

    test_indices = list(range(len(test_full)))
    rng.shuffle(test_indices)
    test_subset = Subset(test_full, test_indices[:test_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def eval_srwn(model, loader, eval_waves, conf_threshold, device):
    model.eval()
    total_right = 0
    total_halted = 0
    total = 0
    sum_halt_wave = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            flat = images.view(images.shape[0], -1)

            logits, confs = model(flat, eval_waves)
            rightmost_logits = logits[:, -1, -1, :]
            right_preds = rightmost_logits.argmax(dim=-1)
            total_right += (right_preds == labels).sum().item()

            best_conf = confs.max(dim=2).values
            best_col = confs.argmax(dim=2)
            reached = best_conf >= conf_threshold
            first_wave = reached.float().argmax(dim=1) + 1
            first_wave = torch.where(reached.any(dim=1), first_wave, torch.full_like(first_wave, eval_waves))

            idx = torch.arange(labels.shape[0], device=device)
            wave_idx = first_wave - 1
            cols = best_col[idx, wave_idx]
            halted_logits = logits[idx, wave_idx, cols, :]
            halted_preds = halted_logits.argmax(dim=-1)

            total_halted += (halted_preds == labels).sum().item()
            sum_halt_wave += first_wave.float().sum().item()
            total += labels.shape[0]

    return {
        "rightmost_test_accuracy": round(total_right / total, 4),
        "halted_test_accuracy": round(total_halted / total, 4),
        "mean_halt_wave": round(sum_halt_wave / total, 4),
    }


def eval_cnn(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.shape[0]
    return {"test_accuracy": round(correct / total, 4)}


def train_srwn(config, train_loader, val_loader, device):
    model = SRWNFashion(
        config["rows"],
        config["cols"],
        config["hidden_dim"],
        input_dim=28 * 28,
        dropout_rate=config.get("dropout_rate", 0.0),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    wave_weights = torch.linspace(0.5, 1.0, config["train_waves"], device=device)
    col_weights = torch.linspace(0.5, 1.0, config["cols"], device=device)
    total_weight = float((wave_weights[:, None] * col_weights[None, :]).sum().item())

    history = {"train_loss": [], "val_acc": []}
    best_val = -1.0
    best_state = None

    for _ in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            flat = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            logits, _confs = model(flat, config["train_waves"])
            loss = 0.0
            for wave in range(config["train_waves"]):
                for col in range(config["cols"]):
                    weight = wave_weights[wave] * col_weights[col]
                    loss = loss + weight * F.cross_entropy(logits[:, wave, col, :], labels)
            norm_loss = loss / total_weight
            norm_loss.backward()
            optimizer.step()

            running_loss += norm_loss.item()
            n_batches += 1

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                flat = images.view(images.shape[0], -1)
                val_logits, _confs = model(flat, config["eval_waves"])
                preds = val_logits[:, -1, -1, :].argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.shape[0]

        history["train_loss"].append(running_loss / max(1, n_batches))
        val_acc = val_correct / val_total
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history


def train_cnn(config, train_loader, val_loader, device):
    model = CNNBaseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    history = {"train_loss": [], "val_acc": []}
    best_val = -1.0
    best_state = None

    for _ in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.shape[0]

        history["train_loss"].append(running_loss / max(1, n_batches))
        val_acc = val_correct / val_total
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history


def srwn_macs(config, mean_halt_wave):
    input_dim = 28 * 28
    chunk_dim = input_dim // config["cols"]
    input_projection = config["cols"] * chunk_dim * config["hidden_dim"]
    node_per_wave = config["rows"] * config["cols"] * (3 * config["hidden_dim"] * config["hidden_dim"])
    head_per_wave = config["cols"] * config["hidden_dim"] * N_CLASSES
    per_wave = node_per_wave + head_per_wave
    fixed = input_projection + config["eval_waves"] * per_wave
    adaptive = input_projection + mean_halt_wave * per_wave
    return {
        "fixed_eval_macs": round(float(fixed), 2),
        "adaptive_estimated_macs": round(float(adaptive), 2),
        "per_wave_macs": round(float(per_wave), 2),
    }


def cnn_macs():
    conv1 = 28 * 28 * 16 * (1 * 3 * 3)
    conv2 = 14 * 14 * 32 * (16 * 3 * 3)
    fc1 = (32 * 7 * 7) * 128
    fc2 = 128 * N_CLASSES
    total = conv1 + conv2 + fc1 + fc2
    return {"fixed_eval_macs": float(total)}


def plot_curves(results, out_dir):
    epochs = list(range(1, len(results["srwn"]["history"]["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, results["srwn"]["history"]["train_loss"], label="SRWN")
    plt.plot(epochs, results["cnn"]["history"]["train_loss"], label="CNN baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Fashion-MNIST train loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fashion_train_loss.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, results["srwn"]["history"]["val_acc"], label="SRWN")
    plt.plot(epochs, results["cnn"]["history"]["val_acc"], label="CNN baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Fashion-MNIST validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fashion_val_accuracy.png"), dpi=140)
    plt.close()


def plot_accuracy_vs_compute(results, out_dir):
    labels = ["SRWN fixed", "SRWN adaptive", "CNN baseline"]
    xs = [
        results["srwn"]["compute"]["fixed_eval_macs"],
        results["srwn"]["compute"]["adaptive_estimated_macs"],
        results["cnn"]["compute"]["fixed_eval_macs"],
    ]
    ys = [
        results["srwn"]["test"]["rightmost_test_accuracy"],
        results["srwn"]["test"]["halted_test_accuracy"],
        results["cnn"]["test"]["test_accuracy"],
    ]

    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, s=90)
    for i, label in enumerate(labels):
        plt.annotate(label, (xs[i], ys[i]), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Estimated MACs per sample")
    plt.ylabel("Test accuracy")
    plt.title("Fashion-MNIST accuracy vs compute")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fashion_accuracy_vs_compute.png"), dpi=140)
    plt.close()


def render_markdown(results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Fashion-MNIST SRWN vs CNN Results",
        "",
        f"Generated: {now}",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]

    for key, value in results["config"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## Final Comparison",
        "",
        "| Model | Test accuracy | Params | Estimated MACs/sample |",
        "|---|---:|---:|---:|",
        f"| SRWN fixed | {results['srwn']['test']['rightmost_test_accuracy']:.4f} | {results['srwn']['params']} | {results['srwn']['compute']['fixed_eval_macs']:.2f} |",
        f"| SRWN adaptive | {results['srwn']['test']['halted_test_accuracy']:.4f} | {results['srwn']['params']} | {results['srwn']['compute']['adaptive_estimated_macs']:.2f} |",
        f"| CNN baseline | {results['cnn']['test']['test_accuracy']:.4f} | {results['cnn']['params']} | {results['cnn']['compute']['fixed_eval_macs']:.2f} |",
        "",
        "## Plot files",
        "",
        "- outputs/fashion_train_loss.png",
        "- outputs/fashion_val_accuracy.png",
        "- outputs/fashion_accuracy_vs_compute.png",
        "",
    ])

    return "\n".join(lines)


def run(config, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = make_loaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        train_size=config["train_size"],
        val_size=config["val_size"],
        test_size=config["test_size"],
        seed=config["seed"],
    )

    srwn_model, srwn_history = train_srwn(config, train_loader, val_loader, device)
    cnn_model, cnn_history = train_cnn(config, train_loader, val_loader, device)

    srwn_test = eval_srwn(srwn_model, test_loader, config["eval_waves"], config["conf_threshold"], device)
    cnn_test = eval_cnn(cnn_model, test_loader, device)

    results = {
        "config": {**config, "device": device},
        "srwn": {
            "history": srwn_history,
            "test": srwn_test,
            "params": count_params(srwn_model),
            "compute": srwn_macs(config, srwn_test["mean_halt_wave"]),
        },
        "cnn": {
            "history": cnn_history,
            "test": cnn_test,
            "params": count_params(cnn_model),
            "compute": cnn_macs(),
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "fashion_metrics.json")
    md_path = os.path.join(out_dir, "fashion_results.md")

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(results))

    plot_curves(results, out_dir)
    plot_accuracy_vs_compute(results, out_dir)

    return {
        "metrics": metrics_path,
        "markdown": md_path,
        "plots": {
            "train_loss": os.path.join(out_dir, "fashion_train_loss.png"),
            "val_accuracy": os.path.join(out_dir, "fashion_val_accuracy.png"),
            "accuracy_vs_compute": os.path.join(out_dir, "fashion_accuracy_vs_compute.png"),
        },
        "summary": {
            "srwn": results["srwn"]["test"],
            "cnn": results["cnn"]["test"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST SRWN vs CNN benchmark")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--data-dir", type=str, default="outputs/fashion_data")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--train-waves", type=int, default=3)
    parser.add_argument("--eval-waves", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--conf-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    config = {
        "seed": args.seed,
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "rows": args.rows,
        "cols": args.cols,
        "hidden_dim": args.hidden_dim,
        "train_waves": args.train_waves,
        "eval_waves": args.eval_waves,
        "learning_rate": args.learning_rate,
        "conf_threshold": args.conf_threshold,
    }

    outputs = run(config, args.out_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
