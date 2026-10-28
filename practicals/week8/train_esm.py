"""Train a small ESM-2 classifier for the Week 8 practical.

The task is multi-label protein subcellular-localization prediction.  The
script uses SwissProt partitions 0--3 for training, partition 4 for
validation, and the supplied HPA table as an external test set.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


TARGET_COLUMNS = [
    "Cytoplasm",
    "Nucleus",
    "Cell membrane",
    "Mitochondrion",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--hpa-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/esm35m_frozen"))
    parser.add_argument("--model-name", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument(
        "--mode",
        choices=("frozen", "partial", "full"),
        default=None,
        help="Convenience alias; use --num-unfrozen-layers for explicit control.",
    )
    parser.add_argument(
        "--num-unfrozen-layers",
        type=int,
        default=None,
        help="0=frozen backbone, N=unfreeze the last N blocks, -1=full fine-tuning.",
    )
    parser.add_argument("--max-length", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_num_unfrozen_layers(mode, num_unfrozen_layers):
    """Resolve the explicit layer count while preserving the mode shortcut."""
    mode_defaults = {"frozen": 0, "partial": 2, "full": -1}

    if num_unfrozen_layers is None:
        return mode_defaults[mode or "frozen"]

    if num_unfrozen_layers < -1:
        raise ValueError("--num-unfrozen-layers must be -1, 0, or a positive integer.")

    if mode is not None:
        expected = mode_defaults[mode]
        if mode == "partial" and num_unfrozen_layers > 0:
            return num_unfrozen_layers
        if num_unfrozen_layers != expected:
            raise ValueError(
                f"Conflicting options: --mode {mode!r} implies "
                f"{expected}, but --num-unfrozen-layers is {num_unfrozen_layers}."
            )

    return num_unfrozen_layers


def load_splits(data_path: Path, hpa_path: Path):
    swiss = pd.read_csv(data_path).rename(
        columns={"ACC": "protein_id", "Sequence": "sequence", "Partition": "partition"}
    )
    hpa = pd.read_csv(hpa_path).rename(columns={"sid": "protein_id", "fasta": "sequence"})

    train = swiss[swiss["partition"].isin([0, 1, 2, 3])].copy()
    val = swiss[swiss["partition"] == 4].copy()

    # Exclude rows whose only labels are categories absent from the HPA table.
    train = train[train[TARGET_COLUMNS].sum(axis=1) > 0].reset_index(drop=True)
    val = val[val[TARGET_COLUMNS].sum(axis=1) > 0].reset_index(drop=True)
    test = hpa.reset_index(drop=True)
    return train, val, test


class ProteinDataset(Dataset):
    def __init__(self, frame, tokenizer, label_columns, max_length):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_columns = label_columns
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            row["sequence"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(
                row[self.label_columns].to_numpy(dtype=np.float32), dtype=torch.float32
            ),
        }


def masked_mean_pool(hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class ProteinFoundationClassifier(nn.Module):
    def __init__(self, backbone, hidden_dim, n_labels):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_dim, n_labels)

    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = masked_mean_pool(output.last_hidden_state, attention_mask)
        return self.classifier(pooled)


def find_transformer_blocks(backbone):
    """Find the Transformer block list for common Hugging Face layouts."""
    candidates = [
        ("encoder.layer", getattr(getattr(backbone, "encoder", None), "layer", None)),
        (
            "esm.encoder.layer",
            getattr(getattr(getattr(backbone, "esm", None), "encoder", None), "layer", None),
        ),
        (
            "transformer.layer",
            getattr(getattr(backbone, "transformer", None), "layer", None),
        ),
    ]
    for path, blocks in candidates:
        if blocks is not None and len(blocks) > 0:
            return path, blocks
    raise RuntimeError(
        "Could not find Transformer blocks. Inspect the loaded backbone with print(backbone)."
    )


def configure_trainability(model, num_unfrozen_layers):
    """Configure 0=frozen, N=last N blocks, -1=full fine-tuning."""
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    if num_unfrozen_layers == -1:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
    elif num_unfrozen_layers > 0:
        _, layers = find_transformer_blocks(model.backbone)
        if num_unfrozen_layers > len(layers):
            raise ValueError(
                f"Requested {num_unfrozen_layers} blocks, but the model has {len(layers)}."
            )
        for layer in layers[-num_unfrozen_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


def collect_metrics(y_true, logits):
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, predictions, average="micro", zero_division=0),
        "exact_match": accuracy_score(y_true, predictions),
    }


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    if training and not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()

    total_loss = 0.0
    all_labels, all_logits = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if training:
                optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_labels.append(labels.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

    metrics = collect_metrics(np.vstack(all_labels), np.vstack(all_logits))
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    args = parse_args()
    args.num_unfrozen_layers = resolve_num_unfrozen_layers(
        args.mode, args.num_unfrozen_layers
    )
    args.mode = (
        "frozen"
        if args.num_unfrozen_layers == 0
        else "full"
        if args.num_unfrozen_layers == -1
        else "partial"
    )
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test = load_splits(args.data_path, args.hpa_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    backbone = AutoModel.from_pretrained(args.model_name)
    model = ProteinFoundationClassifier(
        backbone, backbone.config.hidden_size, len(TARGET_COLUMNS)
    ).to(device)
    configure_trainability(model, args.num_unfrozen_layers)

    train_loader = DataLoader(
        ProteinDataset(train, tokenizer, TARGET_COLUMNS, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ProteinDataset(val, tokenizer, TARGET_COLUMNS, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        ProteinDataset(test, tokenizer, TARGET_COLUMNS, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )

    positive = train[TARGET_COLUMNS].sum().to_numpy()
    pos_weight = (len(train) - positive) / (positive + 1e-6)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    learning_rate = args.learning_rate or (1e-3 if args.mode == "frozen" else 2e-5)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )

    best_val_f1 = -np.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False)
        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(row)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")

    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    (args.output_dir / "labels.json").write_text(json.dumps(TARGET_COLUMNS, indent=2))
    (args.output_dir / "config.json").write_text(
        json.dumps({**vars(args), "device": str(device), "train_size": len(train), "val_size": len(val), "test_size": len(test)}, default=str, indent=2)
    )

    model.load_state_dict(torch.load(args.output_dir / "best_model.pt", map_location=device))
    test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, False)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print({"external_hpa_test": test_metrics})


if __name__ == "__main__":
    main()
