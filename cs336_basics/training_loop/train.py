import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.training_loop.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.training_loop.data_loading import DataLoader
from cs336_basics.transformer_lm_architecture.transformer_lm import TransformerLM
from cs336_basics.transformer_lm_training.adamw import AdamW
from cs336_basics.transformer_lm_training.learning_rate_schedule import CosineSchedule
from cs336_basics.transformer_lm_training.cross_entropy import CrossEntropyLoss, Perplexity
from cs336_basics.transformer_lm_training.gradient_clipping import GradientClipper

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer LM on tokenized data.")

    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--data-format", choices=("npy", "memmap"), default="npy")
    parser.add_argument("--token-dtype", default="int32")

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--total-iters", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)

    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)

    return parser.parse_args()


def load_dataset(path: Path, data_format: str, token_dtype: str) -> np.ndarray:
    if data_format == "npy":
        return np.load(path, mmap_mode="r")

    dtype = np.dtype(token_dtype)
    return np.memmap(path, dtype=dtype, mode="r")


def language_model_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: CrossEntropyLoss,
) -> torch.Tensor:
    logits = model(inputs)
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    return criterion(logits, targets)


def language_model_perplexity(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    metric: Perplexity,
) -> torch.Tensor:
    logits = model(inputs)
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    return metric(logits, targets)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    num_batches: int,
    criterion: CrossEntropyLoss,
    perplexity_metric: Perplexity,
) -> tuple[float, float]:
    model.eval()
    losses = []
    perplexities = []

    for _ in range(num_batches):
        inputs, targets = data_loader.get_batch()
        loss = language_model_loss(model, inputs, targets, criterion)
        perplexity = language_model_perplexity(model, inputs, targets, perplexity_metric)
        losses.append(loss.item())
        perplexities.append(perplexity.item())

    mean_loss = float(sum(losses) / len(losses))
    perplexity = float(sum(perplexities) / len(perplexities))
    return mean_loss, perplexity


def maybe_init_wandb(args: argparse.Namespace) -> Any:
    if args.wandb_project is None:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases logging requested, but `wandb` is not installed."
        ) from exc

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    return wandb


def train(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = load_dataset(args.train_data, args.data_format, args.token_dtype)
    val_data = load_dataset(args.val_data, args.data_format, args.token_dtype)

    if len(train_data) <= args.context_length or len(val_data) <= args.context_length:
        raise ValueError("Both train and val datasets must be longer than context_length.")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineSchedule(
        optimizer=optimizer,
        lr_min=args.min_learning_rate,
        lr_max=args.learning_rate,
        warmup_steps=args.warmup_iters,
        total_steps=args.total_iters,
    )
    criterion = CrossEntropyLoss()
    perplexity_metric = Perplexity()
    clipper = GradientClipper(args.grad_clip)
    train_loader = DataLoader(
        x=train_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
    )
    val_loader = DataLoader(
        x=val_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
    )
    wandb_run = maybe_init_wandb(args)

    start_iteration = 0
    if args.resume_from is not None:
        start_iteration = load_checkpoint(args.resume_from, model, optimizer)
        scheduler.t = start_iteration

    for iteration in range(start_iteration, args.total_iters):
        model.train()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        inputs, targets = train_loader.get_batch()

        optimizer.zero_grad(set_to_none=True)
        loss = language_model_loss(model, inputs, targets, criterion)
        loss.backward()
        clipper(model.parameters())
        optimizer.step()

        step = iteration + 1

        if step % args.log_interval == 0 or step == 1:
            train_loss = float(loss.item())
            train_perplexity = float(
                language_model_perplexity(model, inputs, targets, perplexity_metric).item()
            )
            print(
                f"iter={step} lr={lr:.6g} train_loss={train_loss:.4f} "
                f"train_ppl={train_perplexity:.4f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "iteration": step,
                        "lr": lr,
                        "train/loss": train_loss,
                        "train/perplexity": train_perplexity,
                    },
                    step=step,
                )

        if step % args.eval_interval == 0 or step == args.total_iters:
            val_loss, val_perplexity = evaluate(
                model=model,
                data_loader=val_loader,
                num_batches=args.eval_batches,
                criterion=criterion,
                perplexity_metric=perplexity_metric,
            )
            print(
                f"iter={step} val_loss={val_loss:.4f} val_ppl={val_perplexity:.4f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "iteration": step,
                        "val/loss": val_loss,
                        "val/perplexity": val_perplexity,
                    },
                    step=step,
                )

        if args.checkpoint_path is not None and (
            step % args.checkpoint_interval == 0 or step == args.total_iters
        ):
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, step, args.checkpoint_path)

    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
