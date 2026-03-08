"""
Vocal Isolation from Stereo Music
=================================
Deep learning project using U-Net CNN to separate vocals
from instrumental accompaniment in stereo music recordings.

Usage:
    python main.py train                          # Train the model
    python main.py train --musdb_root ./data/musdb18  # Custom dataset path
    python main.py evaluate                       # Evaluate on test set
    python main.py separate song.wav              # Isolate vocals from a file
    python main.py separate song.wav -o ./output  # Save to custom folder
"""

import argparse
import json
import sys
from pathlib import Path

from vocal_isolation.config import Config
from vocal_isolation.model import build_model
from vocal_isolation.dataset import create_dataloaders
from vocal_isolation.train import Trainer
from vocal_isolation.evaluate import evaluate_dataset
from vocal_isolation.inference import run_inference
from vocal_isolation.visualize import plot_training_history


def cmd_train(args):
    config = Config()
    config.data.musdb_root = args.musdb_root
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size

    print("=" * 60)
    print("  Vocal Isolation - Training")
    print("=" * 60)
    print(f"  Dataset:    {config.data.musdb_root}")
    print(f"  Device:     {config.train.resolve_device()}")
    print(f"  Epochs:     {config.train.epochs}")
    print(f"  Batch size: {config.train.batch_size}")
    print(f"  LR:         {config.train.lr}")
    print("=" * 60)

    model = build_model(config.model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,} (trainable: {trainable:,})")

    loaders = create_dataloaders(config)
    print(f"  Train segments: {len(loaders['train'].dataset)}")
    print(f"  Valid segments: {len(loaders['valid'].dataset)}")
    print("=" * 60)

    trainer = Trainer(model, config)

    if args.resume:
        epoch = trainer.load_checkpoint(args.resume)
        print(f"  Resumed from epoch {epoch}")

    history = trainer.fit(loaders["train"], loaders["valid"])

    plot_training_history(
        history,
        save_path=str(Path(config.train.checkpoint_dir) / "training_history.png"),
    )

    history_path = Path(config.train.checkpoint_dir) / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoint saved to: {config.train.checkpoint_dir}")


def cmd_evaluate(args):
    config = Config()
    config.data.musdb_root = args.musdb_root

    print("=" * 60)
    print("  Vocal Isolation - Evaluation")
    print("=" * 60)

    device = config.train.resolve_device()
    model = build_model(config.model)

    ckpt_path = args.checkpoint or str(Path(config.train.checkpoint_dir) / "best.pt")
    ckpt = __import__("torch").load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"  Loaded checkpoint: {ckpt_path}")

    loaders = create_dataloaders(config)
    results = evaluate_dataset(model, loaders["test"], config, device)

    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    for metric, stats in results.items():
        print(f"  {metric}: mean={stats['mean']:.2f} dB, "
              f"median={stats['median']:.2f} dB, std={stats['std']:.2f}")

    results_path = Path(config.train.checkpoint_dir) / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


def cmd_separate(args):
    config = Config()
    ckpt_path = args.checkpoint or str(Path(config.train.checkpoint_dir) / "best.pt")
    output_dir = args.output or "./vocal_isolation/outputs"

    print("=" * 60)
    print("  Vocal Isolation - Separation")
    print("=" * 60)
    print(f"  Input:      {args.audio_file}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output:     {output_dir}")
    print("=" * 60)

    run_inference(ckpt_path, args.audio_file, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Vocal Isolation from Stereo Music using U-Net CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train
    train_parser = subparsers.add_parser("train", help="Train the U-Net model")
    train_parser.add_argument("--musdb_root", default="./data/musdb18",
                              help="Path to MUSDB18 dataset")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch_size", type=int, default=8)
    train_parser.add_argument("--resume", type=str, default=None,
                              help="Path to checkpoint to resume from")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on test set")
    eval_parser.add_argument("--musdb_root", default="./data/musdb18")
    eval_parser.add_argument("--checkpoint", type=str, default=None)

    # Separate
    sep_parser = subparsers.add_parser("separate", help="Isolate vocals from audio")
    sep_parser.add_argument("audio_file", help="Path to input audio file")
    sep_parser.add_argument("-o", "--output", default=None, help="Output directory")
    sep_parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {"train": cmd_train, "evaluate": cmd_evaluate, "separate": cmd_separate}
    commands[args.command](args)


if __name__ == "__main__":
    main()
