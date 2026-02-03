# train.py
import argparse
import os
import random
import yaml
import numpy as np
import torch

from src.trainers.stage2_trainer import Stage2Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # seed
    set_seed(int(cfg.get("seed", 0)))

    # device
    if cfg.get("device", "cuda") == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"

    trainer = Stage2Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
