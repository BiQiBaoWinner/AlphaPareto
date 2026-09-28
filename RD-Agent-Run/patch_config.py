#!/usr/bin/env python3
"""
Patch RD-Agent .env and yaml configs for a specific experiment.

Usage:
    python patch_config.py --instrument csi300 --seed 42
"""
import argparse
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RDAGENT_DIR = SCRIPT_DIR / "RD-Agent"
ENV_FILE = RDAGENT_DIR / ".env"

MARKET_MAP = {
    "csi300": ("csi300", "SH000300"),
    "csi800": ("csi800", "SH000906"),
    "all":    ("all",    "SH000985"),
}

FACTOR_TEMPLATE_DIR = (
    RDAGENT_DIR / "rdagent" / "scenarios" / "qlib" / "experiment" / "factor_template"
)


def update_env(seed: int):
    """Update CHAT_SEED in .env."""
    if ENV_FILE.exists():
        text = ENV_FILE.read_text(encoding="utf-8")
    else:
        text = ""

    if re.search(r"^CHAT_SEED\s*=", text, re.MULTILINE):
        text = re.sub(r"^CHAT_SEED\s*=.*$", f"CHAT_SEED={seed}", text, flags=re.MULTILINE)
    else:
        text += f"\nCHAT_SEED={seed}\n"

    ENV_FILE.write_text(text, encoding="utf-8")
    print(f"  [OK] CHAT_SEED={seed}")


def update_yaml_market(instrument: str):
    """Patch market & benchmark in all factor_template yaml files."""
    market, benchmark = MARKET_MAP[instrument]

    yaml_files = list(FACTOR_TEMPLATE_DIR.glob("conf*.yaml"))
    for yf in yaml_files:
        text = yf.read_text(encoding="utf-8")
        text = re.sub(r"(market:\s*&market\s+)\S+", rf"\g<1>{market}", text)
        text = re.sub(r"(benchmark:\s*&benchmark\s+)\S+", rf"\g<1>{benchmark}", text)
        yf.write_text(text, encoding="utf-8")
        print(f"  [OK] {yf.name} → market={market}, benchmark={benchmark}")


def clean_caches(instrument: str, seed: int):
    """Remove old logs for this experiment. Pickle cache is NOT cleared
    to allow parallel runs with different seeds (different seeds produce
    different cache keys, so there is no collision)."""
    import shutil

    # Clean the per-experiment log directory
    log_dir = RDAGENT_DIR / f"log_{instrument}_seed{seed}"
    if log_dir.exists():
        shutil.rmtree(log_dir)

    print("  [OK] Caches and old logs cleaned")


def main():
    parser = argparse.ArgumentParser(description="Patch RD-Agent configs for experiment")
    parser.add_argument("--instrument", required=True, choices=["csi300", "csi800", "all"])
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    print(f"=== Patching config: instrument={args.instrument}, seed={args.seed} ===")
    update_env(args.seed)
    update_yaml_market(args.instrument)
    clean_caches(args.instrument, args.seed)
    print("=== Done ===")


if __name__ == "__main__":
    main()
