#!/usr/bin/env python3
"""
Inject experiment configs into the AlphaAgent package directory.

This script copies the pre-built YAML/generate.py configs for a given instrument
into the AlphaAgent installed package paths, and sets the CHAT_SEED in .env.

Usage:
    python patch_config.py --instrument csi300 --seed 42
    python patch_config.py --instrument csi800 --seed 0
    python patch_config.py --instrument all    --seed 128
"""
import argparse
import os
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ALPHAAGENT_DIR = SCRIPT_DIR / "AlphaAgent"

# Target paths inside the AlphaAgent package (these are where the code reads from)
FACTOR_TEMPLATE_DIR = (
    ALPHAAGENT_DIR / "alphaagent" / "scenarios" / "qlib" / "experiment" / "factor_template"
)
FACTOR_DATA_TEMPLATE_DIR = (
    ALPHAAGENT_DIR / "alphaagent" / "scenarios" / "qlib" / "experiment" / "factor_data_template"
)
ENV_FILE = ALPHAAGENT_DIR / ".env"


def copy_configs(instrument: str):
    """Copy instrument-specific configs into the AlphaAgent package directory."""
    src_dir = SCRIPT_DIR / "configs" / instrument

    if not src_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {src_dir}")

    # conf.yaml -> factor_template/
    shutil.copy2(src_dir / "conf.yaml", FACTOR_TEMPLATE_DIR / "conf.yaml")
    print(f"  [OK] conf.yaml -> {FACTOR_TEMPLATE_DIR / 'conf.yaml'}")

    # conf_cn_combined_kdd_ver.yaml -> factor_template/
    shutil.copy2(
        src_dir / "conf_cn_combined_kdd_ver.yaml",
        FACTOR_TEMPLATE_DIR / "conf_cn_combined_kdd_ver.yaml",
    )
    print(f"  [OK] conf_cn_combined_kdd_ver.yaml -> {FACTOR_TEMPLATE_DIR}")

    # generate.py -> factor_data_template/
    shutil.copy2(src_dir / "generate.py", FACTOR_DATA_TEMPLATE_DIR / "generate.py")
    print(f"  [OK] generate.py -> {FACTOR_DATA_TEMPLATE_DIR / 'generate.py'}")


def update_env_seed(seed: int):
    """Update CHAT_SEED in the .env file."""
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    else:
        lines = []

    new_lines = []
    found = False
    for line in lines:
        if line.strip().startswith("CHAT_SEED="):
            new_lines.append(f"CHAT_SEED={seed}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"CHAT_SEED={seed}\n")

    ENV_FILE.write_text("".join(new_lines), encoding="utf-8")
    print(f"  [OK] CHAT_SEED={seed}")


def clean_caches(instrument: str, seed: int):
    """Remove old logs for this experiment only.

    Per-experiment log directory is named log_{instrument}_seed{seed} so
    different parallel runs don't touch each other.

    pickle_cache and git_ignore_folder are NOT cleared — they are shared
    across parallel runs. Different seeds produce different cache keys so
    there is no collision; and git_ignore_folder/RD-Agent_workspace/
    contains in-flight workspaces that other parallel runs may be using.
    """
    log_dir = ALPHAAGENT_DIR / f"log_{instrument}_seed{seed}"
    if log_dir.exists():
        shutil.rmtree(log_dir)

    print(f"  [OK] Old log cleaned: {log_dir.name}")


def main():
    parser = argparse.ArgumentParser(description="Patch AlphaAgent configs for experiment")
    parser.add_argument("--instrument", required=True, choices=["csi300", "csi800", "all"])
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    print(f"=== Patching config: instrument={args.instrument}, seed={args.seed} ===")

    copy_configs(args.instrument)
    update_env_seed(args.seed)
    clean_caches(args.instrument, args.seed)

    print(f"\n=== Done. Ready to run: ===")
    print(f"  cd {ALPHAAGENT_DIR}")
    print(f'  alphaagent mine --direction "your hypothesis direction"')


if __name__ == "__main__":
    main()