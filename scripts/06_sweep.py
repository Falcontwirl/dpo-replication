"""Run the whole experiment (configs/sweep.yaml) sequentially, skipping finished entries.

    python scripts/06_sweep.py                                    # full scale
    python scripts/06_sweep.py --config configs/base.yaml --config configs/smoke.yaml
    python scripts/06_sweep.py --only dpo                         # entries whose name contains "dpo"
    python scripts/06_sweep.py --dry-run

Special entries:
  {expand: ppo_lr_tune}   one PPO run per lr in ppo_tune.lrs (names tune_ppo_<reward>_lr<lr>), evaluated on
                          the ppo_tune.eval_split prompts; followed in sweep.yaml by scripts/05b_select_ppo_lr.py
  "@ppo_lr" in a --set    replaced by the lr chosen in <runs_dir>/ppo_lr_choice.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from dpo_rep.config import load_config

NO_NAME_SCRIPTS = ("02_gen_prefs", "05b_select_ppo_lr")  # scripts that do not create a run directory


def expand(entries: list[dict], cfg) -> list[dict]:
    out = []
    for e in entries:
        if e.get("expand") == "ppo_lr_tune":
            t = cfg.ppo_tune
            for lr in t.lrs:
                out.append({"name": f"tune_ppo_{t.reward}_lr{float(lr):g}", "script": "scripts/05_train_ppo.py",
                            "set": [f"ppo.reward={t.reward}", f"ppo.target_kl={t.target_kl}", f"ppo.lr={lr}",
                                    f"ppo.total_ppo_steps={t.total_ppo_steps}", f"eval.split={t.eval_split}"]})
        else:
            out.append(e)
    return out


def resolve(value: str, cfg) -> str:
    if "@ppo_lr" not in value:
        return value
    path = Path(cfg.paths.runs_dir) / "ppo_lr_choice.json"
    if not path.exists():
        raise SystemExit(f"{value!r} needs {path}; run the ppo_lr_tune entries and 05b_select_ppo_lr.py first")
    return value.replace("@ppo_lr", repr(json.loads(path.read_text())["lr"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--sweep", default="configs/sweep.yaml")
    ap.add_argument("--only", default=None, help="substring filter on entry names")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    configs = args.config or ["configs/base.yaml"]
    cfg = load_config(configs)
    entries = expand(yaml.safe_load(open(args.sweep))["runs"], cfg)

    for e in entries:
        if args.only and args.only not in e["name"]:
            continue
        done = Path(e.get("done", "{runs_dir}/" + e["name"] + "/done").format(**cfg.paths))
        if done.exists():
            print(f"[skip] {e['name']} ({done} exists)")
            continue
        cmd = [sys.executable, e["script"]]
        if not any(s in e["script"] for s in NO_NAME_SCRIPTS):
            cmd += ["--name", e["name"]]
        for c in configs:
            cmd += ["--config", c]
        for s in e.get("set", []):
            cmd += ["--set", s if args.dry_run else resolve(s, cfg)]
        print(f"[run ] {e['name']}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        res = subprocess.run(cmd)
        if res.returncode != 0:
            sys.exit(f"[fail] {e['name']} exited with {res.returncode}")
        print(f"[done] {e['name']} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
