"""Stage 5b: pick the PPO learning rate from the tuning runs (deviations.md U14).

Reads <runs_dir>/tune_ppo_gt_lr*/metrics.jsonl (evaluated on validation prompts from IMDb's unlabeled split),
picks the lr with the best eval reward at KL <= ppo_tune.select_kl, and writes <runs_dir>/ppo_lr_choice.json.
06_sweep.py substitutes that value for `@ppo_lr` in the main PPO / PPO-GT entries.
"""

import json
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from dpo_rep.analysis import load_runs, select_lr
from dpo_rep.config import parse_args


def tune_run_name(lr: float, reward: str) -> str:
    return f"tune_ppo_{reward}_lr{lr:g}"


def main():
    cfg, _ = parse_args(__doc__)
    t = cfg.ppo_tune
    runs = load_runs(cfg.paths.runs_dir, tuning=True)
    candidates = {}
    for lr in t.lrs:
        name = tune_run_name(lr, t.reward)
        if name not in runs:
            raise SystemExit(f"Missing tuning run {name} in {cfg.paths.runs_dir}")
        candidates[float(lr)] = runs[name]
    lr, table = select_lr(candidates, t.select_kl)
    out = {"lr": lr, "select_kl": t.select_kl, "reward": t.reward, "target_kl": t.target_kl,
           "total_ppo_steps": t.total_ppo_steps, "eval_split": t.eval_split, "candidates": table}
    path = Path(cfg.paths.runs_dir) / "ppo_lr_choice.json"
    path.write_text(json.dumps(out, indent=2))
    for row in table:
        print(row)
    print(f"Selected PPO lr = {lr:g}  -> {path}")
    lrs = sorted(float(x) for x in t.lrs)
    if lr in (lrs[0], lrs[-1]) and len(lrs) > 1:
        print(f"WARNING: the selected lr is at the edge of the grid {lrs}; the optimum may lie outside it. "
              f"Consider extending ppo_tune.lrs in that direction and re-running the tuning entries.")


if __name__ == "__main__":
    main()
