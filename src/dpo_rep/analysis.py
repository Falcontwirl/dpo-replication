"""Statistics for the reward-KL frontier and the training-time extension (CLAUDE.md §6).

All functions take plain arrays ordered by training progress (one entry per eval checkpoint).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

from dpo_rep.utils import read_jsonl

PLATEAU_EPS = 0.01      # reward units; ~1 standard error of the mean reward at 512 eval prompts (U12)
PLATEAU_SMOOTH = 3      # centered moving-average window applied before plateau detection
OVEROPT_TAIL = 0.3      # fraction of checkpoints treated as "late training" for the over-optimization check


TUNE_PREFIX = "tune_"   # PPO lr-tuning runs: evaluated on validation prompts, excluded from reported results


def load_runs(runs_dir: str | Path, tuning: bool = False) -> dict[str, list[dict]]:
    """Run directories whose metrics rows carry a `method` field (DPO / PPO / PPO-GT runs).
    tuning=False returns the reported runs; tuning=True returns only the lr-tuning runs."""
    out = {}
    for p in sorted(Path(runs_dir).glob("*/metrics.jsonl")):
        if p.parent.name.startswith(TUNE_PREFIX) != tuning:
            continue
        rows = [r for r in read_jsonl(p) if "method" in r and "reward_mean" in r]
        if rows:
            out[p.parent.name] = sorted(rows, key=lambda r: r["step"])
    return out


def load_overheads(runs_dir: str | Path, data_dir: str | Path) -> dict[str, float]:
    """One-off costs (seconds): SFT, preference-data generation, reward-model training."""
    oh = {"sft": 0.0, "prefs": 0.0, "rm": 0.0}
    sft = Path(runs_dir) / "sft" / "metrics.jsonl"
    if sft.exists():
        oh["sft"] = read_jsonl(sft)[-1]["t_train"]
    meta = Path(data_dir) / "prefs_meta.json"
    if meta.exists():
        oh["prefs"] = json.loads(meta.read_text())["t_total"]
    rm = Path(runs_dir) / "rm" / "metrics.jsonl"
    if rm.exists():
        final = [r for r in read_jsonl(rm) if r.get("final")]
        if final:
            oh["rm"] = final[-1]["t_cost"]
    return oh


def method_overhead(method: str, oh: dict[str, float]) -> float:
    """Cost a method must pay beyond its own training loop (SFT excluded: shared by all, reported separately).
    DPO needs the preference data; PPO (learned reward) needs the preference data and the reward model;
    PPO-GT needs neither."""
    return {"dpo": oh["prefs"], "ppo": oh["prefs"] + oh["rm"], "ppo_gt": 0.0}[method]


def time_to_threshold(t: np.ndarray, r: np.ndarray, threshold: float) -> float | None:
    hit = np.nonzero(r >= threshold)[0]
    return float(t[hit[0]]) if len(hit) else None


def smooth(x: np.ndarray, k: int = PLATEAU_SMOOTH) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x.astype(float)
    pad = k // 2
    xp = np.pad(x.astype(float), pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def plateau_index(r: np.ndarray, eps: float = PLATEAU_EPS, k: int = PLATEAU_SMOOTH) -> int:
    """First checkpoint i after which the smoothed reward never improves by more than eps:
    max_{j > i} rs[j] - rs[i] < eps. Returns the last index if the curve never flattens."""
    rs = smooth(r, k)
    for i in range(len(rs)):
        if i == len(rs) - 1 or rs[i + 1:].max() - rs[i] < eps:
            return i
    return len(rs) - 1


def marginal_gains(r: np.ndarray) -> np.ndarray:
    """Reward change per eval interval (DPO: per 100 steps)."""
    return np.diff(r)


def saturating(t, a, b, tau):
    return a - b * np.exp(-t / tau)


def fit_saturating(t: np.ndarray, r: np.ndarray) -> dict | None:
    """Least-squares fit of r(t) = a - b * exp(-t / tau); 95% CIs from the covariance (normal approx.)."""
    if len(t) < 4 or np.ptp(t) == 0:
        return None
    p0 = [r.max(), max(r.max() - r[0], 1e-3), max(np.median(t), 1e-6)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(saturating, t, r, p0=p0,
                                   bounds=([-np.inf, -np.inf, 1e-9], [np.inf, np.inf, np.inf]), maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    se = np.sqrt(np.clip(np.diag(pcov), 0, None))
    resid = r - saturating(t, *popt)
    ss_tot = ((r - r.mean()) ** 2).sum()
    return {"a": popt[0], "b": popt[1], "tau": popt[2],
            "a_ci": [popt[0] - 1.96 * se[0], popt[0] + 1.96 * se[0]],
            "tau_ci": [popt[2] - 1.96 * se[2], popt[2] + 1.96 * se[2]],
            "r2": 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else None}


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3 or np.ptp(y) == 0:
        return {"rho": None, "p": None}
    res = stats.spearmanr(x, y)
    return {"rho": float(res.statistic), "p": float(res.pvalue)}


def overoptimization(t: np.ndarray, r: np.ndarray, kl: np.ndarray, tail: float = OVEROPT_TAIL) -> dict:
    """Linear slopes of reward and KL over the last `tail` fraction of checkpoints. Flagged when KL rises
    significantly (p < 0.05) while reward does not (slope <= 0 or not significantly positive)."""
    n = max(3, int(np.ceil(len(t) * tail)))
    if len(t) < n or np.ptp(t[-n:]) == 0:
        return {"flag": None}
    rs, ks = stats.linregress(t[-n:], r[-n:]), stats.linregress(t[-n:], kl[-n:])
    kl_up = ks.slope > 0 and ks.pvalue < 0.05
    reward_up = rs.slope > 0 and rs.pvalue < 0.05
    return {"flag": bool(kl_up and not reward_up), "n_points": n, "reward_slope": rs.slope, "reward_p": rs.pvalue,
            "kl_slope": ks.slope, "kl_p": ks.pvalue}


def interp_at(t: np.ndarray, y: np.ndarray, budget: float) -> float | None:
    """Value at a time budget by linear interpolation; None outside the observed range."""
    if budget < t[0] or budget > t[-1]:
        return None
    return float(np.interp(budget, t, y))


def frontier_envelope(kl: np.ndarray, r: np.ndarray, kl_grid: np.ndarray) -> np.ndarray:
    """Best reward achieved at KL <= k for each k in kl_grid (NaN where no point qualifies)."""
    return np.array([r[kl <= k].max() if (kl <= k).any() else np.nan for k in kl_grid])


def run_stats(rows: list[dict], overhead: float, thresholds=(0.8, 0.9, 0.95)) -> dict:
    step = np.array([r["step"] for r in rows], float)
    t = np.array([r["t_train"] for r in rows], float)
    rew = np.array([r["reward_mean"] for r in rows], float)
    kl = np.array([r["kl_exact"] for r in rows], float)
    pi = plateau_index(rew)
    gains = marginal_gains(rew)
    out = {
        "method": rows[0]["method"], "param": rows[0].get("beta", rows[0].get("target_kl")),
        "n_evals": len(rows), "final_step": step[-1], "t_train_final": t[-1], "overhead_s": overhead,
        "final_reward": rew[-1], "final_kl": kl[-1], "max_reward": rew.max(),
        "plateau_step": step[pi], "plateau_t_train": t[pi], "plateau_reward": rew[pi], "plateau_kl": kl[pi],
        "post_plateau_kl_increase": kl[-1] - kl[pi],
        "mean_gain_per_eval_before_plateau": float(gains[:pi].mean()) if pi > 0 else None,
        "mean_gain_per_eval_after_plateau": float(gains[pi:].mean()) if pi < len(gains) else None,
        "fit_vs_t_train": fit_saturating(t, rew), "fit_vs_step": fit_saturating(step, rew),
        "spearman_t_reward": spearman(t, rew), "spearman_t_kl": spearman(t, kl),
        "overoptimization": overoptimization(t, rew, kl),
    }
    for thr in thresholds:
        tt = time_to_threshold(t, rew, thr)
        out[f"t_to_{thr}"] = tt
        out[f"t_to_{thr}_with_overhead"] = tt + overhead if tt is not None else None
        out[f"step_to_{thr}"] = time_to_threshold(step, rew, thr)
    return out


def select_lr(candidates: dict[float, list[dict]], select_kl: float) -> tuple[float, list[dict]]:
    """PPO lr selection (deviations.md U14): the lr whose run reaches the highest eval reward among
    checkpoints with KL <= select_kl. Ties (and runs with no qualifying checkpoint) go to the lower lr."""
    table = []
    for lr, rows in sorted(candidates.items()):
        ok = [r for r in rows if r["kl_exact"] <= select_kl]
        best = max(ok, key=lambda r: r["reward_mean"]) if ok else None
        table.append({"lr": lr, "best_reward_at_kl": best["reward_mean"] if best else None,
                      "best_kl": best["kl_exact"] if best else None, "best_step": best["step"] if best else None,
                      "final_reward": rows[-1]["reward_mean"], "final_kl": rows[-1]["kl_exact"]})
    scored = [t for t in table if t["best_reward_at_kl"] is not None]
    if not scored:
        raise ValueError("No tuning run has a checkpoint within the KL budget")
    winner = max(scored, key=lambda t: (t["best_reward_at_kl"], -t["lr"]))
    return winner["lr"], table
