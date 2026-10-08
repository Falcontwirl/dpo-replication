"""Stage 7: Fig 2-left frontier + training-time statistics (CLAUDE.md §6).

Outputs in <results_dir>/:
  frontier.png             reward vs KL, every eval checkpoint of every run (paper Fig 2 left)
  frontier_by_time.png     same points per method, shaded by training time
  curves_time.png          reward and KL vs training time, without / with one-off costs (prefs, RM)
  frontier_envelope.csv    best reward at KL <= k, per method (tests "DPO's frontier dominates")
  compute_matched.csv      reward and KL of every run at equal wall-clock budgets
  run_stats.csv            per-run statistics (flattened); time_stats.json has the full nested version
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors  # noqa: E402
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _bootstrap  # noqa: F401,E402  (adds src/ to sys.path)
from dpo_rep.analysis import (PLATEAU_EPS, frontier_envelope, interp_at, load_overheads, load_runs,  # noqa: E402
                              method_overhead, run_stats)
from dpo_rep.config import parse_args  # noqa: E402

# Validated categorical palette (dataviz reference slots 1-3) + marker shape as secondary encoding.
STYLE = {"dpo": {"color": "#2a78d6", "marker": "o", "label": "DPO"},
         "ppo": {"color": "#eb6834", "marker": "s", "label": "PPO (learned reward)"},
         "ppo_gt": {"color": "#1baf7a", "marker": "^", "label": "PPO-GT (true reward)"}}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
# Single-hue sequential ramp, starting at a visible light blue so t=0 points do not vanish into the surface.
TIME_CMAP = matplotlib.colors.ListedColormap(plt.cm.Blues(np.linspace(0.25, 1.0, 256)))


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def param_label(method, param):
    return f"β={param:g}" if method == "dpo" else f"KL*={param:g}"


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        elif isinstance(v, list):
            out[f"{prefix}{k}"] = json.dumps([float(x) for x in v])
        else:
            out[f"{prefix}{k}"] = v
    return out


def main():
    cfg, _ = parse_args(__doc__)
    runs = load_runs(cfg.paths.runs_dir)
    if not runs:
        raise SystemExit(f"No DPO/PPO runs with metrics in {cfg.paths.runs_dir}")
    oh = load_overheads(cfg.paths.runs_dir, cfg.paths.data_dir)
    out = Path(cfg.paths.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = [m for m in STYLE if any(r[0]["method"] == m for r in runs.values())]

    # ---- per-run statistics
    stats = {name: run_stats(rows, method_overhead(rows[0]["method"], oh)) for name, rows in runs.items()}
    (out / "time_stats.json").write_text(json.dumps({"overheads_s": oh, "plateau_eps": PLATEAU_EPS, "runs": stats},
                                                    indent=2, default=float))
    pd.DataFrame([{"run": n, **flatten(s)} for n, s in stats.items()]).to_csv(out / "run_stats.csv", index=False)

    # ---- frontier envelope per method
    kl_grid = np.arange(0.5, 20.01, 0.5)
    env = {"kl_max": kl_grid}
    for m in methods:
        pts = [(r["kl_exact"], r["reward_mean"]) for rows in runs.values() if rows[0]["method"] == m for r in rows]
        kl, rew = map(np.array, zip(*pts))
        env[m] = frontier_envelope(kl, rew, kl_grid)
    pd.DataFrame(env).to_csv(out / "frontier_envelope.csv", index=False)

    # ---- compute-matched comparison
    t_all = [r["t_train"] for rows in runs.values() for r in rows if r["t_train"] > 0]
    budgets = np.geomspace(min(t_all), max(t_all) + max(oh["prefs"] + oh["rm"], 1e-9), 12)
    cm_rows = []
    for name, rows in runs.items():
        m = rows[0]["method"]
        t = np.array([r["t_train"] for r in rows])
        rew = np.array([r["reward_mean"] for r in rows])
        kl = np.array([r["kl_exact"] for r in rows])
        for b in budgets:
            for incl, offset in (("excluded", 0.0), ("included", method_overhead(m, oh))):
                cm_rows.append({"run": name, "method": m, "overhead": incl, "budget_s": b,
                                "reward": interp_at(t + offset, rew, b), "kl": interp_at(t + offset, kl, b)})
    pd.DataFrame(cm_rows).to_csv(out / "compute_matched.csv", index=False)

    # ---- Fig 2-left frontier
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    for m in methods:
        st = STYLE[m]
        pts = [(r["kl_exact"], r["reward_mean"]) for rows in runs.values() if rows[0]["method"] == m for r in rows]
        kl, rew = map(np.array, zip(*pts))
        ax.scatter(kl, rew, s=22, color=st["color"], marker=st["marker"], label=st["label"],
                   edgecolors="white", linewidths=0.6, zorder=3)
        order = np.argsort(kl)  # envelope: best reward reached at KL <= x, drawn only over observed KL
        ax.step(kl[order], np.maximum.accumulate(rew[order]), where="post", color=st["color"], linewidth=2,
                alpha=0.5, zorder=2)
    ax.set_xlabel("KL(π_θ ‖ π_ref)  [sum of per-token KL]")
    ax.set_ylabel("Reward  [p(positive)]")
    ax.set_title("IMDb sentiment: reward vs KL frontier", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out / "frontier.png")
    plt.close(fig)

    # ---- frontier shaded by training time (small multiples, one sequential hue)
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.8), dpi=150, sharex=True, sharey=True,
                             squeeze=False)
    t_max = max(t_all) / 60 if t_all else 1
    for ax, m in zip(axes[0], methods):
        for rows in (rows for rows in runs.values() if rows[0]["method"] == m):
            sc = ax.scatter([r["kl_exact"] for r in rows], [r["reward_mean"] for r in rows],
                            c=[r["t_train"] / 60 for r in rows], cmap=TIME_CMAP, vmin=0, vmax=t_max,
                            marker=STYLE[m]["marker"], s=24, edgecolors=MUTED, linewidths=0.4, zorder=3)
        ax.set_title(STYLE[m]["label"], color=INK, fontsize=10, loc="left")
        ax.set_xlabel("KL(π_θ ‖ π_ref)")
        style_axes(ax)
    axes[0][0].set_ylabel("Reward")
    fig.colorbar(sc, ax=axes[0].tolist(), label="training time (min, excl. eval)", shrink=0.85)
    fig.savefig(out / "frontier_by_time.png", bbox_inches="tight")
    plt.close(fig)

    # ---- reward / KL vs training time, without and with one-off costs
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=150, sharex="col")
    for col, incl in enumerate(("excluded", "included")):
        for name, rows in runs.items():
            m = rows[0]["method"]
            st = STYLE[m]
            offset = method_overhead(m, oh) if incl == "included" else 0.0
            t = (np.array([r["t_train"] for r in rows]) + offset) / 60
            for row, key in enumerate(("reward_mean", "kl_exact")):
                y = [r[key] for r in rows]
                axes[row][col].plot(t, y, color=st["color"], linewidth=2, marker=st["marker"], markersize=4)
                if row == 0:
                    axes[row][col].annotate(param_label(m, stats[name]["param"]), (t[-1], y[-1]), xytext=(4, 0),
                                            textcoords="offset points", fontsize=7, color=MUTED, va="center")
        axes[0][col].set_title("training time only" if incl == "excluded"
                               else "including one-off costs (pref. data; + RM for PPO)", color=INK, fontsize=10, loc="left")
        axes[1][col].set_xlabel("wall-clock minutes")
    axes[0][0].set_ylabel("Reward")
    axes[1][0].set_ylabel("KL(π_θ ‖ π_ref)")
    for ax in axes.flat:
        style_axes(ax)
    handles = [matplotlib.lines.Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], linewidth=2,
                                       label=STYLE[m]["label"]) for m in methods]
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="upper center", ncol=len(methods),
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, f"SFT cost shared by all methods, not shown: {oh['sft'] / 60:.1f} min. "
             f"GPU: {next(iter(runs.values()))[0].get('gpu_name')}", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out / "curves_time.png")
    plt.close(fig)

    # ---- console summary
    print(f"One-off costs (s): {oh}")
    print(pd.DataFrame(env).iloc[[1, 3, 5, 9, 19, 39]].to_string(index=False))
    cols = ["method", "param", "final_reward", "final_kl", "t_to_0.9", "plateau_step", "plateau_t_train",
            "post_plateau_kl_increase"]
    print(pd.DataFrame([{"run": n, **{c: s[c] for c in cols}} for n, s in stats.items()]).to_string(index=False))
    print(f"Wrote results to {out}/")


if __name__ == "__main__":
    main()
