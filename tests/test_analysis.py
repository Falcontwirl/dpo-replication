import numpy as np
import pytest

from dpo_rep.analysis import (fit_saturating, select_lr, frontier_envelope, interp_at, method_overhead, overoptimization,
                              plateau_index, run_stats, saturating, time_to_threshold)


def test_time_to_threshold():
    t = np.array([0, 10, 20, 30.0])
    r = np.array([0.5, 0.85, 0.92, 0.96])
    assert time_to_threshold(t, r, 0.9) == 20
    assert time_to_threshold(t, r, 0.99) is None


def test_fit_saturating_recovers_parameters():
    t = np.linspace(0, 1000, 25)
    r = saturating(t, 0.95, 0.4, 150.0)
    fit = fit_saturating(t, r)
    assert fit["a"] == pytest.approx(0.95, abs=1e-3)
    assert fit["tau"] == pytest.approx(150.0, rel=1e-2)
    assert fit["r2"] > 0.999


def test_plateau_index_on_saturating_curve():
    t = np.arange(0, 3000, 100.0)
    r = saturating(t, 0.95, 0.4, 300.0)
    i = plateau_index(r, eps=0.01, k=1)
    # 0.4 * exp(-t/300) < 0.01  <=>  t > 300 * ln 40 ~ 1107
    assert t[i] == 1200


def test_overoptimization_flag():
    t = np.arange(20.0)
    r = np.concatenate([np.linspace(0.5, 0.9, 10), np.linspace(0.9, 0.88, 10)])
    kl = np.linspace(0, 20, 20)
    assert overoptimization(t, r, kl)["flag"] is True
    assert overoptimization(t, np.linspace(0.5, 0.99, 20), kl)["flag"] is False


def test_frontier_envelope_and_interp():
    kl = np.array([1.0, 2.0, 5.0, 10.0])
    r = np.array([0.6, 0.9, 0.85, 0.95])
    env = frontier_envelope(kl, r, np.array([0.5, 2.0, 6.0, 20.0]))
    assert np.isnan(env[0]) and list(env[1:]) == [0.9, 0.9, 0.95]
    assert interp_at(np.array([0.0, 10.0]), np.array([0.0, 1.0]), 5.0) == 0.5
    assert interp_at(np.array([0.0, 10.0]), np.array([0.0, 1.0]), 11.0) is None


def test_method_overhead():
    oh = {"sft": 5.0, "prefs": 10.0, "rm": 3.0}
    assert method_overhead("dpo", oh) == 10.0
    assert method_overhead("ppo", oh) == 13.0
    assert method_overhead("ppo_gt", oh) == 0.0


def test_run_stats_smoke():
    rows = [{"method": "dpo", "beta": 0.1, "step": s, "t_train": s * 2.0,
             "reward_mean": float(saturating(s, 0.97, 0.3, 300)), "kl_exact": s / 200} for s in range(0, 2100, 100)]
    st = run_stats(rows, overhead=50.0)
    assert st["t_to_0.9"] is not None and st["t_to_0.9_with_overhead"] == st["t_to_0.9"] + 50.0
    assert st["fit_vs_step"]["tau"] == pytest.approx(300, rel=0.05)
    assert st["spearman_t_reward"]["rho"] == pytest.approx(1.0)


def test_select_lr_respects_kl_budget_and_ties():
    def rows(points):
        return [{"step": i, "reward_mean": r, "kl_exact": k} for i, (r, k) in enumerate(points)]
    cands = {
        1e-6: rows([(0.6, 0.0), (0.80, 2.0), (0.85, 4.0)]),
        3e-6: rows([(0.6, 0.0), (0.90, 5.0), (0.99, 9.0)]),   # 0.99 is over budget; 0.90 counts
        1e-5: rows([(0.6, 0.0), (0.95, 12.0)]),               # only the step-0 point is within budget
    }
    lr, table = select_lr(cands, select_kl=6.0)
    assert lr == 3e-6
    assert [t["best_reward_at_kl"] for t in table] == [0.85, 0.90, 0.6]
    tie = {1e-6: rows([(0.9, 1.0)]), 3e-6: rows([(0.9, 1.0)])}
    assert select_lr(tie, 6.0)[0] == 1e-6
