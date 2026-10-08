# DPO Replication — IMDb Controlled Sentiment (DPO vs PPO)

Replication of **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"**
(Rafailov, Sharma, Mitchell, Ermon, Manning, Finn — NeurIPS 2023). arXiv: https://arxiv.org/abs/2305.18290

## 1. Project goal

- Replicate **Experiment 1 (Sec 6.1, Fig 2 left)**: the reward-vs-KL frontier on IMDb controlled sentiment generation, comparing **DPO against PPO** (learned reward) and **PPO-GT** (ground-truth reward).
- **Extension (not in paper):** quantify the relationship between extra training time and performance (spec in §6).
- Out of scope: TL;DR summarization, Anthropic-HH dialogue, CNN/DailyMail OOD, human study (summarized in §7 for context only).

## 2. Working rule: track deviations

**Every** difference from the paper — hyperparameters, models, data, evaluation, hardware, or a choice for something the paper leaves unspecified — must be recorded in [`deviations.md`](deviations.md) in the same change that introduces it. Never silently diverge from the paper, and never silently "correct" a documented deviation back. If unsure whether something is a deviation, log it.

**From-scratch rule (deviations.md D5).**
- All algorithms are our own code in plain PyTorch.
- HF `transformers` / `datasets` may be used **only** to load pretrained weights, tokenizers and IMDb. Not for `generate`, `Trainer`, or any loss.
- **Do not use TRL**, and **do not consult or copy** the authors' reference repo.

## Code & commands

```
configs/base.yaml     full-scale config (cloud GPU); [paper] values are tagged, the rest map to deviations.md U-rows
configs/smoke.yaml    tiny overrides (gpt2-small, ~100 prefixes); writes to runs_smoke/ data_smoke/ results_smoke/
configs/sweep.yaml    the full experiment in order: sft -> prefs -> rm -> 4 DPO -> PPO lr tuning (3 runs) + selection
                      -> 4 PPO + 4 PPO-GT runs using the selected lr (deviations.md U14)
src/dpo_rep/          library: config, utils (timing/logging), modeling (log-probs, value head, RM), sampling
                      (own KV-cache sampler), kl, losses (DPO/BT/PPO/GAE), data, gt_reward, evaluate, ppo, analysis
scripts/01..07        stages: sft, gen_prefs, train_rm, train_dpo, train_ppo, 05b select_ppo_lr, sweep, analyze
tests/                CPU unit tests on a tiny random GPT-2 (no downloads); the DPO test embeds the App B code as an oracle
```

- **Setup:**
  - `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`
  - Scripts import via `scripts/_bootstrap.py`, which puts `src/` on the path. This matters on macOS, where Python 3.14 ignores the editable install's `.pth` because the OS flags it hidden.
- **Tests:** `.venv/bin/python -m pytest`
- **Smoke run (Mac):** `.venv/bin/python scripts/06_sweep.py --config configs/base.yaml --config configs/smoke.yaml`, then `scripts/07_analyze.py` with the same configs.
- **RunPod A100 (the full-run machine):**
  - Clone the repo under `/workspace`, the persistent volume.
  - `bash scripts/runpod.sh setup` installs deps, runs the tests and does a GPU smoke run.
  - `bash scripts/runpod.sh sweep` launches the full sweep in the background, logging to `logs/sweep.log`.
  - `status` and `analyze` are the other subcommands.
  - `HF_HOME` defaults to `/workspace/hf_cache`.
- **Full run (any GPU):**
  - Run `python scripts/06_sweep.py`; it skips finished entries, so it can be re-launched after an interruption.
  - Then run `python scripts/07_analyze.py`. Results go to `results/`.
- **Single stage:**
  - Example: `python scripts/04_train_dpo.py --set dpo.beta=0.5 --name dpo_beta0.5`.
  - Any config key can be overridden with `--set section.key=value`.
- **Per-run outputs:** `runs/<name>/metrics.jsonl` has one row per eval with:
  - `step`, `examples_seen`, `tokens`
  - timing: `t_train` (training only), `t_eval`, `t_total`, and for PPO the breakdown `t_rollout` / `t_score` / `t_forward` / `t_update`
  - results: `reward_mean` / `reward_se`, `kl_exact` / `kl_sample`
  - training statistics and `gpu_name`

  A crashed run must have its directory deleted before re-running, because `make_run_dir` refuses to overwrite metrics.

## 3. Method

**RLHF objective (Eq 3)** — KL-constrained reward maximization:

$$\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal D,\,y\sim\pi_\theta(y|x)}[r(x,y)] - \beta\,\mathbb D_{KL}[\pi_\theta(y|x)\,\|\,\pi_{ref}(y|x)]$$

**Optimal policy (Eq 4):** $\pi_r(y|x) = \frac{1}{Z(x)}\pi_{ref}(y|x)\exp\!\big(\tfrac{1}{\beta}r(x,y)\big)$

**Implicit reward (Eq 5):** $r(x,y) = \beta\log\frac{\pi_r(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)$. Plugging into Bradley–Terry, $Z(x)$ cancels.

**DPO loss (Eq 7):**

$$\mathcal L_{DPO} = -\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big]$$

**Gradient:** with $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$,

$$\nabla_\theta\mathcal L_{DPO} = -\beta\,\mathbb E\big[\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))\,(\nabla_\theta\log\pi(y_w|x)-\nabla_\theta\log\pi(y_l|x))\big]$$

The σ term up-weights pairs the implicit reward currently ranks wrongly; without it the model degenerates (App C.3 / Table 3).

- $\pi_{ref} = \pi^{SFT}$; the policy is initialized from $\pi^{SFT}$.
- Log-probs are **sequence-level** (sum over completion tokens, prompt tokens excluded).

**Reference implementation (App B, verbatim):**

```python
import torch.nn.functional as F

def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
    """
    pi_logps: policy logprobs, shape (B,)
    ref_logps: reference model logprobs, shape (B,)
    yw_idxs: preferred completion indices in [0, B-1], shape (T,)
    yl_idxs: dispreferred completion indices in [0, B-1], shape (T,)
    beta: temperature controlling strength of KL penalty

    Each pair of (yw_idxs[i], yl_idxs[i]) represents the
      indices of a single preference pair.
    """

    pi_yw_logps,  pi_yl_logps =  pi_logps[yw_idxs],  pi_logps[yl_idxs]
    ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]

    pi_logratios  = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps

    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards = beta * (pi_logps - ref_logps).detach()

    return losses, rewards
```

## 4. Experiment 1 spec (paper values)

Sources: Sec 6 "Tasks"/"Methods" (p.7–8), Sec 6.1 (p.8), App B (p.20), App C.1 (p.20).

**Task.** $x$ = prefix of an IMDb movie review (Maas et al. 2011), **2–8 tokens** long. Policy must continue it with **positive sentiment**.

**Ground-truth reward.** `siebert/sentiment-roberta-large-english`; reward = $p(\text{positive}\mid x,y)$. (Authors chose this over smaller default models because those "generate low-quality text and rewards to be somewhat inaccurate".)

**Pipeline:**
1. **SFT:** `gpt2-large` fine-tuned on a subset of IMDb **train** split for **1 epoch** → $\pi^{SFT}$ (= $\pi_{ref}$).
2. **Preference data:** $\pi^{SFT}$ samples **4 completions** for each of **25,000 prefixes**; form **6 pairs per prefix** (all $\binom{4}{2}$) labeled by the ground-truth reward: $y_w$ is the completion with higher $p(\text{positive})$. ≈150k pairs.
3. **Reward model (for PPO):** initialized from `gpt2-large`, trained **3 epochs** on the preference pairs; keep the checkpoint with **best validation accuracy**.
4. **Train methods** (sweep over conservativeness hyperparameter):

| Method | Description | Sweep (paper) |
|---|---|---|
| **DPO** | Eq 7 on the preference pairs | β ∈ {0.05, 0.1, 1, 5} |
| **PPO** | PPO with learned reward model | target KL ∈ {3, 6, 9, 12} |
| **PPO-GT (our impl.)** | PPO on ground-truth classifier reward; authors' modified version: **normalized rewards**, further-tuned hyperparameters, **1024 samples per PPO step** (same modifications used for "normal" PPO) | target KL (presumably same) |
| **PPO-GT (TRL)** *(not replicating, no TRL; see D2/D5)* | Off-the-shelf TRL PPO with TRL default hyperparameters, GT reward | — |
| Unlikelihood *(not replicating)* | max log p(y_w), min α·log p(y_l) | α ∈ {0.05, 0.1, 0.5, 1} |
| Preferred-FT *(not replicating)* | SFT on y_w only | random seeds |

Paper total: **22 runs**.

**DPO optimizer defaults (App B):** batch size **64**, **RMSprop**, lr **1e-6**, linear warmup 0 → 1e-6 over **150 steps**. (β=0.1 is the global default; the sentiment sweep overrides it.)

**Evaluation.**
- **Every 100 training steps, until convergence**, evaluate the policy on a set of **test prompts**. Use the IMDb test split, which is implied but not stated.
- Record two values:
  - Mean **ground-truth reward** of samples.
  - Mean **sequence-level KL(π_θ ‖ π_ref)**, i.e. the **sum of per-timestep KL divergences** (footnote 3).
- Each eval is one point in the reward (y) vs KL (x) scatter. Figure axes: KL 0–20, reward ≈0.4–1.0.

**Target result (Fig 2 left; numbers read off the plot, approximate):**
- **DPO frontier strictly dominates** all others: reward ≈0.9 by KL ≈1–2, ≈0.95–1.0 at KL ≈3–5.
- PPO (our impl.) and PPO-GT (our impl.) climb more slowly. They reach ≈0.85–0.9 only around KL ≈10–12.
- PPO-GT (TRL) is worse still: ≈0.7–0.8 at KL ≈10–17.
- Key claims to reproduce:
  1. DPO and PPO optimize the same objective, but **DPO's reward/KL tradeoff strictly dominates PPO's**.
  2. **DPO beats PPO even when PPO has ground-truth rewards** (PPO-GT).

## 5. Unspecified in the paper (we must choose — log each choice in deviations.md)

Our current choices are in deviations.md (U1–U13), and the values are in `configs/base.yaml`.

- SFT: size of the IMDb train "subset", max sequence length, lr/batch/optimizer.
- How 2–8 token prefixes are sampled (uniform length? which reviews?), and whether SFT-sampling prefixes overlap SFT training text.
- Sampling settings for preference-data generation (temperature, top-k/p, max new tokens).
- Tie handling when two completions get equal classifier score.
- Train/val split for the reward model; RM optimizer/lr/batch.
- PPO hyperparameters beyond target KL (lr, epochs per batch, minibatch size, value head, GAE λ/γ, clip range, adaptive-KL controller details, reward normalization method).
- DPO: number of epochs / total steps, max lengths, precision.
- Eval: number of test prompts, prefix sampling for eval, generation settings, KL estimator (exact per-token KL over vocab vs. sample-based log-ratio), "until convergence" stopping rule.
- Number of seeds per configuration.
- Hardware (not reported).

## 6. Extension: training time vs performance (our addition — see deviations.md D3)

Logging must be **purely observational**: it must not change any training dynamics, schedules, or stopping.

**Log at every eval checkpoint (every 100 steps), per run:**
- step
- examples / preference pairs seen
- tokens processed
- wall-clock time, recorded twice: training-only (excluding eval) and cumulative total
- mean true reward (± std/SE)
- mean KL
- reward/KL ratio

**Per-run statistics:**
- Curves: reward and KL vs steps and vs wall-clock.
- Steps and wall-clock to first reach reward thresholds **0.8 / 0.9 / 0.95**.
- Marginal reward gain per additional 100 steps; **plateau point**, defined as the first checkpoint after which the gain stays below a small ε. Define ε when implementing and log it.
- Saturating fit $r(t) = a - b\,e^{-t/\tau}$; report $a$ (asymptote) and $\tau$ (time constant) with CIs.
- **Spearman ρ** of training time vs reward and training time vs KL.
- **Over-optimization check:** late-training segments where KL keeps rising while reward is flat or falling.

**Cross-method statistics (DPO vs PPO / PPO-GT):**
- **Compute-matched comparison:** reward and KL at equal wall-clock budgets.
  - PPO's cost includes its per-step rollout generation and its reward-model training.
  - DPO's one-off preference-data generation cost is reported both **amortized** and **excluded**.
  - SFT cost is shared by both methods; report it separately.
- How the reward–KL frontier position evolves with training time for each β / target-KL.

## 7. Other experiments in the paper (context only — out of scope)

- **TL;DR summarization (Sec 6.2, Fig 2 right).**
  - Setup:
    - Model: GPT-J 6B SFT (`CarperAI/openai_summarize_tldr_sft`), β = 0.5.
    - Training data: Stiennon et al. human comparisons (off-policy).
    - Eval: win rate vs reference summaries at temperatures 0–1, judged by `gpt-4-0314` with the "concise" prompt (C).
  - Results: DPO ≈61% at temp 0 vs PPO ≈57% at temp 0. Best-of-128 ≈57% at temp 0.5. PPO collapses at high temperature.
- **Anthropic-HH single-turn dialogue (Sec 6.2, Fig 3).**
  - Setup: Pythia-2.8B; Preferred-FT on the chosen responses serves as π_ref; β = 0.1.
  - Eval: GPT-4 win rate vs the chosen responses.
  - Result: DPO ≈60%+ at temperatures 0.7–1.0. It is the only efficient method to beat the chosen responses, and is comparable to Best-of-128.
- **CNN/DailyMail OOD (Sec 6.3, Table 1).** The TL;DR policies are applied to news articles. Win rate vs ground truth: DPO 0.36/0.31 vs PPO 0.26/0.23 (temperatures 0/0.25).
- **Human study (Sec 6.4, Table 2).** GPT-4 (C) agrees with humans about as often as humans agree with each other (≈67–85% vs ≈65–87%).

## 8. Environment & references

- Local machine: Apple **M2, 24 GB unified memory, no CUDA** (MPS only), used for development and smoke tests only.
- Full runs: **one NVIDIA A100 on RunPod** (see deviations.md D4). The full sweep is estimated at ~20–28 GPU-hours, which is not yet measured. Python 3.14 locally; torch 2.14, transformers 5.x (the v5 API uses `dtype=`, not `torch_dtype=`).
- Paper: https://arxiv.org/abs/2305.18290 (v3, 29 Jul 2024).
- Authors' reference code (`eric-mitchell/direct-preference-optimization`) and TRL are **intentionally not used** (from-scratch rule, §2).
