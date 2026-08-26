# Phase 5 closure — consistency pass

**Date:** 2026-08-26 · **Scope:** documentation and reporting consistency only ·
**Result:** both validator modes exit 0

This is a short record of a reconciliation pass. It does not repeat the Phase 5
report — see [phase_05_report.md](phase_05_report.md) for the full evidence, and
`reports/metrics/phase_05/` for the machine-readable numbers behind it.

**No model was retrained and no recommendation result changed.** The only code
touched was report *generation*; every metric file is byte-identical to what the
experiments produced.

---

## 1. Why this pass was needed

Phase 5's final model was refitted late in the phase, after a defect was found
in which the registered "final" artifact had been fitted with the development
`--subset-users 5000` default and could answer for 5,000 of 50,000 users. The
refit changed every headline number and **reversed the phase's conclusion**.

Most of the report regenerates from metric files and so corrected itself. Two
passages did not, because they were hand-written prose rather than derived
values, and they survived the refit while the numbers around them changed. Both
ended up contradicting tables printed a few lines above them.

That is the failure mode this pass exists to close: prose that outlives the
numbers it describes is worse than no prose, because it carries the authority of
the surrounding document.

---

## 2. Contradictions found and corrected

| Where | Prose claimed | Metric files say | Fix |
|---|---|---|---|
| §26 Final cold metrics | cold Recall was "zero at K=5 and K=10" | Recall@5 **0.008811**, Recall@10 **0.012775** — positive at every measured cutoff | Derived from the cold table via `_cold_positivity` |
| §48.1 Known limitations | NDCG@20 "is significantly **below** LightGCN's" | +0.002765 **above**, 95% CI [+0.002038, +0.003509] — significantly *above*, as §29, §33 and §50 already said | Derived from the standalone ranking via `_accuracy_limitation` |

Both are now computed from the same data that fills the adjacent tables, so
neither can drift again. `tests/unit/scripts/test_phase5_report.py` pins both,
including the "genuine zero" and "genuinely trailing" branches so the fix cannot
be inverted into a different overstatement.

---

## 3. Authoritative sources used

In the priority order the closure brief specifies:

1. `reports/metrics/phase_05/two_tower_final_test_metrics.json` — strict, warm
   and cold views of the official final test
2. `reports/metrics/phase_05/five_source_fusion_metrics.csv` — all ten systems
3. `reports/metrics/phase_05/bootstrap_deltas.csv` — 18 paired comparisons
4. `reports/metrics/phase_05/candidate_recall.csv`,
   `two_tower_unique_contribution.json`, `index_benchmark.csv`,
   `missing_modality_metrics.csv`
5. `artifacts/metadata/two_tower/phase5-two-tower-final.json` and the embedding
   and index manifests

Every README and report number was re-read from these files during this pass
rather than carried over.

---

## 4. Corrected headline metrics

Official final test, fitted on train+validation, scored once:

| Metric | Two-tower | Best collaborative | Source |
|---|---|---|---|
| strict NDCG@20 | **0.008873** | 0.006108 (lightgcn) | final test metrics |
| strict Recall@20 | **0.021860** | 0.014780 (lightgcn) | final test metrics |
| Coverage@20 | **0.79431** | 0.461994 (sasrec) | fusion metrics |
| exposure Gini@20 | **0.758331** (lowest) | 0.879600 (mf) | fusion metrics |
| cold Recall@20 | **0.018062** | 0.001322 (lightgcn) | final test metrics |
| unreachable cold users | **0** | 724 (all four) | fusion metrics |

Fusion, labelled by metric as the brief requires:

- Five-source RRF vs four-source RRF: **NDCG@20 +0.003409** (CI [+0.003001,
  +0.003802]), **Recall@20 +0.007260** (CI [+0.006300, +0.008260]),
  **cold Recall@20 +0.000441** (CI [+0.000000, +0.001322]).
- Candidate Recall@budget-500: four-source 0.21890 → five-source **0.28874**.
- Targets reached by no other source: **3,492**.

---

## 5. Corrected significance statements

Read from `bootstrap_deltas.csv`, never asserted:

**Statistically reliable** (interval excludes zero) — the two-tower beats
popularity, BPR and LightGCN on NDCG@20, Recall@20 *and* cold Recall@20; the
five-source blend beats the four-source blend and LightGCN on NDCG@20 and
Recall@20; LightGCN+two-tower beats LightGCN on all three.

**Not statistically reliable** (interval includes zero) — the five-source
blend's **cold Recall@20** gain over both four-source RRF and LightGCN
(+0.000441, CI [+0.000000, +0.001322]). This is stated plainly in §31 and §33
and is *not* described as an improvement.

The distinction matters: the two-tower moves cold recall enormously **on its
own** (13.7× LightGCN, interval excluding zero), but diluting it into a
five-way uniform blend does **not** produce a reliable cold gain. What fusion
reliably buys is warm accuracy and reachability, not cold ranking.

---

## 6. Documentation changes

**README** — CI section now shows the exact `set -o pipefail` + `tee`
invocation the workflow runs, with the reason the pipeline guard is
load-bearing. No stale phase claims were found; the roadmap, implementation
table and every metric already matched the metric files and were left alone.

**`docs/architecture/system_architecture.md`** — the pipeline diagram now marks
the five candidate generators as implemented (✅ with phase), names reciprocal
rank fusion explicitly, and draws a divider at **"CURRENT END OF THE
PIPELINE"**. Everything below it — feature builder, LambdaRank, post-ranking
filters, MMR, online serving — is marked 📋 Phase 6, contract only.

**`docs/phase_reports/phase_04_report.md`** — links forward to the Phase 5
report and this closure record.

**`docs/phase_reports/phase_05a_report.md`** — banner marking it superseded for
results. Its metrics predate the final model and must not be quoted.

**`src/omnirank/models/two_tower/__init__.py`** — verified already accurate
(`PHASE: 5 - complete`, describing the implemented components). No change
needed; public exports untouched.

---

## 7. Provenance limitation

The registered artifact records `git_commit: 42ad33e`, while HEAD at closure is
`6b92287`. Model version, configuration hash, mapping checksum, feature version
and dataset version all match across the model, embedding and index manifests,
and the artifact's recorded metrics are identical to the metric files.

The lag is expected and benign: training ran against `42ad33e` plus the
working-tree changes that `6b92287` then captured. **It is recorded here rather
than papered over**, and it was not grounds for retraining — the brief is
explicit that expensive training must not be re-run to update prose, and nothing
about the artifact is incompatible.

---

## 8. Verification

| Check | Result |
|---|---|
| `ruff format --check .` | 275 files already formatted |
| `ruff check .` | All checks passed |
| `mypy --strict src` | no issues in 110 source files |
| `pytest` | **1,671 passed** |
| `python scripts/validate_phase5.py --ci` | exit **0** — 22/35, 13 real-data checks SKIP |
| `python scripts/validate_phase5.py` | exit **0** — **30/30**, no failures, warnings or skips |

CI mode records every real-data check as SKIP rather than PASS, and stamps
`mode: ci` in the JSON report, so a green badge cannot be mistaken for evidence
of real completion.

---

## 9. Remaining technical debt

Carried forward unchanged from §49 of the Phase 5 report:

- Two independent scorers (split evaluation via `run_experiment`, fold
  evaluation in `fold_evaluation.py`) kept in agreement by test rather than by
  construction.
- `--reuse-screen` / `--reuse-folds` trust the CSV's contents without verifying
  it came from the current code.
- `KMP_DUPLICATE_LIB_OK` for FAISS/torch coexistence (ADR-009) — a workaround,
  justified by the exact brute-force check, not a fix.
- Weighted-RRF weights are fixed in advance rather than searched; the weighted
  variant scored **below** uniform and is reported as a negative result.
- Fusion diagnostics issue their own retrieval sweeps rather than reusing the
  scoring pass.

Added by this pass: none. No code behaviour changed.

---

## 10. Phase 6 starting point

Build the ranking dataset from the five-source RRF candidate pool at
**per-source budget 500**, using the registered artifacts as frozen inputs.

The budget is not arbitrary: `candidate_recall.csv` shows five-source candidate
recall reaching 0.28874 at budget 500 and **exactly 0.28874 again at budget
1200**. The sources saturate, so depth past 500 costs latency and returns
nothing.

Three properties the ranker must handle, all measured in Phase 5:

1. **A hard ceiling.** No ranker recovers a target retrieval never proposed;
   0.28874 is the upper bound on anything ranking can achieve at this budget.
2. **Cold items in candidate sets for the first time.** Any feature that divides
   by an interaction count will fail on them.
3. **A two-tower ordering now worth using** — it is the strongest single source
   here — but that status is one refit old and rests on one corpus, so validate
   any feature derived from its score rather than assuming it.

Phase 6 has **not** been started.
