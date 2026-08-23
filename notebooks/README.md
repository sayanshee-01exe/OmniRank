# Notebooks

**Exploration and reporting only.** No notebook is part of the pipeline.

The rule, borrowed from experience with notebook-driven projects: any logic a
notebook produces that something else needs must move into `src/omnirank/`
before the notebook is committed. A notebook that is the only place a
transformation exists cannot be tested, cannot be imported by the serving path,
and is the most reliable source of training/serving skew there is.

Naming: `<order>_<topic>.ipynb`, e.g. `1_dataset_overview.ipynb`.

## Phase 1 status

Empty — there is no data to explore yet.
