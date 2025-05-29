
# FINE 695 – Quant ML Project

Granular, test‑driven task list (single‑concern, verifiable steps). After **every** task run a unit or smoke test.

---

## Legend

- **Start** – observable cue you have begun.
    
- **End / Test** – objective condition that flips the card to _Done_.
    

---

## 0  Repository bootstrap

|#|Task|Start|End / Test|
|---|---|---|---|
|0.1|Create private Git repo `fine695-ml-portfolio`|VS Code open|`git status` clean, remote `main` exists|
|0.2|Add `.editorconfig`, `.gitignore` (Python, QC, Lightning, VSCode, parquet‑cache)|repo exists|First commit contains only intended files|
|0.3|Add GitHub Actions `pytest.yml` running `pytest -q`|workflow pushed|CI green on initial push|

---

## 1  Dev & cloud environments

|#|Task|Start|End / Test|
|---|---|---|---|
|1.1|Create Conda env `fine695` (`python 3.11`) + install: numpy, pandas, pyarrow, scikit‑learn, pytorch, lightning, lightgbm, catboost, ngboost, xgboost, pytorch‑tabnet, saint‑torch, node‑torch, autogluon, lean‑cli|terminal|`python -c "import lightgbm,catboost,autogluon.tabular"` exits 0|
|1.2|Enable Cursor AI in VS Code, set formatter to black|VSCode open|Saving file auto‑formats|
|1.3|`lightning login` + `lightning whoami`|shell|Token verified|
|1.4|`lean login` (QuantConnect)|shell|`lean status` → logged in|

---

## 2  Course data ingestion (local)

|#|Task|Start|End / Test|
|---|---|---|---|
|2.0|Copy **`group_project_sample_v3.csv`** (> 1 GB) into `data/raw/`|file present|`ls -lh` shows > 1 GB|
|2.1|Convert CSV to Apache Parquet via chunked read → `data/raw/course_panel.parquet`|CSV exists|Parquet size < 300 MB, row count logged|
|2.2|Inspect column list; assert presence of `stock_exret` + 147 predictors|parquet ready|Unit test `test_columns.py` passes|
|2.3|Load **`mkd_ind1.csv`** (columns `RF`,`year`,`month`,`sp_ret`) to `data/raw/mkt.parquet`|file present|Parquet rows ≥ 300|
|2.4|Sanity check mkt file: `RF` non‑null, `sp_ret` floats|parquet ready|`pytest tests/test_mkt.py` green|

---

## 3  (OPTIONAL) supplemental data from QuantConnect

|#|Task|Start|End / Test|
|---|---|---|---|
|3.1|QC Research notebook: pull Morningstar fundamentals 1998‑2024 for _extra_ features (value ratios, accrual components) → parquet|QC IDE|`data/raw/fundamentals.parquet` exists|
|3.2|QC Research notebook: pull daily prices + corporate actions 1998‑2024 for momentum & volatility signals → parquet|task 3.1 done|`data/raw/prices.parquet` exists|
|3.3|`lean cloud pull` to sync extras locally|QC project synced|Files checksum > 0|

_(If QC access not used, comment out tasks 3.1–3.3; outline still passes requirements.)_

---

## 4  Master feature table

|#|Task|Start|End / Test|
|---|---|---|---|
|4.1|Script `make_features.py`:|||
|* read `course_panel.parquet`||||
|* compute / rename the official **147 factors**||||
|* engineer **extras**: earnings‐surprise (eps_actual − eps_meanest), accruals, asset growth, 1‑/3‑/6‑/12‑mo momentum (uses QC prices if available, else momentum from monthly course returns), quality (ROE, CFO/Assets), seasonality dummies|raw parquet(s) ready|Output `features_2000‑2024.parquet` columns ≥ 160||
|4.2|Add 1‑month lag to all predictors (`date_lag = date - 1 month`); drop any row where predictor date ≥ return month|script drafted|`tests/test_lag.py` passes (no look‑ahead)|
|4.3|Build **target table** `targets.parquet` with next‑month `stock_exret` directly from course data|course parquet|Row counts align with features|
|4.4|Merge features t with target t+1 → `ml_dataset.parquet`|previous two ready|Non‑null target; duplicates = 0|
|4.5|Merge `mkd_ind1.parquet` to create two benchmark series for later evaluation:|||
|* `rf_monthly` (risk‑free)||||
|* `sp_ret` (S&P 500)|mkt file ready|`benchmarks.parquet` rows ≥ 300||

---

## 5  Dataset split & loaders

|#|Task|Start|End / Test|
|---|---|---|---|
|5.1|`split.py`: **time‑based** splits – train 2000‑2009, val 2010‑2014, test 2015‑2024|ml_dataset ready|Three parquet files sizes logged|
|5.2|`data_module.py` (`LightningDataModule`) – scaling, categorical encoding, batch loader|splits ready|`pytest tests/test_data_module.py` passes|

---

## 6  Model family A – Tree/Boosting ensemble

|#|Task|Start|End / Test|
|---|---|---|---|
|6.1|LightGBM regressor (`models/lgbm.py`): train; output `pred_lgbm.csv` (permno,date,pred)|data module|CSV rows = test rows|
|6.2|CatBoost regressor (`models/catboost.py`): train; output `pred_cat.csv`|env ok|CSV saved|
|6.3|NGBoost (`models/ngboost.py`): train; mean forecast → `pred_ngb.csv`|ngboost imported|CSV saved|
|6.4|NODE (`models/node.py`) torch: train; output `pred_node.csv`|node‑torch installed|CSV saved|

---

## 7  Model family B – Deep tabular nets

|#|Task|Start|End / Test|
|---|---|---|---|
|7.1|TabNet module (`models/tabnet.py`) – local 1‑epoch smoke test|code skeleton|Loss decreases|
|7.2|SAINT (`models/saint.py`) – smoke test|env ok|Loss decreases|
|7.3|TabTransformer (`models/tabtransformer.py`) – smoke test|transformer libs installed|CKPT saved|
|7.4|Launch Lightning AI GPU job to train all three to convergence; download `pred_tabnet.csv`, `pred_saint.csv`, `pred_tt.csv`|cloud job queued|Artifacts present locally|

---

## 8  Model family C – AutoML

|#|Task|Start|End / Test|
|---|---|---|---|
|8.1|AutoGluon Tabular 3‑hour budget; save `pred_auto.csv`|autogluon import ok|CSV saved|
|8.2|Blend/stack top‑N predictions: `blend.py` (rank average) → `pred_blend.csv`|≥ 5 CSVs exist|Blend file saved; correlation matrix plotted|

---

## 9  (Stretch) Model family D – Reinforcement allocation

|#|Task|Start|End / Test|
|---|---|---|---|
|9.1|Build FinRL environment (monthly, 100 actions); reward = portfolio Sharpe|dataset ready|`env.reset()` returns state|
|9.2|Train PPO agent 2000‑2009; export weight table `pred_rl.csv`|training done|CSV has ≤ 100 columns per month|

---

## 10  Prediction export & QC upload

|#|Task|Start|End / Test|
|---|---|---|---|
|10.1|Concatenate all `pred_*.csv` for comparison → `all_preds.parquet`|prediction CSVs|File saved|
|10.2|Upload **chosen** prediction file (blend or single best) to QC ObjectStore: `lean object-store upload`|lean login|Path `qc://predictions.csv` visible in QC IDE|

---

## 11  LEAN algorithm – portfolio rules

|#|Task|Start|End / Test|
|---|---|---|---|
|11.1|`MainAlgorithm.py` skeleton; verify compiles|QC project|Empty back‑test 1 day succeeds|
|11.2|Implement monthly rebalance:|||
|* Load current month slice from `predictions.csv`||||
|* Rank by predicted `exret`||||
|* Select 50‑100 names, weight ≤ 10 %, enforce turnover ≤ 25 %|skeleton compiles|2019 sample back‑test meets constraints||
|11.3|Track alpha, beta, TE using **`mkd_ind1`** series: join `sp_ret` and `RF` into QC custom data; compute metrics|algo runs|Stats JSON contains Alpha, Beta, TE fields|

---

## 12  Full back‑test & constraint asserts

|#|Task|Start|End / Test|
|---|---|---|---|
|12.1|Run 2000‑2024 back‑test (QC cloud)|algo ready|PDF & JSON ready (< 2 h)|
|12.2|`check_constraints.py`: assert positions 50‑100, turnover ≤ 25 %, TE ≤ 5 %, max weight ≤ 10 %|JSON present|Script exit 0|

---

## 13  Performance analysis

|#|Task|Start|End / Test|
|---|---|---|---|
|13.1|`analysis.ipynb`: load equity curve & `sp_ret`; plot cumulative excess return|back‑test done|`plots/cum_curve.png` saved|
|13.2|Rolling 36‑mo alpha, beta, Sharpe (uses `RF`) → PNG|notebook open|`plots/rolling_stats.png` saved|
|13.3|Extract top‑10 avg holdings 2010‑2024 → CSV|back‑test orders|`tables/top10.csv` saved|
|13.4|Compare model variants: bar chart of OOS Information Ratio (`models_comparison.png`)|all_preds.parquet|PNG saved|

---

## 14  Deck preparation

|#|Task|Start|End / Test|
|---|---|---|---|
|14.1|PowerPoint template|blank pptx|`deck/FINE695.pptx` slides ≥ 7|
|14.2|Executive Summary slide|template ready|Slide 1 filled|
|14.3|Methodology slide (dataset provenance, lag audit, model zoo, Lightning–QC pipeline)|slide 1 done|Slide 2 complete|
|14.4|Performance slides (cum curve, metrics table, model bar chart)|plots ready|Slides 3‑4 complete|
|14.5|Holdings + factor discussion slide|top10 CSV ready|Slide 5 complete|
|14.6|Risks & future work slide|slide 5 done|Slide 6 complete|

---

## 15  QA & delivery

|#|Task|Start|End / Test|
|---|---|---|---|
|15.1|Run black, isort, pylint ≥ 8.0|codebase|CI green|
|15.2|Tag `v1.0`; attach back‑test PDF, deck, tasks.md|repo ready|GitHub release page shows assets|

---

### Model coverage checklist

- Gradient boosting: **LightGBM, CatBoost, NGBoost**
    
- Differentiable tree ensemble: **NODE**
    
- Deep attention: **TabNet, SAINT, TabTransformer**
    
- AutoML stack: **AutoGluon**
    
- (Stretch) RL allocation: **FinRL PPO**
    
- Ensemble blend of best predictions
    

Each model outputs its own `pred_<name>.csv` enabling head‑to‑head OOS comparison.

> **✅ Outline now explicitly leverages `group_project_sample_v3.csv`, incorporates `mkd_ind1` for RF & S&P 500, preserves lag integrity, and still tests a rich suite of models.**