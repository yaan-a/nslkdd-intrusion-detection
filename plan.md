# Project plan

Repository: https://github.com/yaan-a/nslkdd-intrusion-detection
Written: 18 October 2026. Effort figures are rough estimates, not measurements, and assume one person working alone.

## 0. Where things stand

Checked against a fresh clone of `main` (latest commit: merge of PR #6, `feat/add-attack-type-analysis`):

| Item | State |
|---|---|
| Replication scripts, official-protocol stacking model, Streamlit app | On `main` |
| Issue #1 work (`analyze_attack_types.py`, `results/per_attack_recall.csv`, `results/per_family_recall.csv`) | **Merged** via PR #6 |
| README with per-attack-type results | On `main`; still has the `Collaborator Name` placeholder |
| Data (`KDDTrain+.csv`, `KDDTest+.csv`, merged file) and both model files (26 MB and 33 MB) | On `main` |
| CI (issue #2): `utils.py`, `tests/test_utils.py`, `python-test.yml` | On `main`; the workflow still lists the deleted `add-ci-tests` branch |
| Tags | **None yet** (`v1.0` and `v1.1` are missing) |
| Issues #3 and #4 | Open, not started |

Not verified from outside the repo: whether issues #1 and #2 are actually closed on GitHub, whether anyone reviewed the PRs, and what you changed locally in `requirements.txt` and `setup.py` (they are not on `main`).

Facts about the model that shape the plan (checked on the model bundle and the test data):

- The bundle is a dictionary, **not** a scikit-learn `Pipeline`. Preprocessing is label encoders, then dropping `num_outbound_cmds`, then a `RobustScaler` over 40 columns, then keeping 34 columns.
- The 34 model inputs are 2 categorical (`protocol_type`: 3 values, `service`: 70 values) and 32 numeric.
- The 6 columns removed by VIF filtering (`num_root`, `srv_serror_rate`, `srv_rerror_rate`, `dst_host_srv_serror_rate`, `flag`, `dst_host_same_srv_rate`) have no effect on predictions. Filling them with 0 gave the same probabilities on all 22,544 test records (maximum difference 0.0). An API only needs to ask for the 34 inputs.
- One single-record prediction took about 26 ms in my sandbox. Your machine will differ.

## 1. What the Lab 3 manual asks, in plain terms

It asks you to wrap a trained model in a small web application and publish the code with automated checks. Concretely:

1. **A local API** (Flask) with two endpoints: `GET /health` and `POST /predict`. You send a JSON record, it loads the model, validates the input and returns a prediction.
2. **A web form** (Angular) with one field per model input, which calls the API and shows the result.
3. **Tests**: a direct inference test, API tests with `curl` and Python, and four automated `pytest` tests (health, valid input, missing feature, invalid JSON).
4. **One Git repository** on GitHub, with a clean history and no secrets, virtual environments or `node_modules`.
5. **GitHub Actions** that run the backend tests and build the Angular app on every push.
6. **GitHub Pages** hosting the Angular app as a static site. Pages cannot run Python, so the public page shows the form with the prediction button disabled and a notice. Live predictions work only on your own machine with both servers running.
7. **Evidence and writing**: README, screenshots, troubleshooting notes and a technical report.

It does not ask for attack-type identification, causes, solutions or prevention. Those are a separate requirement (section 5).

**The manual's example code does not work as-is for your model.** The example `app.py` calls `MODEL.predict(frame)` on a fitted `Pipeline` that takes raw features. Yours needs the adapter described in section 4. The manual allows this (Stage 4.4) but leaves the details to you.

## 2. Decisions to make first

| # | Decision | Default I would use |
|---|---|---|
| 1 | New repository for Lab 3, or add `backend/` and `frontend/` to the existing repo? The manual says "one root repository". | Same repo. The model, data and tests are already here. Ask your instructor if a separate repo is required. |
| 2 | Must the app show attack type, cause, solution and prevention? | Treat it as required for the final version, after the basic app works. |
| 3 | Is a fixed, reviewed lookup table acceptable for "solution" and "prevention", or must the system generate text per detection? | Lookup table. It is reviewable and safe. |
| 4 | How does the review requirement get satisfied when you work alone? GitHub does not let you approve your own PR. | Ask the instructor. Do not self-approve and report it as a review. |
| 5 | Deadlines for Lab 2, Lab 3 and the research deliverable. | Unknown. Section 8 shows what to cut if time is short. |
| 6 | One copy of the model file or two? The manual expects `backend/models/model_bundle.joblib`. | One copy. Point the backend at `model/nslkdd_model.pkl` and document the deviation. Copy the file only if the instructor insists (it adds 26 MB to the repo). |

## 3. Priorities at a glance

| Priority | Work | Rough effort | Depends on | Done when |
|---|---|---|---|---|
| **P0** | Close out Lab 2 | 1 to 2 h, plus review wait | nothing | Issues closed, `v1.1` tagged, table and screenshots done |
| **P1** | Lab 3: working full-stack app with the existing binary model | 14 to 22 h | P0 | All manual checkpoints pass, CI and Pages are green |
| **P2** | Attack type, cause, solution and prevention | 17 to 31 h | P1 working | App returns a family, confidence, indicators and reviewed guidance |
| **P3** | Research: unseen-attack and conformal study | weeks | P0, partly P2 | Pilot result, then Decision Gate A |
| **P4** | Write-up and submission | 10 to 25 h | P1 to P3 | README, report, evidence submitted |

Work in this order. Do not start P3 while P1 is unfinished, because P1 has a hard deadline structure and P3 is open-ended.

## 4. P0: close out Lab 2

1. Confirm on GitHub that issues #1 and #2 closed automatically. If not, close them and add a comment linking the merged PR.
2. Decide how the review step is satisfied (decision 4 above).
3. Fix the README placeholder (`Collaborator Name`) in a small PR, or remove the collaborator line if you worked alone.
4. Remove `add-ci-tests` from the push trigger in `python-test.yml` (optional cleanup).
5. Fill the K4 baseline versus revised table. The overall metrics are identical, since #1 and #2 changed no model or preprocessing. Say so, and add the per-attack-type recall table as the new result.
6. Tag the milestone:
   ```powershell
   git switch main
   git pull origin main
   git tag -a v1.0 -m "Baseline stacking model (official protocol)" <commit before PR #6>
   git tag -a v1.1 -m "Add attack-type analysis and CI"
   git push origin v1.0 v1.1
   git log --oneline
   ```
   Find the baseline commit with `git log --oneline`. If you cannot tell which commit was the baseline, tag only `v1.1`.
7. Screenshots: Issues list, merged PRs, Actions runs (the failing and passing ones), the Tags page, `git log --oneline`.

Keep #3 and #4 open. The manual does not require them.

## 5. P1: Lab 3, the full-stack application

Goal: a working app using the existing **binary** model (normal vs. attack), with every manual checkpoint passing. Do not add multiclass or guidance yet. A correct, tested, deployed basic app comes first.

### 5.1 Setup (manual Stages 1 to 3), about 1 to 2 h

- Install and record versions: VS Code, Python, Git, Node LTS, and the Angular CLI.
- Use the repo you already have as the root. **Do not run `git init`** and do not create a new `.venv` if you already have one that runs the model.
- Check `python --version` in your active venv. An earlier error trace of yours showed Python 3.14, while the README and CI say 3.12. CI must use the same major.minor version as your local environment, or the pickle may not load.
- The manual's version numbers (Node 24, Angular 22, `ng new` output file names, build output path `dist/frontend/browser`) are examples that I could not verify. Use what your installed tools generate, and update `node-version` and the Pages upload path to match.
- Add `docs/environment.txt` with the tool versions.

### 5.2 Model access (Stages 4 and 5), about 2 to 3 h

Do **not** wrap the bundle in a new pickled class. Write a plain module, `backend/inference.py`, that loads the existing bundle once and exposes `predict_record(dict)`.

`predict_record` steps, identical to `export_model.py` and `analyze_attack_types.py`:
1. Check the input has exactly the 34 `kept_features` and nothing else.
2. Encode `protocol_type` and `service` with the bundle's encoders. Reject unknown categories with a clear error.
3. Fill the 6 VIF-dropped columns with 0 (`flag` after encoding).
4. Scale all 40 columns with the bundle's scaler, keep the 34 `kept_features` in order, convert to float32.
5. `predict_proba`; label is attack if the probability is at least 0.5.

Tests for this stage:
- A single known record gives the same probability as the offline pipeline.
- **Over all 22,544 `KDDTest+` records the adapter reproduces TN 9,432, FP 279, FN 4,601, TP 8,232.** This is the strongest check and is the equivalent of the manual's "compare with the training notebook".
- Sample payloads: one normal and one attack record from `KDDTest+`, saved to `backend/test_payload_normal.json` and `backend/test_payload_attack.json`.

### 5.3 Schema (Stage 5 and 9), about 1 h

Generate the schema from data with a script, `backend/make_schema.py`, instead of typing 34 entries. It should write `backend/schema.json` with: field name, type (number or category), allowed categories, group, and min and max from `KDDTrain+` and `KDDTest+`. Verify the ranges against the data before relying on them.

Validation rules:
- **Reject**: wrong or missing keys, non-numbers, negative values for counts, unknown categories, rate fields outside 0 to 1, flag fields outside {0, 1}.
- **Warn, do not reject**: values outside the training 1st to 99th percentile. The bounds stored in the bundle come from percentiles and would reject real attacks that have large byte counts.

### 5.4 Flask API (Stages 6 and 7), about 2 to 3 h

- `GET /health`: status, number of features, model version.
- `POST /predict`: returns `{ "prediction": "attack" | "normal", "label": 1 | 0, "probabilities": [p_normal, p_attack], "threshold": 0.5, "model_version": "...", "disclaimer": "Research prototype only." }`. Keep it compatible with the manual's `PredictionResponse`.
- Errors: 400 for schema problems with a list of what is wrong, 500 for unexpected failures with no stack trace in the body.
- Tests: the manual's four (health, valid, missing feature, invalid JSON), plus invalid category, out-of-range rate, and the full `KDDTest+` reproduction. Mark the full-dataset test as slow so you can skip it locally.
- Manual checks: `curl.exe` against `/health` and `/predict`, and `backend/test_api.py` with `trust_env=False`.

### 5.5 Angular form (Stages 8 to 10), about 4 to 6 h

- Generate `frontend/` inside the repo, with `--skip-git`.
- Build the form from `schema.json` (convert it to `feature-schema.ts`, or load it as an asset) so the front end and back end cannot drift apart.
- A 34-field form is unusable by hand. Add three things:
  - **Example buttons** that fill the form with a real normal record and a real attack record from `KDDTest+`.
  - **Grouped sections** (basic connection, content, time-window traffic, host-based traffic) with a short help text per field.
  - **A searchable or filterable select** for the 70 `service` values.
- Result panel: label, probability, threshold, disclaimer. Do not call a research score an operational verdict.
- Dev proxy as in the manual: `/api/*` to `http://127.0.0.1:5000`. Test with `curl.exe http://localhost:4200/api/health`.
- Check validation by blanking a field, and check the failure path by stopping Flask.

### 5.6 Repository, CI and Pages (Stages 11 to 14), about 3 to 5 h

- Stages 11 and 12 are already done by your existing repo. Review `.gitignore` instead: add `frontend/node_modules/`, `frontend/dist/`, `frontend/.angular/`, `.pytest_cache/`. Keep the model and data files tracked, since they are small, public benchmark artifacts.
- **Create `backend/requirements.txt` by hand**, not with `pip freeze` from your research environment. Include `flask`, `pandas`, `numpy`, `joblib`, `scikit-learn==<the version your model needs>`, `pytest`, `requests`. A raw `pip freeze` will pull in Windows-only and heavy packages (Streamlit, statsmodels, matplotlib) that can break a Linux CI runner.
- **Pickle compatibility in CI**: your model was created with scikit-learn 1.9.1. Pin that version in CI, or the model-dependent tests will fail or warn.
- CI: you already have `python-test.yml`. Either add the manual's `ci.yml` and accept two workflows, or merge them into one so each push does not run two overlapping jobs. Because the model is tracked, enable the model-dependent tests in CI (the manual's "optional" line), and say so in the README.
- Pages workflow: set `--base-href /nslkdd-intrusion-detection/`, verify the build output folder locally first, set Pages source to GitHub Actions. This needs a public repository, and yours is public.
- The public site shows the form with the button disabled and a notice. Optional improvement, about 1 h: a few precomputed example predictions shown as static text, so the public page is not just an empty form. Label them clearly as precomputed.

### 5.7 Evidence and write-up (Stage 15), about 2 to 4 h

- README sections the manual lists: model provenance, schema, versions, startup commands for the two terminals, test commands, repository and Pages URLs, deployment limits and a research disclaimer.
- Screenshots: `/health`, a `curl` predict, pytest passing, the form with a result, Actions runs, the Pages site. Use no sensitive data (NSL-KDD is a public benchmark, so this is fine).
- Troubleshooting log, technical report, and the checklist at the end of the manual.

### 5.8 P1 risks

- **Schema drift** between API and form: avoided by generating both from `schema.json`.
- **Version drift** between local and CI: avoided by pinning and by running the full reproduction test in CI.
- **Unverified tool versions** in the manual: check each against what your machine produces.
- **Heavy repo history**: each re-committed model adds 26 to 33 MB. Commit models rarely.

## 6. P2: attack type, cause, solution and prevention

Start only after P1 works end to end.

What is realistic:

| Ask | Delivered as | Limit |
|---|---|---|
| Attack type | A family-level classifier: normal, DoS, Probe, R2L, U2R | Cannot name types it has not seen. R2L and U2R will likely be weak (the binary model catches only 4.6% and 13.4%). |
| Cause | (a) Indicators: which input values are most unusual for this record. (b) The typical mechanism of that attack family, from the knowledge base. | The real cause of a specific incident is not in the data. State this plainly. |
| Solution | Reviewed response guidance per family | A lookup, not learned. |
| Prevention | Reviewed hardening guidance per family, plus descriptive facts from the data (which services, protocols and flags that family uses) | Descriptive correlations from a simulated, late-1990s dataset. |

### 6.1 Steps

1. **Multiclass model, about 4 to 8 h.** Five classes. Train on `KDDTrain+` only. Handle imbalance with class weights or resampling, and tune nothing on `KDDTest+`. Report per-class recall, macro-F1 and the confusion matrix, on `KDDTest+` and on a random split, and say honestly if R2L and U2R stay weak. Save it as a separate bundle. Do not overwrite the binary model.
2. **Indicators, about 3 to 5 h.** Use a transparent method: for each input, how far the record's scaled value sits from the normal-traffic median, and the top few features. This avoids a heavy SHAP dependency for a stacking model with KNN. If you want SHAP, apply it to a single Random Forest family model and add the dependency to CI deliberately.
3. **Knowledge base, about 4 to 8 h.** One file, `backend/knowledge/attack_guidance.json`, with an entry per family and optionally a few well-known types (`neptune`, `smurf`, `guess_passwd`, `buffer_overflow`). Fields: mechanism, typical indicators in this dataset, immediate response, prevention, sources, reviewer and review date. Write it from published guidance (NIST incident-handling publications, MITRE ATT&CK and D3FEND, CISA advisories). **Read each source and cite it exactly.** Do not generate this text with a language model and then publish it unreviewed. Ask someone with security knowledge to review it, and flag items that are dated for 1990s attacks, with a modern equivalent where one exists.
4. **API and UI, about 4 to 6 h.** Extend the response with `family`, `family_probabilities`, `indicators`, `guidance`. Show a result panel with a clear "indicators, not proven cause" label.
5. **Uncertainty rule, about 1 to 2 h.** If the predicted family probability is below a threshold chosen on training-derived data, return "unrecognized, escalate to an analyst" and **no recommendation**. This is the "not sure" idea from the research plan, so P2 and P3 share this piece.
6. **Tests and evaluation, about 2 to 4 h.** Unit tests for the knowledge base (every family has every field, every entry has a source). Reproduce the multiclass metrics in a test.

### 6.2 P2 risks

- Weak R2L and U2R classification means many attacks get no useful label. Report it, and show the escalation path.
- The knowledge base is the part most likely to contain errors and needs a human reviewer.
- Do not present the system as able to block or remediate attacks automatically.

## 7. P3: research (unseen-attack calibration)

Full detail is in `roadmap.md`. Condensed order and gates:

1. **Hand literature check** (about 3 to 6 h): Google Scholar, Springer, IEEE and ACM, for conformal prediction, coverage, unseen attacks and open-set detection on NSL-KDD. Include calibration, which has not been searched yet. Write a one-page related-work table. If someone has already done the exact study, change the angle.
2. **Pilot, about 2 to 4 h**: Random Forest, calibrate on a slice of `KDDTrain+`, measure coverage on a random held-out slice, on `KDDTest+` seen attack types, and on `KDDTest+` unseen types. If coverage does not degrade, change direction now.
3. **Issue #3, about 4 to 6 h**: all six models under the official protocol with identical features.
4. **Decision Gate A**: if most of the 99% to 78% drop comes from features or preprocessing, reframe toward a protocol audit. Record the evidence in a comment on #3.
5. **Leave-attack-type-out, calibration, split conformal, abstention baselines, shift-aware calibration** (roadmap steps 2.1 to 2.5). Several weeks of work.
6. **Seeds and confidence intervals, ablations** (roadmap steps 3.1 and 3.2).

Observations so far that the study must respect (from the per-attack results): the failures divide into two groups. DoS and Probe detection drops sharply on unseen types (99.8% to 22.1%, and 100% to 66.7%), which is the effect the research question targets. R2L and U2R are poor whether seen or not, which is a class-imbalance problem the question does not cover. Keep these separate.

## 8. If time is short

Cut in this order, last cut first:

1. **Minimum acceptable**: P0 plus P1 (basic app, CI, Pages, README). This satisfies the two manuals.
2. **Next**: P2 with family-level classification and a four-entry knowledge base (one per family), with a human review recorded.
3. **Next**: the P3 hand literature check and the pilot, with a clear "future work" statement.
4. **Drop first**: shift-aware calibration, open-set baselines, second dataset, SVM replication, SHAP.

## 9. Working rules for everything above

- One branch and one PR per piece of work, with `Closes #N`, a description of only what was actually run, and passing CI.
- Create GitHub issues before starting each priority. Suggested labels: `enhancement`, `ci`, `analysis`, `future-work`, plus `backend`, `frontend`, `docs`.
- Never tune anything using `KDDTest+` labels.
- Record what you ran, with versions and seeds, in the PR or README. Report results as found.
- Keep the README current. Update it in the same PR as the change it describes.
- Commit model files rarely.

## 10. Checklists

**P0 done**: issues #1 and #2 closed; review handled; README placeholder fixed; K4 table; `v1.1` tag; screenshots.

**P1 done**: adapter reproduces TN/FP/FN/TP on all of `KDDTest+`; `/health` and `/predict` verified with curl and Python; the four pytest tests plus the extras pass; Angular form renders all 34 fields with example buttons; proxy works end to end; CI green with the model-dependent tests running; Pages site live with the disabled-button notice; README, screenshots and report finished.

**P2 done**: multiclass metrics reported honestly; indicators shown with a clear label; knowledge base reviewed with sources; low-confidence path returns "escalate" with no advice; tests pass.

**P3 first milestone**: related-work table written; pilot result recorded; Decision Gate A decided with evidence.

## 11. Open questions for your instructor

1. Does Lab 3 require a new repository, or can it live in the existing one?
2. Must the app show attack type, cause, solution and prevention, and is a reviewed lookup table acceptable for the last two?
3. How should the review step be handled when working alone?
4. What are the deadlines for each deliverable?
5. Which Python, Node and Angular versions do you expect? The manual's examples (Python 3.12, Node 24, Angular 22) could not be verified from here.
