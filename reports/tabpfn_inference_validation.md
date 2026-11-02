# TabPFN inference validation

Branch: `test/tabpfn-inference`. Date: October 8, 2026.
Environment: Windows, Python 3.14.7, tabpfn 9.1.0, CPU (CUDA unavailable).

## Current implementation and results

TabPFN is now registered as `tabpfn`, appended after existing models. Its pipeline
fits PCA on training features only, targeting 200 components and reducing this
count to the training row count or input width when necessary. Prediction reuses
the fitted PCA without refitting it. The constructor exposes `n_components` and
`random_state` (defaults 200 and 42). Embeddings remain in the shared featurizer.

Executed the complete suite with browser authentication prompts disabled:

```powershell
$env:TABPFN_NO_BROWSER = '1'
.\.venv\Scripts\python.exe -m pytest tests -vv --tb=short
```

**15 passed, 2 failed**. Passing checks cover registry lookup, evaluator selection,
PCA-200 configuration, actual PCA compression of a 220-by-240 matrix to 200
components, narrow inputs, 2048-dimensional inputs with few rows, training-only
centering, unchanged PCA during prediction, and component-count restoration on
refit. All seven existing tree-model tests pass. CheMeleon independently produced
distinct finite 2048-dimensional embeddings.

The isolated PCA behavior checks substitute sklearn DummyRegressor for the final
estimator, so they do not prove TabPFN prediction behavior. The two real inference
tests use the actual TabPFN regressor on synthetic features and CheMeleon
embeddings. Both fail during fit with `TabPFNLicenseError`, before predictions,
because pretrained weight authorization is unavailable. No TabPFN predictions or
CYP accuracy metrics have been produced. Sample labels are illustrative only.

## Remaining inference blocker

Complete the PriorLabs account/license setup and configure `TABPFN_TOKEN` locally,
then rerun the two real inference tests. Do not commit credentials. Initial
browser login attempts also hit Windows `OSError: [WinError 10038]` inside the
installed library's stdin polling; disabling browser prompts exposes the clear
license error. No credentials were supplied or license accepted by the agent.

The earlier validation found missing PCA and registry integration; both code
gaps are now fixed. A meaningful CYP accuracy evaluation still requires measured
labels and the pinned dataset split.
