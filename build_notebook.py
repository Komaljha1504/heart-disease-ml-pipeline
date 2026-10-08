"""Builds notebooks/heart_disease_pipeline.ipynb from the cells below, then executes it.

    python build_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Heart disease prediction: a tabular ML pipeline

**Question.** Given 13 routine clinical measurements, can we predict whether a patient has angiographic heart disease, and which measurements matter most?

**Data.** UCI *Heart Disease* (Cleveland clinic), 303 patients. Public domain, from the UCI ML Repository (id 45). The original target `num` is 0 (no disease) to 4 (severity); following the usual convention I treat any value above 0 as **disease**.

**Approach, in one line each**
1. Look at the data first (balance, missing values, distributions, correlations).
2. Build one leak-free `Pipeline` (imputation, encoding, scaling happen *inside* each cross-validation fold).
3. Add 4 engineered features and test honestly whether they help.
4. Compare 5 model families under the **same** repeated stratified 5-fold CV.
5. Tune the top 2 with **nested** CV, so the tuned score is not optimistically biased.
6. Explain the best model with permutation importance, and be clear about the limits (303 patients is small).
""")

code("""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 42
RES = Path("../results"); RES.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", palette=["#0b6f8a", "#e4572e"])
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})
""")

md("## 1. Load and inspect")
code("""
raw = pd.read_csv("../data/uci_heart_raw.csv")
df = raw.copy()
df["target"] = (df["num"] > 0).astype(int)      # 1 = any heart disease
df = df.drop(columns="num")
print(df.shape)
display(df.head())
print("\\nMissing values per column:"); print(df.isna().sum()[df.isna().sum() > 0])
print("\\nTarget balance:"); print(df.target.value_counts().rename({0: "no disease", 1: "disease"}))
""")

md("""
Notes from the first look: 303 rows is **small**, so every number below carries real uncertainty and I report mean ± standard deviation across folds. The classes are roughly balanced (about 46% disease), so accuracy is meaningful, but I also track recall because in a screening setting a missed patient costs more than a false alarm. Only `ca` (4 rows) and `thal` (2 rows) have missing values, and they are the kind of missingness an imputer handles safely.
""")

md("## 2. Exploratory analysis")
code("""
num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for a, c in zip(ax.ravel(), num_cols):
    sns.kdeplot(data=df, x=c, hue="target", fill=True, common_norm=False, ax=a, legend=False)
    a.set_title(c)
sns.heatmap(df[num_cols + ["target"]].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax[1, 2], cbar=False)
ax[1, 2].set_title("correlation")
fig.suptitle("Numeric features by outcome (blue = no disease, orange = disease)", y=1.0, fontweight="bold")
plt.tight_layout(); plt.savefig(RES / "eda_numeric.png", bbox_inches="tight"); plt.show()
""")
code("""
fig, ax = plt.subplots(2, 4, figsize=(15, 6.5))
for a, c in zip(ax.ravel(), cat_cols):
    rate = df.groupby(c)["target"].agg(["mean", "size"])
    sns.barplot(x=rate.index.astype(str), y=rate["mean"], ax=a, color="#0b6f8a")
    for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
        a.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
    a.set_title(f"{c}: share with disease"); a.set_ylim(0, 1.1); a.set_xlabel(""); a.set_ylabel("")
plt.tight_layout(); plt.savefig(RES / "eda_categorical.png", bbox_inches="tight"); plt.show()
""")
md("""
**What the EDA suggests.** Maximum heart rate (`thalach`) is lower and ST depression (`oldpeak`) higher in patients with disease. Chest-pain type `cp = 4` (asymptomatic) has by far the highest disease rate, which is a known and slightly counter-intuitive finding in this dataset. Males, exercise-induced angina (`exang`), more blocked vessels (`ca`) and the reversible-defect thallium result (`thal = 7`) all raise the rate. Cholesterol and resting blood pressure separate the classes only weakly. These patterns tell me what to expect from the importance analysis later.
""")

md("""
## 3. Preprocessing and feature engineering

Everything is one scikit-learn `Pipeline`, so imputation statistics, scaling and encoding are learned **only on the training part of each fold**. Fitting them on the full data first would leak information and flatter the scores.

Four engineered features, each with a reason:

| Feature | Definition | Why |
|---|---|---|
| `hr_reserve` | `thalach / (220 - age)` | Peak heart rate relative to the age-predicted maximum; a fitter heart reaches more of it |
| `st_depressed` | `oldpeak >= 1.0` | ST depression of 1 mm or more is the usual clinical threshold for ischaemia |
| `asymptomatic` | `cp == 4` | The strongest category in the EDA, made explicit for linear models |
| `chol_per_age` | `chol / age` | Cholesterol load relative to age |

I will test whether they actually help rather than assume it.
""")
code("""
def add_features(X):
    X = X.copy()
    X["hr_reserve"] = X["thalach"] / (220 - X["age"])
    X["st_depressed"] = (X["oldpeak"] >= 1.0).astype(int)
    X["asymptomatic"] = (X["cp"] == 4).astype(int)
    X["chol_per_age"] = X["chol"] / X["age"]
    return X

ENG = ["hr_reserve", "st_depressed", "asymptomatic", "chol_per_age"]

def make_pre(engineered):
    num = num_cols + (["hr_reserve", "chol_per_age"] if engineered else [])
    cat = cat_cols + (["st_depressed", "asymptomatic"] if engineered else [])
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    steps = ([("fe", FunctionTransformer(add_features))] if engineered else []) + [("pre", pre)]
    return steps

X = df.drop(columns="target"); y = df["target"]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)   # the SAME folds for every model
print(f"{cv.get_n_splits()} train/test splits per model, identical for all models")
""")

md("""
## 4. Model comparison under identical cross-validation

Five model families, all with the same preprocessing and the same 15 folds (5-fold × 3 repeats, stratified). I compare them with and without the engineered features.
""")
code("""
MODELS = {
    "Logistic regression": LogisticRegression(max_iter=2000, random_state=SEED),
    "SVM (RBF)": SVC(probability=True, random_state=SEED),
    "Random forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=2),
    "Gradient boosting": GradientBoostingClassifier(random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9, eval_metric="logloss",
                             random_state=SEED, n_jobs=2),
}
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}

def pipe(model, engineered):
    return Pipeline(make_pre(engineered) + [("clf", clone(model))])

def evaluate(est):
    r = cross_validate(est, X, y, cv=cv, scoring=SCORING, n_jobs=2)
    return {m: (r[f"test_{m}"].mean(), r[f"test_{m}"].std()) for m in SCORING}

rows = []
for eng in (False, True):
    for name, m in MODELS.items():
        s = evaluate(pipe(m, eng))
        rows.append({"model": name, "engineered": eng, **{k: v for k, v in s.items()}})
res = pd.DataFrame(rows)

def fmt(t): return f"{t[0]:.3f} ± {t[1]:.3f}"
table = res.copy()
for m in SCORING: table[m] = table[m].map(fmt)
display(table[table.engineered].drop(columns="engineered").set_index("model"))
""")
code("""
# Do the engineered features help? Compare mean ROC-AUC with vs without, per model.
cmp = res.pivot(index="model", columns="engineered", values="roc_auc").map(lambda t: t[0])
cmp.columns = ["AUC without", "AUC with engineered"]; cmp["change"] = cmp["AUC with engineered"] - cmp["AUC without"]
display(cmp.round(3))
print("Average change in AUC from the 4 engineered features: %+.4f" % cmp["change"].mean())
""")
code("""
fig, ax = plt.subplots(figsize=(8, 4.2))
d = res[res.engineered].sort_values("roc_auc", key=lambda s: s.map(lambda t: t[0]))
ax.barh(d.model, d.roc_auc.map(lambda t: t[0]), xerr=d.roc_auc.map(lambda t: t[1]), color="#0b6f8a", capsize=4)
for i, t in enumerate(d.roc_auc): ax.text(t[0] + t[1] + 0.004, i, f"{t[0]:.3f}", va="center", fontsize=9)
ax.set_xlim(0.75, 1.0); ax.set_xlabel("ROC-AUC (mean ± std over 15 folds)"); ax.set_title("Model comparison, same CV folds")
plt.tight_layout(); plt.savefig(RES / "model_comparison.png", bbox_inches="tight"); plt.show()
""")

md("""
## 5. Tune the top two models (nested cross-validation)

Picking hyper-parameters on the same folds you report on inflates the score. **Nested CV** fixes that: an inner 3-fold grid search picks parameters using only each outer training set, and the outer folds (the same 15 as before) score the result. The tuned number is therefore an honest estimate.
""")
code("""
base = res[res.engineered].assign(auc=lambda d: d.roc_auc.map(lambda t: t[0])).sort_values("auc", ascending=False)
top2 = list(base.model.head(2)); print("Top 2 by untuned ROC-AUC:", top2)

GRIDS = {
    "Logistic regression": {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
    "SVM (RBF)": {"clf__C": [0.3, 1, 3, 10], "clf__gamma": ["scale", 0.01, 0.03, 0.1]},
    "Random forest": {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 3, 5], "clf__max_features": ["sqrt", 0.5]},
    "Gradient boosting": {"clf__n_estimators": [50, 100, 200], "clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.1]},
    "XGBoost": {"clf__n_estimators": [100, 200, 400], "clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.02, 0.05, 0.1]},
}
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
tuned_scores, tuned_est = {}, {}
for name in top2:
    gs = GridSearchCV(pipe(MODELS[name], True), GRIDS[name], cv=inner, scoring="roc_auc", n_jobs=2)
    tuned_scores[name] = evaluate(gs)                 # nested: honest estimate
    tuned_est[name] = clone(gs).fit(X, y)             # refit on all data only to read the chosen parameters
    print(f"{name}: best params {tuned_est[name].best_params_}")

out = pd.DataFrame({
    "untuned AUC": {n: fmt(base.set_index('model').loc[n, 'roc_auc']) for n in top2},
    "tuned AUC (nested CV)": {n: fmt(tuned_scores[n]['roc_auc']) for n in top2},
    "tuned accuracy": {n: fmt(tuned_scores[n]['accuracy']) for n in top2},
    "tuned recall": {n: fmt(tuned_scores[n]['recall']) for n in top2},
    "tuned F1": {n: fmt(tuned_scores[n]['f1']) for n in top2},
})
display(out)
""")

md("## 6. Final model: out-of-fold performance and what drives it")
code("""
best = max(top2, key=lambda n: tuned_scores[n]["roc_auc"][0]); print("Final model:", best)
final = tuned_est[best].best_estimator_
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = cross_val_predict(clone(final), X, y, cv=skf, method="predict_proba")[:, 1]
pred = (oof >= 0.5).astype(int)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
RocCurveDisplay.from_predictions(y, oof, ax=ax[0], name=best)
ax[0].plot([0, 1], [0, 1], "--", color="grey"); ax[0].set_title(f"ROC (out-of-fold), AUC = {roc_auc_score(y, oof):.3f}")
ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax[1], cmap="Blues", display_labels=["no disease", "disease"], colorbar=False)
ax[1].set_title("Confusion matrix (out-of-fold, threshold 0.5)")
plt.tight_layout(); plt.savefig(RES / "final_model_roc_confusion.png", bbox_inches="tight"); plt.show()
""")
code("""
# Permutation importance, computed on held-out folds only, on the ORIGINAL columns (so one-hot pieces are not split apart).
imps = []
for tr, te in skf.split(X, y):
    m = clone(final).fit(X.iloc[tr], y.iloc[tr])
    r = permutation_importance(m, X.iloc[te], y.iloc[te], scoring="roc_auc", n_repeats=20, random_state=SEED)
    imps.append(pd.Series(r.importances_mean, index=X.columns))
imp = pd.concat(imps, axis=1)
imp_df = pd.DataFrame({"mean": imp.mean(axis=1), "std": imp.std(axis=1)}).sort_values("mean")

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp_df.index, imp_df["mean"], xerr=imp_df["std"], color="#0b6f8a", capsize=3)
ax.axvline(0, color="grey", lw=1); ax.set_xlabel("drop in ROC-AUC when the feature is shuffled (held-out folds)")
ax.set_title("What the final model relies on")
plt.tight_layout(); plt.savefig(RES / "feature_importance.png", bbox_inches="tight"); plt.show()
display(imp_df.sort_values("mean", ascending=False).round(4).head(8))
""")

code("""
summary = {
    "dataset": "UCI Heart Disease (Cleveland), 303 patients, target = any disease",
    "cv": "5-fold stratified x 3 repeats (15 splits), identical for every model",
    "comparison_with_engineered_features": {r.model: {k: [round(v[0], 4), round(v[1], 4)] for k, v in r._asdict().items() if k in SCORING}
                                              for r in res[res.engineered].itertuples()},
    "comparison_without_engineered_features": {r.model: {k: [round(v[0], 4), round(v[1], 4)] for k, v in r._asdict().items() if k in SCORING}
                                                 for r in res[~res.engineered].itertuples()},
    "engineered_feature_avg_auc_change": round(float(cmp["change"].mean()), 4),
    "tuned_nested_cv": {n: {k: [round(v[0], 4), round(v[1], 4)] for k, v in s.items()} for n, s in tuned_scores.items()},
    "best_params": {n: {k: (v if isinstance(v, (int, float, str)) or v is None else str(v)) for k, v in tuned_est[n].best_params_.items()} for n in top2},
    "final_model": best,
    "top_features": imp_df.sort_values("mean", ascending=False).head(6)["mean"].round(4).to_dict(),
}
Path("../results/metrics.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ("final_model", "engineered_feature_avg_auc_change", "tuned_nested_cv", "top_features")}, indent=1))
""")

CONCL = ROOT / "conclusion.md"
if CONCL.exists():
    md(CONCL.read_text(encoding="utf-8"))

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "notebooks" / "heart_disease_pipeline.ipynb"
ExecutePreprocessor(timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(out.parent)}}).preprocess(nb, {"metadata": {"path": str(out.parent)}})
nbf.write(nb, out)
print("wrote", out)
