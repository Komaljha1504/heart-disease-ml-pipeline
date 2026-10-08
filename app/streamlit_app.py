"""Streamlit demo for the heart-disease pipeline: try a patient profile, see the risk estimate and why, and browse the results.

    streamlit run app/streamlit_app.py

EDUCATIONAL DEMO ONLY. Not medical advice.
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
NUM = ["age", "trestbps", "chol", "thalach", "oldpeak", "hr_reserve", "chol_per_age"]
CAT = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "st_depressed", "asymptomatic"]
NICE = {"age": "Age", "trestbps": "Resting blood pressure", "chol": "Cholesterol", "thalach": "Max heart rate", "oldpeak": "ST depression",
        "hr_reserve": "Heart-rate reserve", "chol_per_age": "Cholesterol per year of age", "sex": "Sex", "cp": "Chest-pain type",
        "fbs": "Fasting blood sugar > 120", "restecg": "Resting ECG", "exang": "Exercise-induced angina", "slope": "ST slope",
        "ca": "Major vessels (fluoroscopy)", "thal": "Thallium test", "st_depressed": "ST depression >= 1", "asymptomatic": "Asymptomatic chest pain"}

st.set_page_config(page_title="Heart Disease Pipeline", page_icon="🫀", layout="wide")
st.markdown(
    """
<style>
.block-container{max-width:1100px;padding-top:1.4rem}
.hero{background:linear-gradient(135deg,#0a2540 0%,#0b6f8a 55%,#0f9b8e 100%);color:#fff;padding:26px 30px;border-radius:20px}
.hero h1{margin:0 0 6px;font-size:30px;color:#fff}
.hero p{margin:0;color:rgba(255,255,255,.88);font-size:15px;max-width:800px;line-height:1.55}
.pills{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}
.pill{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.28);padding:4px 12px;border-radius:999px;font-size:12.5px;font-weight:600}
.notice{background:#fff7e6;border:1px solid #f5d9a3;color:#7a5200;padding:12px 16px;border-radius:12px;font-size:13.5px;margin:14px 0}
.kpi{background:#f1f8f8;border-radius:12px;padding:12px 14px}
.kpi b{display:block;font-size:24px;color:#0b6f8a}
.kpi span{font-size:12px;color:#4b6b73}
.risk{border-radius:16px;padding:18px 22px;color:#fff}
.risk b{font-size:42px;display:block;line-height:1.1}
</style>
<div class="hero"><h1>Heart Disease Prediction Pipeline</h1>
<p>A leak-free scikit-learn pipeline on the UCI Heart Disease data: <b>5 models</b> compared under identical repeated cross-validation, the top two tuned with
<b>nested CV</b>, and permutation importance on held-out folds. Try a patient profile below, or browse the results.</p>
<div class="pills"><span class="pill">scikit-learn</span><span class="pill">303 patients</span><span class="pill">15 CV splits</span><span class="pill">Nested-CV tuning</span><span class="pill">Permutation importance</span></div></div>
<div class="notice"><b>Educational demo, not medical advice.</b> It was trained on 303 patients from one clinic and uses results of specialist tests
(vessel imaging, thallium scan), so it cannot screen anyone from a routine check-up.</div>
""",
    unsafe_allow_html=True,
)


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["hr_reserve"] = X["thalach"] / (220 - X["age"])
    X["st_depressed"] = (X["oldpeak"] >= 1.0).astype(int)
    X["asymptomatic"] = (X["cp"] == 4).astype(int)
    X["chol_per_age"] = X["chol"] / X["age"]
    return X


@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "data" / "uci_heart_raw.csv")
    df["target"] = (df["num"] > 0).astype(int)
    return df.drop(columns="num")


@st.cache_resource
def fit_model():
    df = load_data()
    X, y = add_features(df.drop(columns="target")), df["target"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.3, max_iter=2000, random_state=42))]).fit(X, y)
    return model


def explain(model, row: pd.DataFrame):
    """Per-feature contribution to the log-odds (logistic regression is linear, so this is exact)."""
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    Z = pre.transform(row)[0]
    contrib = Z * clf.coef_[0]
    names = pre.get_feature_names_out()
    out = {}
    for n, c in zip(names, contrib):
        col = n.split("__", 1)[1]
        key = next((f for f in sorted(NICE, key=len, reverse=True) if col == f or col.startswith(f + "_")), col)
        out[key] = out.get(key, 0.0) + float(c)
    return pd.Series(out)


df = load_data()
model = fit_model()
metrics = json.loads((ROOT / "results" / "metrics.json").read_text())

tab_try, tab_res = st.tabs(["Try a patient profile", "Results & analysis"])

with tab_try:
    presets = {"Custom": None,
               "Example: lower-risk profile": df[df.target == 0].sort_values("thalach").iloc[-5],
               "Example: higher-risk profile": df[df.target == 1].sort_values("oldpeak").iloc[-5]}
    choice = st.selectbox("Start from", list(presets), help="Pick a real row from the dataset, then edit the values.")
    base = presets[choice] if presets[choice] is not None else df.median(numeric_only=True)
    g = lambda k, d: float(base[k]) if k in base and not pd.isna(base[k]) else d

    c1, c2, c3 = st.columns(3)
    age = c1.slider("Age", 29, 77, int(g("age", 54)))
    sex = c1.selectbox("Sex", [1, 0], format_func=lambda v: "Male" if v == 1 else "Female", index=0 if g("sex", 1) == 1 else 1)
    cp = c1.selectbox("Chest-pain type", [1, 2, 3, 4], index=int(g("cp", 3)) - 1,
                      format_func=lambda v: {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}[v])
    trestbps = c2.slider("Resting blood pressure (mm Hg)", 90, 200, int(g("trestbps", 130)))
    chol = c2.slider("Cholesterol (mg/dl)", 120, 570, int(g("chol", 240)))
    thalach = c2.slider("Max heart rate achieved", 70, 205, int(g("thalach", 150)))
    oldpeak = c2.slider("ST depression (exercise vs rest)", 0.0, 6.5, float(g("oldpeak", 1.0)), 0.1)
    fbs = c3.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], index=int(g("fbs", 0)), format_func=lambda v: "Yes" if v else "No")
    restecg = c3.selectbox("Resting ECG", [0, 1, 2], index=int(g("restecg", 0)),
                           format_func=lambda v: {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}[v])
    exang = c3.selectbox("Exercise-induced angina", [0, 1], index=int(g("exang", 0)), format_func=lambda v: "Yes" if v else "No")
    slope = c3.selectbox("ST slope", [1, 2, 3], index=int(g("slope", 2)) - 1, format_func=lambda v: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[v])
    ca = c3.selectbox("Major vessels coloured (0 to 3)", [0, 1, 2, 3], index=int(g("ca", 0)))
    thal = c3.selectbox("Thallium test", [3, 6, 7], index=[3, 6, 7].index(int(g("thal", 3))),
                        format_func=lambda v: {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}[v])

    row = add_features(pd.DataFrame([dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg, thalach=thalach,
                                           exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)]))
    p = float(model.predict_proba(row)[0, 1])
    colour = "linear-gradient(135deg,#0f9b8e,#0b6f8a)" if p < 0.35 else ("linear-gradient(135deg,#d99a1f,#c26a12)" if p < 0.65 else "linear-gradient(135deg,#d6455d,#a0203a)")
    label = "Lower estimated likelihood" if p < 0.35 else ("Uncertain, mid-range" if p < 0.65 else "Higher estimated likelihood")
    left, right = st.columns([1, 1.4], gap="large")
    left.markdown(f'<div class="risk" style="background:{colour}"><span>Model estimate of heart disease</span><b>{p*100:.0f}%</b><span>{label}</span></div>', unsafe_allow_html=True)
    left.caption("Probability from the logistic regression fitted on all 303 patients. Cross-validated, the model reaches ROC-AUC 0.92, so these estimates are informative but far from certain, and they are not a diagnosis.")
    c = explain(model, row).sort_values()
    top = pd.concat([c.head(4), c.tail(5)]).drop_duplicates()
    chart_df = pd.DataFrame({"factor": [NICE.get(k, k) for k in top.index], "effect": top.values})
    chart_df["direction"] = np.where(chart_df.effect > 0, "pushes risk up", "pushes risk down")
    ch = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("effect:Q", title="effect on log-odds (this patient)"),
        y=alt.Y("factor:N", sort="-x", title=None, axis=alt.Axis(labelLimit=280, labelOverlap=False)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["pushes risk up", "pushes risk down"], range=["#d6455d", "#0f9b8e"]), legend=alt.Legend(orient="bottom", title=None)),
    ).properties(height=340, title="Why: the factors that moved this estimate most")
    right.altair_chart(ch, use_container_width=True)

with tab_res:
    comp = metrics["comparison_with_engineered_features"]
    rows = [{"model": m, "AUC": v["roc_auc"][0], "std": v["roc_auc"][1], "Accuracy": v["accuracy"][0], "Recall": v["recall"][0], "F1": v["f1"][0], "Precision": v["precision"][0]} for m, v in comp.items()]
    r = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    k = st.columns(4)
    best = r.iloc[0]
    for col, (v, t) in zip(k, [(f"{best.AUC:.2f}", f"ROC-AUC, {best.model} (± {best['std']:.2f})"), (f"{best.Accuracy*100:.1f}%", "accuracy"),
                               ("5", "models, identical 15 CV splits"), ("0", "gain from tuning or engineered features")]):
        col.markdown(f'<div class="kpi"><b>{v}</b><span>{t}</span></div>', unsafe_allow_html=True)
    st.markdown("")
    base_ch = alt.Chart(r)
    bars = base_ch.mark_bar(cornerRadiusEnd=5, color="#0b6f8a").encode(x=alt.X("AUC:Q", scale=alt.Scale(domain=[0.75, 1.0]), title="ROC-AUC (mean ± std over 15 splits)"), y=alt.Y("model:N", sort="-x", title=None))
    err = base_ch.mark_errorbar(color="#0a2540").encode(y=alt.Y("model:N", sort="-x"), x=alt.X("lo:Q"), x2="hi:Q").transform_calculate(lo="datum.AUC - datum.std", hi="datum.AUC + datum.std")
    st.altair_chart((bars + err).properties(height=240, title="Model comparison under the same cross-validation"), use_container_width=True)
    st.dataframe(r.set_index("model").round(3), use_container_width=True)

    a, b = st.columns(2)
    a.image(str(ROOT / "results" / "feature_importance.png"), caption="Permutation importance on held-out folds")
    b.image(str(ROOT / "results" / "final_model_roc_confusion.png"), caption="Out-of-fold ROC and confusion matrix")
    st.markdown(
        "**Honest takeaways.** The simplest model won, and the gaps between the top models (about 0.01 AUC) are smaller than the fold-to-fold spread (about 0.04). "
        "Tuning and my four engineered features gave no gain, and I report that instead of hiding it. With 303 patients from one clinic the intervals are wide and there is no external test set."
    )
st.caption("Code, notebook and full write-up: github.com/Komaljha1504/heart-disease-ml-pipeline  |  Data: UCI Heart Disease (Cleveland), CC BY 4.0")
