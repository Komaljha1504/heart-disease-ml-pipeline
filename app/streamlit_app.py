"""Streamlit demo for the heart-disease pipeline: enter a patient profile in the sidebar, see the risk estimate and why, and browse the results.

    streamlit run app/streamlit_app.py

EDUCATIONAL DEMO ONLY. Not medical advice.
"""
import json
import math
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
ROSE, TEAL, AMBER, PURPLE = "#e8505b", "#2fb67c", "#f5a524", "#7c5cff"

st.set_page_config(page_title="CardioLens | Heart disease pipeline", page_icon="🫀", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
<style>
.block-container{max-width:1150px;padding-top:1.2rem}
[data-testid="stSidebar"]{border-right:1px solid #f1d9cd}
.hero{position:relative;overflow:hidden;background:linear-gradient(120deg,#ff7a6b 0%,#e8505b 42%,#a64ac9 100%);padding:28px 32px;border-radius:22px;box-shadow:0 18px 40px -24px rgba(166,74,201,.6)}
.hero h1{margin:0 0 6px;font-size:34px;color:#fff;letter-spacing:-.02em}
.hero h1 span{color:#ffe4a8}
.hero p{margin:0;color:rgba(255,255,255,.92);font-size:15px;max-width:760px;line-height:1.6}
.chips{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}
.chip{border:1px solid rgba(255,255,255,.45);background:rgba(255,255,255,.16);padding:4px 12px;border-radius:999px;font-size:12.5px;color:#fff}
.notice{background:#fff3d6;border:1px solid #f3d38b;color:#6b4a00;padding:11px 16px;border-radius:12px;font-size:13.5px;margin:14px 0}
.card{background:#fff;border:1px solid #f1d9cd;border-radius:16px;padding:16px 18px;box-shadow:0 10px 24px -18px rgba(166,74,201,.5)}
.kpi b{display:block;font-size:26px;background:linear-gradient(90deg,#e8505b,#a64ac9);-webkit-background-clip:text;background-clip:text;color:transparent}
.kpi span{font-size:12px;color:#7a6577}
.verdict{font-size:20px;font-weight:700;margin-top:-6px}
</style>
<div class="hero"><h1>Cardio<span>Lens</span></h1>
<p>Enter a patient profile in the sidebar and see how a <b>logistic-regression model</b> estimates the likelihood of heart disease, and <b>why</b>.
Behind it: a leak-free pipeline with <b>5 models</b> compared under identical repeated cross-validation and the top two tuned with <b>nested CV</b>.</p>
<div class="chips"><span class="chip">scikit-learn</span><span class="chip">303 patients</span><span class="chip">15 CV splits</span><span class="chip">Nested-CV tuning</span><span class="chip">Permutation importance</span></div></div>
<div class="notice"><b>Educational demo, not medical advice.</b> Trained on 303 patients from one clinic; it needs results of specialist tests (vessel imaging, thallium scan), so it cannot screen anyone from a routine check-up.</div>
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
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.3, max_iter=2000, random_state=42))]).fit(X, y)


def explain(model, row: pd.DataFrame):
    """Per-feature contribution to the log-odds (logistic regression is linear, so this is exact)."""
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    contrib = pre.transform(row)[0] * clf.coef_[0]
    out = {}
    for n, c in zip(pre.get_feature_names_out(), contrib):
        col = n.split("__", 1)[1]
        key = next((f for f in sorted(NICE, key=len, reverse=True) if col == f or col.startswith(f + "_")), col)
        out[key] = out.get(key, 0.0) + float(c)
    return pd.Series(out)


def gauge(p: float, colour: str):
    d = pd.DataFrame({"s": [0.0], "e": [p], "full": [1.0], "label": [f"{p*100:.0f}%"]})
    th = lambda f: alt.Theta(f, type="quantitative", scale=alt.Scale(domain=[0, 1], range=[-math.pi / 2, math.pi / 2]), stack=None)
    back = alt.Chart(d).mark_arc(innerRadius=78, outerRadius=105, color="#f3e0d6", cornerRadius=14).encode(theta=th("s"), theta2="full")
    front = alt.Chart(d).mark_arc(innerRadius=78, outerRadius=105, color=colour, cornerRadius=14).encode(theta=th("s"), theta2="e")
    text = alt.Chart(d).mark_text(fontSize=44, fontWeight="bold", color="#2b1d2e", dy=-6).encode(text="label:N")
    return (back + front + text).properties(height=150, width=300).configure_view(strokeWidth=0).configure(background="transparent")


df = load_data()
model = fit_model()
metrics = json.loads((ROOT / "results" / "metrics.json").read_text())

# ---------------- sidebar: patient form ----------------
sb = st.sidebar
sb.markdown("### 🫀 Patient profile")
presets = {"Custom": None,
           "Example: lower-risk patient": df[df.target == 0].sort_values("thalach").iloc[-5],
           "Example: higher-risk patient": df[df.target == 1].sort_values("oldpeak").iloc[-5]}
choice = sb.selectbox("Start from", list(presets), help="Pick a real row from the dataset, then edit any value.")
base = presets[choice] if presets[choice] is not None else df.median(numeric_only=True)
g = lambda k, d: float(base[k]) if k in base and not pd.isna(base[k]) else d
key = choice.replace(" ", "_")  # new widget keys per preset so the values reset

age = sb.slider("Age", 29, 77, int(g("age", 54)), key=f"a{key}")
sex = sb.radio("Sex", [1, 0], format_func=lambda v: "Male" if v == 1 else "Female", index=0 if g("sex", 1) == 1 else 1, horizontal=True, key=f"s{key}")
cp = sb.selectbox("Chest-pain type", [1, 2, 3, 4], index=int(g("cp", 3)) - 1, key=f"c{key}",
                  format_func=lambda v: {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}[v])
trestbps = sb.slider("Resting blood pressure (mm Hg)", 90, 200, int(g("trestbps", 130)), key=f"t{key}")
chol = sb.slider("Cholesterol (mg/dl)", 120, 570, int(g("chol", 240)), key=f"h{key}")
thalach = sb.slider("Max heart rate achieved", 70, 205, int(g("thalach", 150)), key=f"m{key}")
oldpeak = sb.slider("ST depression (exercise vs rest)", 0.0, 6.5, float(g("oldpeak", 1.0)), 0.1, key=f"o{key}")
with sb.expander("More measurements", expanded=True):
    fbs = st.radio("Fasting blood sugar > 120 mg/dl", [0, 1], index=int(g("fbs", 0)), format_func=lambda v: "Yes" if v else "No", horizontal=True, key=f"f{key}")
    exang = st.radio("Exercise-induced angina", [0, 1], index=int(g("exang", 0)), format_func=lambda v: "Yes" if v else "No", horizontal=True, key=f"e{key}")
    restecg = st.selectbox("Resting ECG", [0, 1, 2], index=int(g("restecg", 0)), key=f"r{key}",
                           format_func=lambda v: {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}[v])
    slope = st.selectbox("ST slope", [1, 2, 3], index=int(g("slope", 2)) - 1, key=f"l{key}",
                         format_func=lambda v: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[v])
    ca = st.selectbox("Major vessels coloured (0 to 3)", [0, 1, 2, 3], index=int(g("ca", 0)), key=f"v{key}")
    thal = st.selectbox("Thallium test", [3, 6, 7], index=[3, 6, 7].index(int(g("thal", 3))), key=f"th{key}",
                        format_func=lambda v: {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}[v])

row = add_features(pd.DataFrame([dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg, thalach=thalach,
                                       exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)]))
p = float(model.predict_proba(row)[0, 1])
colour, verdict = (TEAL, "Lower estimated likelihood") if p < 0.35 else ((AMBER, "Uncertain, mid-range") if p < 0.65 else (ROSE, "Higher estimated likelihood"))

# ---------------- main ----------------
tab_try, tab_res = st.tabs(["🩺  Estimate", "📊  Results & analysis"])

with tab_try:
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.markdown("##### Estimated likelihood of heart disease")
        st.altair_chart(gauge(p, colour), use_container_width=True)
        st.markdown(f'<div class="verdict" style="color:{colour};text-align:center">{verdict}</div>', unsafe_allow_html=True)
        st.caption("From the logistic regression fitted on all 303 patients. Cross-validated it reaches ROC-AUC 0.92, so estimates are informative but far from certain, and they are not a diagnosis.")
    with right:
        c = explain(model, row).sort_values()
        top = pd.concat([c.head(4), c.tail(5)])
        top = top[~top.index.duplicated()]
        cdf = pd.DataFrame({"factor": [NICE.get(k, k) for k in top.index], "effect": top.values})
        cdf["direction"] = np.where(cdf.effect > 0, "raises the estimate", "lowers the estimate")
        ch = alt.Chart(cdf).mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X("effect:Q", title="effect on log-odds for this patient"),
            y=alt.Y("factor:N", sort="-x", title=None, axis=alt.Axis(labelLimit=280, labelOverlap=False)),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["raises the estimate", "lowers the estimate"], range=[ROSE, PURPLE]), legend=alt.Legend(orient="bottom", title=None)),
        ).properties(height=330, title="Why: what moved this estimate most")
        st.altair_chart(ch, use_container_width=True)

with tab_res:
    comp = metrics["comparison_with_engineered_features"]
    rows = [{"model": m, "AUC": v["roc_auc"][0], "std": v["roc_auc"][1], "Accuracy": v["accuracy"][0], "Recall": v["recall"][0], "F1": v["f1"][0], "Precision": v["precision"][0]} for m, v in comp.items()]
    r = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    best = r.iloc[0]
    cols = st.columns(4)
    for col, (v, t) in zip(cols, [(f"{best.AUC:.2f}", f"ROC-AUC, {best.model} (± {best['std']:.2f})"), (f"{best.Accuracy*100:.1f}%", "accuracy, cross-validated"),
                                  ("5", "models on identical 15 CV splits"), ("0", "gain from tuning or engineered features")]):
        col.markdown(f'<div class="card kpi"><b>{v}</b><span>{t}</span></div>', unsafe_allow_html=True)
    st.markdown("")
    bars = alt.Chart(r).mark_bar(cornerRadiusEnd=6, color=ROSE).encode(
        x=alt.X("AUC:Q", scale=alt.Scale(domain=[0.75, 1.0]), title="ROC-AUC (mean ± std over 15 splits)"), y=alt.Y("model:N", sort="-x", title=None))
    err = alt.Chart(r).mark_errorbar(color="#2b1d2e").encode(y=alt.Y("model:N", sort="-x"), x=alt.X("lo:Q"), x2="hi:Q").transform_calculate(lo="datum.AUC - datum.std", hi="datum.AUC + datum.std")
    st.altair_chart((bars + err).properties(height=250, title="Model comparison under the same cross-validation"), use_container_width=True)
    st.dataframe(r.set_index("model").round(3), use_container_width=True)
    a, b = st.columns(2)
    a.image(str(ROOT / "results" / "feature_importance.png"), caption="Permutation importance on held-out folds", use_container_width=True)
    b.image(str(ROOT / "results" / "final_model_roc_confusion.png"), caption="Out-of-fold ROC and confusion matrix", use_container_width=True)
    st.markdown(
        "**Honest takeaways.** The simplest model won, and the gaps between the top models (about 0.01 AUC) are smaller than the fold-to-fold spread (about 0.04). "
        "Tuning and my four engineered features gave no gain, and I report that instead of hiding it. With 303 patients from one clinic the intervals are wide and there is no external test set."
    )
st.caption("Code, notebook and full write-up: github.com/Komaljha1504/heart-disease-ml-pipeline  |  Data: UCI Heart Disease (Cleveland), CC BY 4.0")
