from pathlib import Path
import sys
import html
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data import load_data
from bayes import MixedBayesianPredictor
from stats import proportion_ci, chi_square

st.set_page_config(page_title="Diabetes — Probability Lab", page_icon="◉", layout="wide", initial_sidebar_state="collapsed")

DATA_PATH = ROOT / "data" / "diabetes_prediction_dataset.csv"
df = load_data(DATA_PATH)
predictor = MixedBayesianPredictor(df)

# -------------------- Theme --------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#121417; --muted:#70757d; --paper:#f6f4ef; --card:rgba(255,255,255,.76); --line:rgba(18,20,23,.09); --accent:#174a43; --accent2:#c78b42; --soft:#e7efe9; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 10% 0%, rgba(199,139,66,.10), transparent 26%), radial-gradient(circle at 90% 8%, rgba(23,74,67,.10), transparent 28%), linear-gradient(135deg,#faf9f5 0%,#f2f1ec 52%,#f8f7f3 100%); color:var(--ink); }
.block-container { max-width: 1280px; padding-top: 1.3rem; padding-bottom: 4rem; }
[data-testid="stHeader"] { background:rgba(250,249,245,.72); backdrop-filter:blur(16px); }
[data-testid="stToolbar"] { display:none; }
section[data-testid="stSidebar"] { background:#121817; }
section[data-testid="stSidebar"] * { color:#fff; }
.hero { padding: 34px 36px 30px; border:1px solid var(--line); border-radius:30px; background:linear-gradient(135deg,rgba(255,255,255,.88),rgba(255,255,255,.55)); box-shadow:0 24px 70px rgba(25,30,29,.08); position:relative; overflow:hidden; animation:rise .65s ease both; }
.hero:after { content:""; position:absolute; width:280px;height:280px;border-radius:50%;right:-90px;top:-110px;background:radial-gradient(circle,rgba(199,139,66,.22),transparent 68%); animation:drift 8s ease-in-out infinite; }
.eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:800; color:#174a43; }
.hero h1 { font-family:'Manrope',sans-serif; font-size:clamp(2rem,4vw,3.65rem); line-height:1.03; margin:.45rem 0 .7rem; letter-spacing:-.055em; }
.hero p { max-width:720px; color:#62676e; font-size:1.03rem; line-height:1.65; margin:0; }
.badge { display:inline-flex; align-items:center; gap:7px; padding:7px 11px; border-radius:999px; background:#eef3ef; color:#174a43; font-size:.76rem; font-weight:700; margin-top:20px; }
.dot { width:7px;height:7px;border-radius:50%;background:#c78b42; box-shadow:0 0 0 5px rgba(199,139,66,.12); }
.nav { margin:18px 0 12px; padding:7px; border:1px solid var(--line); background:rgba(255,255,255,.65); border-radius:18px; box-shadow:0 10px 30px rgba(20,25,23,.05); }
.card { border:1px solid var(--line); background:var(--card); backdrop-filter:blur(16px); border-radius:24px; padding:24px; box-shadow:0 16px 45px rgba(20,25,23,.06); animation:rise .5s ease both; }
.metric { padding:20px 22px; border-radius:22px; background:rgba(255,255,255,.72); border:1px solid var(--line); box-shadow:0 12px 35px rgba(20,25,23,.05); transition:transform .25s ease, box-shadow .25s ease; }
.metric:hover { transform:translateY(-3px); box-shadow:0 20px 45px rgba(20,25,23,.10); }
.metric .k { color:#737980;font-size:.74rem;text-transform:uppercase;letter-spacing:.12em;font-weight:800; }
.metric .v { font-family:'Manrope',sans-serif;font-size:2rem;font-weight:800;letter-spacing:-.04em;margin-top:6px; }
.metric .s { color:#737980;font-size:.82rem;margin-top:3px; }
.section-title { font-family:'Manrope',sans-serif; font-size:1.45rem; font-weight:800; letter-spacing:-.035em; margin:6px 0 3px; }
.section-sub { color:#777c83; font-size:.9rem; margin-bottom:18px; }
.question { font-size:.75rem; font-weight:800; text-transform:uppercase; letter-spacing:.11em; color:#62676e; margin:3px 0 8px; }
.result { padding:28px; border-radius:28px; background:linear-gradient(135deg,#153f39,#205d53); color:#fff; box-shadow:0 24px 60px rgba(23,74,67,.22); position:relative; overflow:hidden; animation:glowin .7s ease both; }
.result:after { content:""; position:absolute; width:360px;height:360px;right:-130px;top:-180px;border-radius:50%;border:1px solid rgba(255,255,255,.13); box-shadow:0 0 0 40px rgba(255,255,255,.025),0 0 0 80px rgba(255,255,255,.02); }
.result .small { color:rgba(255,255,255,.65); text-transform:uppercase; letter-spacing:.12em;font-size:.72rem;font-weight:800; }
.result .big { font-family:'Manrope',sans-serif; font-size:clamp(3rem,7vw,5.8rem); line-height:.9; font-weight:800; letter-spacing:-.07em; margin:10px 0; }
.result .label { font-size:1rem; color:rgba(255,255,255,.82); }
.pill { display:inline-block;padding:7px 12px;border-radius:999px;background:rgba(255,255,255,.12);font-size:.78rem;font-weight:700;margin-top:15px; }
.footer { margin-top:36px; padding:18px 2px; color:#8a8d91; font-size:.78rem; border-top:1px solid var(--line); }
.stButton > button { border-radius:14px; border:1px solid rgba(18,20,23,.12); font-weight:700; min-height:44px; transition:all .22s ease; }
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 10px 25px rgba(20,25,23,.10); }
div[data-testid="stForm"] { border:0; padding:0; }
div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input { border-radius:13px !important; }
div[role="radiogroup"] { gap:10px; }
div[role="radiogroup"] label { border:1px solid var(--line); border-radius:14px; padding:11px 16px; background:rgba(255,255,255,.65); transition:.2s ease; }
div[role="radiogroup"] label:hover { transform:translateY(-1px); border-color:rgba(23,74,67,.28); }
[data-testid="stMetric"] { background:transparent; }
[data-testid="stDataFrame"] { border-radius:18px; overflow:hidden; }
@keyframes rise { from {opacity:0;transform:translateY(13px)} to {opacity:1;transform:translateY(0)} }
@keyframes glowin { from {opacity:0;transform:scale(.985)} to {opacity:1;transform:scale(1)} }
@keyframes drift { 0%,100% {transform:translate(0,0)} 50% {transform:translate(-16px,13px)} }
@media(max-width:760px){ .hero{padding:26px 22px;border-radius:23px}.card{padding:18px;border-radius:20px}.block-container{padding:1rem 1rem 3rem}.result{padding:22px;border-radius:22px} }
</style>
""", unsafe_allow_html=True)

# -------------------- Helpers --------------------
def fmt_pct(x): return f"{x*100:.2f}%"
def card_metric(k, v, s):
    st.markdown(f'<div class="metric"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div><div class="s">{html.escape(s)}</div></div>', unsafe_allow_html=True)

def plotly_theme(fig, height=410):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#33383d"), margin=dict(l=10,r=10,t=48,b=10),
        hoverlabel=dict(bgcolor="#121817", font_color="#fff"),
        legend=dict(bgcolor="rgba(255,255,255,.65)", bordercolor="rgba(0,0,0,.06)", borderwidth=1)
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(18,20,23,.07)", zeroline=False)
    return fig

# -------------------- Session flow --------------------
if "gender_selected" not in st.session_state: st.session_state.gender_selected = None
if "prediction" not in st.session_state: st.session_state.prediction = None

# -------------------- Header --------------------
st.markdown("""<div class="hero"><div class="eyebrow">Probability & Statistics · Diabetes Lab</div><h1>Read the probability.<br>Understand the pattern.</h1><p>A statistical diabetes analysis built around conditional probability, Bayesian inference and interactive evidence — presented as a quiet, premium clinical-style dashboard.</p><div class="badge"><span class="dot"></span> 100,000 observations · statistical model active</div></div>""", unsafe_allow_html=True)

# -------------------- Gender-first gate --------------------
if st.session_state.gender_selected is None:
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="eyebrow">01 · Start here</div><div class="section-title">Before anything else — how should we classify the patient?</div><div class="section-sub">Choose the biological sex category used by the dataset. This becomes part of the conditional probability calculation.</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,1,2])
    with c1:
        if st.button("Male", use_container_width=True, key="male_start"):
            st.session_state.gender_selected = "Male"; st.rerun()
    with c2:
        if st.button("Female", use_container_width=True, key="female_start"):
            st.session_state.gender_selected = "Female"; st.rerun()
    with c3:
        st.markdown('<div style="padding:10px 0 0 8px;color:#777c83;font-size:.82rem">The source dataset also contains an “Other” category, but the patient flow intentionally focuses on Male/Female as requested.</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">For educational/statistical use only. This tool estimates probability from the supplied dataset and is not a medical diagnosis.</div>', unsafe_allow_html=True)
    st.stop()

# -------------------- Navigation --------------------
page = st.radio("", ["Prediction", "Probability", "Statistics", "Visuals"], horizontal=True, label_visibility="collapsed")

# -------------------- Prediction --------------------
if page == "Prediction":
    st.markdown(f'<div class="nav"><span class="eyebrow">02 · Patient profile</span> &nbsp; <strong>{html.escape(st.session_state.gender_selected)}</strong> <span style="color:#777">· gender locked in</span></div>', unsafe_allow_html=True)
    left,right = st.columns([1.08,.92], gap="large")
    with left:
        st.markdown('<div class="card"><div class="section-title">Complete the profile</div><div class="section-sub">These measurements are passed into the probability model.</div>', unsafe_allow_html=True)
        with st.form("patient_form"):
            a,b = st.columns(2)
            with a:
                age = st.number_input("Age", min_value=0.1, max_value=100.0, value=40.0, step=1.0)
                bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.3, step=0.1, format="%.1f")
                hba1c = st.number_input("HbA1c level", min_value=3.0, max_value=10.0, value=5.7, step=0.1, format="%.1f")
            with b:
                glucose = st.number_input("Blood glucose level", min_value=40, max_value=300, value=120, step=1)
                hypertension = st.selectbox("Hypertension", [0,1], format_func=lambda x: "Yes" if x else "No")
                heart = st.selectbox("Heart disease", [0,1], format_func=lambda x: "Yes" if x else "No")
            smoking = st.selectbox("Smoking history", ["never","No Info","former","current","not current","ever"], index=0)
            submitted = st.form_submit_button("Calculate probability  →", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("← Choose a different gender", key="change_gender"):
            st.session_state.gender_selected=None; st.session_state.prediction=None; st.rerun()
    with right:
        if submitted:
            sample={"gender":st.session_state.gender_selected,"age":age,"hypertension":hypertension,"heart_disease":heart,"smoking_history":smoking,"bmi":bmi,"HbA1c_level":hba1c,"blood_glucose_level":glucose}
            p1,p0=predictor.predict(sample)
            st.session_state.prediction=(sample,p1,p0)
        if st.session_state.prediction:
            sample,p1,p0=st.session_state.prediction
            if p1 < .10: level="Lower probability"
            elif p1 < .30: level="Elevated probability"
            elif p1 < .60: level="Higher probability"
            else: level="Very high probability"
            st.markdown(f'<div class="result"><div class="small">Estimated diabetes probability</div><div class="big">{p1*100:.1f}%</div><div class="label">{html.escape(level)} · based on this dataset and profile</div><div class="pill">{html.escape(sample["gender"])} · Bayesian estimate</div></div>', unsafe_allow_html=True)
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1: card_metric("Diabetes",fmt_pct(p1),"posterior probability")
            with c2: card_metric("Non-diabetes",fmt_pct(p0),"posterior probability")
            fig=go.Figure(go.Bar(x=[p1*100,p0*100],y=["Diabetes","Non-diabetes"],orientation="h",text=[fmt_pct(p1),fmt_pct(p0)],textposition="auto",marker=dict(color=["#174a43","#d9ddd9"])))
            fig.update_layout(xaxis_title="Probability (%)", yaxis_title="", xaxis=dict(range=[0,100]))
            st.plotly_chart(plotly_theme(fig,300),use_container_width=True,config={"displayModeBar":False})
            st.caption("Educational estimate only — not a diagnosis or clinical risk score.")
        else:
            st.markdown('<div class="card" style="min-height:420px;display:flex;align-items:center;justify-content:center;text-align:center"><div><div class="eyebrow">Waiting for profile</div><div class="section-title">Your probability result will appear here.</div><div class="section-sub">Complete the profile and run the calculation to reveal the posterior probability.</div></div></div>',unsafe_allow_html=True)

# -------------------- Probability --------------------
elif page == "Probability":
    st.markdown('<div class="section-title">Probability, without the black box</div><div class="section-sub">The dataset lets us inspect conditional probabilities directly — the same ideas behind the predictor.</div>',unsafe_allow_html=True)
    overall=df.diabetes.mean()
    male=df.loc[df.gender=="Male","diabetes"].mean(); female=df.loc[df.gender=="Female","diabetes"].mean()
    c1,c2,c3,c4=st.columns(4)
    with c1: card_metric("P(Diabetes)",fmt_pct(overall),"entire dataset")
    with c2: card_metric("P(D | Male)",fmt_pct(male),"conditional probability")
    with c3: card_metric("P(D | Female)",fmt_pct(female),"conditional probability")
    with c4: card_metric("Observations",f"{len(df):,}","records analysed")
    st.markdown('<div style="height:18px"></div>',unsafe_allow_html=True)
    l,r=st.columns(2,gap="large")
    with l:
        gender_rates=df[df.gender.isin(["Male","Female"])].groupby("gender",as_index=False).diabetes.mean(); gender_rates["rate"]=gender_rates.diabetes*100
        fig=px.bar(gender_rates,x="gender",y="rate",text=gender_rates["rate"].map(lambda x:f"{x:.2f}%"),title="Diabetes probability by gender",labels={"gender":"Gender","rate":"Probability (%)"},template="simple_white")
        fig.update_traces(marker_color="#174a43",textposition="outside")
        st.plotly_chart(plotly_theme(fig),use_container_width=True)
    with r:
        risk_rows=[]
        for f,label in [("hypertension","Hypertension"),("heart_disease","Heart disease")]:
            for v in [0,1]:
                s=df[df[f]==v].diabetes.mean(); risk_rows.append({"Factor":label,"State":"Yes" if v else "No","Probability":s*100})
        rr=pd.DataFrame(risk_rows)
        fig=px.bar(rr,x="Factor",y="Probability",color="State",barmode="group",title="Conditional probability by existing condition",labels={"Probability":"P(Diabetes | condition) (%)"},color_discrete_sequence=["#174a43","#c78b42"])
        st.plotly_chart(plotly_theme(fig),use_container_width=True)
    st.markdown('<div class="card"><div class="section-title">Bayes intuition</div><div class="section-sub">For a profile X, the model combines the prior probability with the likelihood of each observed feature.</div><div style="font-family:Manrope;font-size:1.25rem;font-weight:700;padding:12px 0">P(D | X) ∝ P(D) × P(Gender | D) × P(Age | D) × …</div><div style="color:#777;line-height:1.6">Numerical variables use Gaussian likelihoods. Categorical variables use smoothed frequency likelihoods. Calculations are performed in log-space for numerical stability.</div></div>',unsafe_allow_html=True)

# -------------------- Statistics --------------------
elif page == "Statistics":
    st.markdown('<div class="section-title">Statistical evidence</div><div class="section-sub">Descriptive statistics, confidence intervals and independence tests from the full dataset.</div>',unsafe_allow_html=True)
    n=len(df); successes=int(df.diabetes.sum()); p,lo,hi=proportion_ci(successes,n)
    c1,c2,c3=st.columns(3)
    with c1: card_metric("Diabetes prevalence",fmt_pct(p),f"95% CI: {lo*100:.2f}%–{hi*100:.2f}%")
    with c2: card_metric("Mean age",f"{df.age.mean():.1f}",f"SD {df.age.std():.1f} years")
    with c3: card_metric("Mean glucose",f"{df.blood_glucose_level.mean():.1f}",f"SD {df.blood_glucose_level.std():.1f}")
    st.markdown('<div style="height:18px"></div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="section-title">Descriptive summary</div></div>',unsafe_allow_html=True)
    st.dataframe(df[["age","bmi","HbA1c_level","blood_glucose_level","diabetes"]].describe().T.round(3),use_container_width=True)
    st.markdown('<div style="height:18px"></div>',unsafe_allow_html=True)
    for feature,label in [("gender","Gender"),("hypertension","Hypertension"),("heart_disease","Heart disease"),("smoking_history","Smoking history")]:
        table,chi2,pval,dof=chi_square(df,feature)
        sig="statistically significant" if pval<0.05 else "not statistically significant"
        st.markdown(f'<div class="card"><div class="section-title">{label} × diabetes</div><div class="section-sub">Chi-square = {chi2:.3f} · df = {dof} · p-value = {pval:.3e} · <strong>{sig}</strong> at α = 0.05</div></div>',unsafe_allow_html=True)
        st.dataframe(table,use_container_width=True)

# -------------------- Visuals --------------------
else:
    st.markdown('<div class="section-title">Explore the dataset</div><div class="section-sub">Interactive visuals for the variables that drive the probability model.</div>',unsafe_allow_html=True)
    v1,v2=st.columns(2,gap="large")
    with v1:
        fig=px.histogram(df,x="blood_glucose_level",color="diabetes",nbins=45,marginal="box",title="Blood glucose distribution",labels={"blood_glucose_level":"Blood glucose","diabetes":"Diabetes"},color_discrete_map={0:"#cfd4d0",1:"#174a43"})
        st.plotly_chart(plotly_theme(fig),use_container_width=True)
    with v2:
        fig=px.scatter(df.sample(min(5000,len(df)),random_state=7),x="bmi",y="blood_glucose_level",color="diabetes",opacity=.55,title="BMI vs blood glucose",labels={"bmi":"BMI","blood_glucose_level":"Blood glucose"},color_discrete_map={0:"#cfd4d0",1:"#174a43"})
        st.plotly_chart(plotly_theme(fig),use_container_width=True)
    v3,v4=st.columns(2,gap="large")
    with v3:
        tmp=df.groupby("smoking_history",as_index=False).diabetes.mean(); tmp["rate"]=tmp.diabetes*100
        fig=px.bar(tmp.sort_values("rate"),x="rate",y="smoking_history",orientation="h",text=tmp.sort_values("rate")["rate"].map(lambda x:f"{x:.2f}%"),title="Diabetes probability by smoking history",labels={"rate":"Probability (%)","smoking_history":"Smoking history"})
        fig.update_traces(marker_color="#c78b42",textposition="outside")
        st.plotly_chart(plotly_theme(fig),use_container_width=True)
    with v4:
        corr=df[["age","bmi","HbA1c_level","blood_glucose_level","diabetes"]].corr()
        fig=px.imshow(corr,text_auto=".2f",aspect="auto",title="Numerical correlation matrix",color_continuous_scale=["#f2f3ef","#9ebbb4","#174a43"])
        st.plotly_chart(plotly_theme(fig),use_container_width=True)
    st.markdown('<div class="card"><div class="section-title">Dataset snapshot</div><div class="section-sub">A small sample is shown for readability; all statistical calculations use the complete 100,000-row dataset.</div></div>',unsafe_allow_html=True)
    st.dataframe(df.sample(20,random_state=42),use_container_width=True)

st.markdown('<div class="footer">Diabetes Probability Lab · Built for Probability & Statistics · Statistical/educational use only · The model does not provide medical diagnosis.</div>',unsafe_allow_html=True)
