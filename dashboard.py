import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ShopIQ | Shopping Intent Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME (matched to reference dashboard: dark navy + teal/blue gradient)
# =========================================================
BG = "#0a1220"
CARD = "#0e1a2f"
CARD_BORDER = "#1c2c47"
TEAL = "#2dd4bf"
BLUE = "#3b82f6"
GREEN = "#22c55e"
ORANGE = "#f97316"
TEXT_MUTED = "#8fa2c0"

# SIDEBAR: PINNED ALWAYS OPEN
# We now remove the collapse control entirely (both the arrow inside the
# sidebar that closes it, and the arrow that would reappear to reopen it),
# so the user has no way to hide the sidebar at all.
CUSTOM_CSS = f"""
<style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    div[data-testid="stToolbar"] {{visibility: hidden;}}

    header[data-testid="stHeader"] {{
        background: transparent;
        box-shadow: none;
    }}
    /* Remove every version of the sidebar collapse/expand control */
    [data-testid="collapsedControl"] {{display: none !important;}}
    [data-testid="stSidebarCollapseButton"] {{display: none !important;}}
    [data-testid="stSidebarCollapsedControl"] {{display: none !important;}}
    section[data-testid="stSidebar"] button[kind="header"] {{display: none !important;}}
    section[data-testid="stSidebar"] > div:first-child > button {{display: none !important;}}

    /* Lock the sidebar width/visibility so it can't be dragged or hidden */
    section[data-testid="stSidebar"] {{
        min-width: 320px !important;
        max-width: 320px !important;
        transform: none !important;
        visibility: visible !important;
    }}

    .stApp {{
        background: {BG};
        color: #eef2f9;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    section[data-testid="stSidebar"] {{
        background: {CARD};
        border-right: 1px solid {CARD_BORDER};
    }}
    .block-container {{padding-top: 1.5rem; max-width: 1200px;}}

    .navbar {{
        display:flex; align-items:center; justify-content:space-between;
        padding: 14px 24px; border-radius: 16px;
        background: {CARD}; border: 1px solid {CARD_BORDER};
        margin-bottom: 22px;
    }}
    .navbar .brand {{font-weight:700; font-size:18px; display:flex; align-items:center; gap:8px;}}
    .navbar .badge {{
        background: linear-gradient(90deg,{TEAL},{BLUE});
        padding:6px 14px; border-radius:8px; font-size:13px; font-weight:600; color:#04101f;
    }}

    .hero {{
        border-radius: 22px; padding: 3px;
        background: linear-gradient(120deg, {TEAL}, {BLUE});
        margin-bottom: 22px;
    }}
    .hero-inner {{
        background: #0b1526; border-radius: 20px; padding: 34px;
    }}
    .pill {{
        display:inline-block; background:#132238; color:{TEAL};
        border:1px solid #1f3a55; padding:5px 14px; border-radius:20px;
        font-size:12px; font-weight:600; margin-bottom:14px;
    }}
    .hero-title {{font-size: 34px; font-weight:800; line-height:1.2; margin-bottom:10px;}}
    .hero-desc {{color:{TEXT_MUTED}; font-size:14.5px; max-width:520px;}}

    .stat-box {{
        background: {CARD}; border:1px solid {CARD_BORDER}; border-radius:18px;
        padding:20px 22px; height:100%;
    }}
    .stat-icon {{
        width:38px; height:38px; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        background:#132238; margin-bottom:14px; font-size:18px;
    }}
    .stat-value {{font-size:28px; font-weight:800; color:#fff;}}
    .stat-label {{color:{TEXT_MUTED}; font-size:12.5px; margin-top:4px;}}

    .section-title {{font-size:18px; font-weight:700; margin: 26px 0 14px 0; color:#fff;}}
    .card {{
        background:{CARD}; border:1px solid {CARD_BORDER}; border-radius:18px;
        padding: 18px 20px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cdd8ea"),
    margin=dict(l=10, r=10, t=30, b=10),
)

# =========================================================
# DATA LOADING (logic ported from shopping.py's load_data)
# =========================================================
MONTHS = {"Jan":0,"Feb":1,"Mar":2,"Apr":3,"May":4,"June":5,
          "Jul":6,"Aug":7,"Sep":8,"Oct":9,"Nov":10,"Dec":11}

FEATURE_COLS = [
    "Administrative","Administrative_Duration","Informational","Informational_Duration",
    "ProductRelated","ProductRelated_Duration","BounceRates","ExitRates","PageValues",
    "SpecialDay","Month","OperatingSystems","Browser","Region","TrafficType",
    "VisitorType","Weekend",
]


@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    df = df.dropna()

    df["Month"] = df["Month"].map(MONTHS)
    df["VisitorType"] = df["VisitorType"].apply(lambda v: 1 if v == "Returning_Visitor" else 0)
    df["Weekend"] = df["Weekend"].astype(str).map({"TRUE": 1, "True": 1, "FALSE": 0, "False": 0}).astype(int)
    df["Revenue"] = df["Revenue"].astype(str).map({"TRUE": 1, "True": 1, "FALSE": 0, "False": 0}).astype(int)

    evidence = df[FEATURE_COLS].astype(float)
    labels = df["Revenue"].astype(int)
    return df, evidence, labels


@st.cache_resource(show_spinner=False)
def train_and_eval(evidence, labels, k, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=test_size, random_state=42
    )
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    correct = int((y_test.values == y_pred).sum())
    incorrect = int((y_test.values != y_pred).sum())

    tp = ((y_test.values == 1) & (y_pred == 1)).sum()
    fn = ((y_test.values == 1) & (y_pred == 0)).sum()
    tn = ((y_test.values == 0) & (y_pred == 0)).sum()
    fp = ((y_test.values == 0) & (y_pred == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    imp = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)

    return dict(
        model=model, y_test=y_test, y_pred=y_pred,
        correct=correct, incorrect=incorrect,
        sensitivity=sensitivity, specificity=specificity,
        cm=confusion_matrix(y_test, y_pred),
        importances=imp.importances_mean,
    )


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    uploaded = st.file_uploader("Dataset CSV file (shopping.csv)", type=["csv"])
    k = st.slider("Number of neighbors (k)", 1, 15, 1)
    test_size = st.slider("Test set ratio", 0.05, 0.4, 0.1, 0.05)
    st.markdown("---")
    st.caption("The last column must be Revenue (TRUE/FALSE) — matching the structure of shopping.py")

# =========================================================
# NAVBAR
# =========================================================
st.markdown(f"""
<div class="navbar">
    <div class="brand">🛍️ <span>ShopIQ</span></div>
    <div class="badge">KNN Intent Classifier</div>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.markdown(f"""
    <div class="hero"><div class="hero-inner">
        <div class="pill">⚡ NEXT-GEN RETAIL AI</div>
        <div class="hero-title">Shopping Intent<br>Prediction Dashboard</div>
        <div class="hero-desc">Upload an online shopping dataset CSV from the sidebar to train the KNN model and see real results.</div>
    </div></div>
    """, unsafe_allow_html=True)
    st.stop()

# =========================================================
# LOAD + TRAIN
# =========================================================
with st.spinner("Processing data and training the model..."):
    df, evidence, labels = load_data(uploaded)
    res = train_and_eval(evidence, labels, k, test_size)

total = res["correct"] + res["incorrect"]
accuracy = res["correct"] / total if total else 0

# =========================================================
# HERO SECTION (real accuracy + real feature-importance chart)
# =========================================================
imp_series = pd.Series(res["importances"], index=FEATURE_COLS).sort_values(ascending=False).head(6)
bar_colors = [TEAL, BLUE, GREEN, ORANGE, "#a78bfa", "#f472b6"]

fig_hero = go.Figure(go.Bar(
    x=imp_series.index, y=imp_series.values,
    marker_color=bar_colors[:len(imp_series)],
    marker_line_width=0,
))
fig_hero.update_layout(**PLOTLY_DARK, height=230, showlegend=False,
                        xaxis=dict(tickangle=-20, tickfont=dict(size=10)),
                        yaxis=dict(showgrid=False, visible=False))

hero_l, hero_r = st.columns([1.1, 1])
with hero_l:
    st.markdown(f"""
    <div class="hero"><div class="hero-inner">
        <div class="pill">⚡ TRAINED ON {len(df):,} RECORDS</div>
        <div class="hero-title">AI-Powered Shopping<br>Intent Prediction</div>
        <div class="hero-desc">A K-Nearest Neighbors model analyzes website visitor behavior to predict the likelihood of a final purchase (Revenue).</div>
    </div></div>
    """, unsafe_allow_html=True)
with hero_r:
    st.markdown(f"""
    <div class="hero"><div class="hero-inner" style="padding-bottom:6px;">
        <div style="color:{GREEN}; font-size:30px; font-weight:800;">{accuracy*100:.2f}%</div>
        <div class="stat-label" style="margin-bottom:10px;">Overall model accuracy</div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_hero, use_container_width=True, config={"displayModeBar": False})
    st.markdown("<div style='font-size:11px;color:{};padding-bottom:14px;'>Most influential features in the model's decision (Permutation Importance)</div></div></div>".format(TEXT_MUTED), unsafe_allow_html=True)

# =========================================================
# KPI CARDS (real numbers, no mock data)
# =========================================================
c1, c2, c3, c4 = st.columns(4)
cards = [
    ("✅", f"{res['correct']:,}", "Correct predictions", c1),
    ("❌", f"{res['incorrect']:,}", "Incorrect predictions", c2),
    ("📈", f"{res['sensitivity']*100:.2f}%", "True Positive Rate (Sensitivity)", c3),
    ("📉", f"{res['specificity']*100:.2f}%", "True Negative Rate (Specificity)", c4),
]
for icon, val, label, col in cards:
    with col:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-icon">{icon}</div>
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# CONFUSION MATRIX + REVENUE DISTRIBUTION
# =========================================================
st.markdown('<div class="section-title">📊 Model Result Analysis</div>', unsafe_allow_html=True)
cm_col, dist_col = st.columns(2)

with cm_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    cm = res["cm"]
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Predicted: No Purchase", "Predicted: Purchase"],
        y=["Actual: No Purchase", "Actual: Purchase"],
        color_continuous_scale=["#0e1a2f", TEAL],
    )
    fig_cm.update_layout(**PLOTLY_DARK, height=300, coloraxis_showscale=False)
    st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with dist_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    rev_counts = df["Revenue"].map({0: "No Purchase", 1: "Purchase"}).value_counts()
    fig_donut = go.Figure(go.Pie(
        labels=rev_counts.index, values=rev_counts.values, hole=0.6,
        marker_colors=[BLUE, GREEN],
    ))
    fig_donut.update_layout(**PLOTLY_DARK, height=300, showlegend=True,
                             legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# MONTH-WISE + VISITOR TYPE (real dataset behavior)
# =========================================================
month_col, visitor_col = st.columns(2)
inv_months = {v: k for k, v in MONTHS.items()}

with month_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    month_rate = df.groupby("Month")["Revenue"].mean().sort_index()
    fig_month = go.Figure(go.Bar(
        x=[inv_months[m] for m in month_rate.index],
        y=month_rate.values * 100,
        marker_color=TEAL,
    ))
    fig_month.update_layout(**PLOTLY_DARK, height=280, title="Purchase Rate by Month (%)",
                             yaxis=dict(showgrid=True, gridcolor="#152238"))
    st.plotly_chart(fig_month, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with visitor_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    vis_rate = df.groupby("VisitorType")["Revenue"].mean()
    vis_rate.index = vis_rate.index.map({1: "Returning Visitor", 0: "New / Other Visitor"})
    fig_visitor = go.Figure(go.Bar(
        x=vis_rate.index, y=vis_rate.values * 100,
        marker_color=[BLUE, ORANGE],
    ))
    fig_visitor.update_layout(**PLOTLY_DARK, height=280, title="Purchase Rate by Visitor Type (%)",
                               yaxis=dict(showgrid=True, gridcolor="#152238"))
    st.plotly_chart(fig_visitor, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# RAW DATA PREVIEW
# =========================================================
st.markdown('<div class="section-title">🗂️ Data Preview</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.dataframe(df.head(50), use_container_width=True, height=280)
st.markdown('</div>', unsafe_allow_html=True)