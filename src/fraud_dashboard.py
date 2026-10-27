import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, PercentFormatter

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN TOKENS
# ============================================================

BG = "#0A0D12"
SURFACE = "#10141B"
SURFACE_2 = "#151A23"
LINE = "#1F2631"
TEXT = "#EDF0F5"
MUTED = "#8A93A3"
DIM = "#5A6372"
ACCENT = "#F22952"
NEUTRAL = "#6F82A8"

RISK_COLORS = {
    "HIGH": "#F0506E",
    "MEDIUM": "#E9B44C",
    "LOW": "#4CC9A0",
}

RISK_COPY = {
    "HIGH": (
        "High risk",
        "Hold for review",
        "The model sees a strong fraud pattern. Pause this transaction "
        "and review the contributing factors before approving.",
    ),
    "MEDIUM": (
        "Medium risk",
        "Verify before approving",
        "The model sees some warning signs. Additional verification, "
        "such as contacting the account holder, is appropriate.",
    ),
    "LOW": (
        "Low risk",
        "Standard processing",
        "The model found no strong fraud signal. Normal processing "
        "may be appropriate.",
    ),
}


# ============================================================
# THEME (CSS)
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');

:root {
    --bg: #0A0D12;
    --surface: #10141B;
    --surface-2: #151A23;
    --line: #1F2631;
    --line-soft: rgba(255,255,255,0.06);
    --text: #EDF0F5;
    --muted: #8A93A3;
    --dim: #5A6372;
    --accent: #F22952;
    --high: #F0506E;
    --medium: #E9B44C;
    --low: #4CC9A0;
}

/* ---------- Base ---------- */

.stApp {
    background:
        radial-gradient(900px 380px at 12% -8%, rgba(242,41,82,0.06), transparent 65%),
        var(--bg);
    color: var(--text);
    font-family: 'Instrument Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.main .block-container {
    max-width: 1280px;
    padding-top: 2.2rem;
    padding-bottom: 5rem;
}

h1, h2, h3, h4, h5 {
    color: var(--text) !important;
    font-family: 'Instrument Sans', sans-serif;
    letter-spacing: -0.02em;
}

p, li, span, label { font-family: 'Instrument Sans', sans-serif; }
.stMarkdown p { color: var(--muted); }
hr { border-color: var(--line) !important; margin: 1.2rem 0 !important; }

::selection { background: rgba(242,41,82,0.35); }

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {
    background: #0C1015 !important;
    border-right: 1px solid var(--line-soft) !important;
}

section[data-testid="stSidebar"] > div {
    background: transparent !important;
}

.side-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 8px 0 34px 0;
    padding-bottom: 22px;
    border-bottom: 1px solid var(--line-soft);
}

.side-brand-mark {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: linear-gradient(145deg, #F22952, #B81238);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 6px 18px rgba(242,41,82,0.28);
    flex-shrink: 0;
}

.side-brand-name {
    color: var(--text);
    font-size: 1.7rem;
    font-weight: 650;
    letter-spacing: -0.02em;
}

.side-title {
    color: var(--text);
    font-size: 0.86rem;
    font-weight: 600;
    margin: 6px 0 12px 0;
}

.side-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid var(--line-soft);
    font-size: 0.84rem;
}

.side-row:last-of-type {
    border-bottom: none;
}

.side-row .name {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--text);
    font-weight: 500;
}

.side-row .range {
    color: var(--muted);
    font-size: 0.78rem;
}

.dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.side-note {
    color: var(--dim);
    font-size: 0.76rem;
    line-height: 1.55;
    margin-top: 18px;
}

/* ---------- Hero ---------- */

.hero {
    padding: 52px 0 10px 0;
}

.hero-title {
    font-family: 'Newsreader', Georgia, serif;
    font-weight: 400;
    font-size: 3.5rem;
    line-height: 1.04;
    letter-spacing: -0.03em;
    color: var(--text);
    max-width: 760px;
}

.hero-text {
    color: var(--muted);
    font-size: 1.05rem;
    line-height: 1.65;
    max-width: 620px;
    margin-top: 18px;
}

.capabilities {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    margin: 40px 0 8px 0;
    border-top: 1px solid var(--line-soft);
    border-bottom: 1px solid var(--line-soft);
}

.capability {
    padding: 22px 28px 22px 0;
}

.capability + .capability {
    padding-left: 28px;
    border-left: 1px solid var(--line-soft);
}

.capability-title {
    color: var(--text);
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 6px;
}

.capability-text {
    color: var(--muted);
    font-size: 0.86rem;
    line-height: 1.6;
}

.steps {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    padding: 4px 0 8px 0;
}

.step {
    background: var(--surface);
    border: 1px solid var(--line-soft);
    border-radius: 12px;
    padding: 16px 18px;
}

.step-no {
    color: var(--accent);
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 8px;
}

.step-title {
    color: var(--text);
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 4px;
}

.step-text {
    color: var(--muted);
    font-size: 0.8rem;
    line-height: 1.5;
}

/* ---------- Section headers ---------- */

.section {
    margin: 34px 0 14px 0;
}

.section-title {
    color: var(--text);
    font-size: 1.22rem;
    font-weight: 600;
    letter-spacing: -0.02em;
}

.section-sub {
    color: var(--muted);
    font-size: 0.88rem;
    margin-top: 4px;
}

.card-title {
    color: var(--text);
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 2px;
}

.card-sub {
    color: var(--dim);
    font-size: 0.8rem;
    margin-bottom: 8px;
}

/* ---------- Tabs ---------- */

div[data-baseweb="tab-list"] {
    gap: 30px;
    border-bottom: 1px solid var(--line-soft);
}

button[data-baseweb="tab"] {
    background: transparent !important;
    padding: 14px 0 !important;
    height: auto !important;
}

button[data-baseweb="tab"] p {
    color: var(--muted) !important;
    font-weight: 500;
    font-size: 0.95rem;
}

button[data-baseweb="tab"][aria-selected="true"] p {
    color: var(--text) !important;
}

div[data-baseweb="tab-highlight"] {
    background-color: var(--accent) !important;
    height: 2px !important;
}

div[data-baseweb="tab-border"] {
    background-color: transparent !important;
}

/* ---------- Inputs ---------- */

div[data-testid="stWidgetLabel"] p {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500;
}

div[data-baseweb="input"],
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] {
    background: var(--surface-2) !important;
    border-radius: 10px !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    background: var(--surface-2) !important;
    transition: border-color .15s ease, box-shadow .15s ease;
}

div[data-baseweb="input"]:focus-within > div,
div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(242,41,82,0.7) !important;
    box-shadow: 0 0 0 3px rgba(242,41,82,0.14);
}

input {
    color: var(--text) !important;
    font-weight: 500;
}

div[data-testid="stNumberInput"] button {
    background: transparent !important;
    color: var(--muted) !important;
}

div[data-testid="stNumberInput"] button:hover {
    color: var(--text) !important;
}

div[data-baseweb="popover"] ul {
    background: var(--surface-2) !important;
    border: 1px solid var(--line);
}

/* ---------- Containers ---------- */

div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] div[data-testid="stWidgetLabel"]) {
    border-radius: 16px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--line-soft) !important;
    border-radius: 16px;
}

/* ---------- Buttons ---------- */

.stButton > button {
    width: 100%;
    min-height: 50px;
    border-radius: 12px;
    border: 1px solid var(--accent);
    background: linear-gradient(180deg, #F63A5F, #E01F49);
    color: #FFFFFF;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.005em;
    box-shadow: 0 8px 22px rgba(242,41,82,0.22);
    transition: transform .12s ease, box-shadow .12s ease;
}

.stButton > button:hover {
    border-color: #FF4C70;
    color: #FFFFFF;
    box-shadow: 0 10px 28px rgba(242,41,82,0.32);
}

.stButton > button:active {
    transform: translateY(1px);
}

.stButton > button:focus-visible {
    outline: 2px solid #fff;
    outline-offset: 2px;
}

.stDownloadButton > button {
    min-height: 46px;
    border-radius: 12px;
    background: transparent;
    border: 1px solid var(--line);
    color: var(--text);
    font-weight: 600;
}

.stDownloadButton > button:hover {
    border-color: var(--accent);
    color: var(--text);
}

/* ---------- Metrics ---------- */

div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line-soft);
    border-radius: 14px;
    padding: 16px 18px;
}

div[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-size: 0.8rem !important;
}

div[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 600;
    letter-spacing: -0.02em;
}

/* ---------- KPI cards (batch) ---------- */

.kpis {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
}

.kpi {
    background: var(--surface);
    border: 1px solid var(--line-soft);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}

.kpi::before {
    content: "";
    position: absolute;
    left: 0;
    top: 18px;
    bottom: 18px;
    width: 3px;
    border-radius: 0 3px 3px 0;
    background: var(--kpi-color, transparent);
}

.kpi-label {
    color: var(--muted);
    font-size: 0.8rem;
    font-weight: 500;
}

.kpi-value {
    color: var(--text);
    font-size: 1.9rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    margin-top: 6px;
}

/* ---------- Result card ---------- */

.result {
    display: flex;
    align-items: center;
    gap: 40px;
    padding: 30px 34px;
    border-radius: 20px;
    background: linear-gradient(135deg, var(--tint), var(--surface) 55%);
    border: 1px solid var(--line-soft);
    border-left: 3px solid var(--tone);
}

.result-gauge {
    flex: 0 0 auto;
}

.result-body {
    flex: 1 1 auto;
    min-width: 0;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 12px;
    border-radius: 999px;
    background: var(--tint-strong);
    color: var(--tone);
    font-size: 0.8rem;
    font-weight: 600;
}

.badge .dot {
    background: var(--tone);
}

.result-title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.1rem;
    letter-spacing: -0.02em;
    color: var(--text);
    margin: 12px 0 8px 0;
    line-height: 1.1;
}

.result-text {
    color: var(--muted);
    font-size: 0.95rem;
    line-height: 1.6;
    max-width: 560px;
}

.chips {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 18px;
}

.chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--line-soft);
    border-radius: 10px;
    padding: 8px 12px;
}

.chip-label {
    color: var(--dim);
    font-size: 0.72rem;
}

.chip-value {
    color: var(--text);
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 1px;
}

/* ---------- SHAP contribution bars ---------- */

.contrib {
    margin-bottom: 16px;
}

.contrib-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
}

.contrib-name {
    color: var(--text);
    font-size: 0.88rem;
    font-weight: 500;
}

.contrib-value {
    color: var(--dim);
    font-size: 0.76rem;
    margin-left: 8px;
    font-weight: 400;
}

.contrib-score {
    color: var(--muted);
    font-size: 0.82rem;
    font-variant-numeric: tabular-nums;
}

.bar {
    height: 6px;
    border-radius: 6px;
    background: rgba(255,255,255,0.05);
    margin-top: 7px;
    overflow: hidden;
}

.bar > div {
    height: 100%;
    border-radius: 6px;
}

.empty {
    color: var(--dim);
    font-size: 0.88rem;
    padding: 6px 0;
}

/* ---------- Expanders ---------- */

div[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--line-soft);
    border-radius: 14px;
}

div[data-testid="stExpander"] details {
    border: none;
}

div[data-testid="stExpander"] summary p {
    color: var(--text) !important;
    font-weight: 500;
}

/* ---------- File uploader / dataframe ---------- */

div[data-testid="stFileUploader"] section {
    background: var(--surface);
    border: 1px dashed #2B3442;
    border-radius: 14px;
}

div[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent);
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--line-soft);
    border-radius: 12px;
    overflow: hidden;
}

.note {
    background: var(--surface);
    border: 1px solid var(--line-soft);
    border-radius: 14px;
    padding: 18px 22px;
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.7;
}

.note code {
    background: var(--surface-2);
    color: var(--text);
    border-radius: 6px;
    padding: 2px 7px;
    font-size: 0.8rem;
}

/* ---------- Responsive ---------- */

@media (max-width: 900px) {
    .hero-title {
        font-size: 2.5rem;
    }

    .capabilities,
    .steps,
    .kpis {
        grid-template-columns: 1fr;
    }

    .capability + .capability {
        padding-left: 0;
        border-left: none;
        border-top: 1px solid var(--line-soft);
    }

    .result {
        flex-direction: column;
        align-items: flex-start;
        gap: 22px;
        padding: 24px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# CHART THEME
# ============================================================

plt.rcParams.update(
    {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": TEXT,
        "font.family": "sans-serif",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#1A212B",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def new_fig(width=6.2, height=3.4):
    fig, ax = plt.subplots(figsize=(width, height))

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)

    return fig, ax


def show_fig(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def compact_number(value, _pos=None):
    abs_value = abs(value)

    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs_value >= threshold:
            return f"{value / threshold:.1f}".rstrip("0").rstrip(".") + suffix

    return f"{value:.0f}"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("models/behavioral_random_forest.pkl")
    features = joblib.load("models/model_features.pkl")
    explainer = shap.TreeExplainer(model)

    return model, features, explainer


model, model_features, explainer = load_model()


# ============================================================
# HELPER FUNCTIONS (model logic unchanged)
# ============================================================

def format_feature_name(name):
    names = {
        "step": "Transaction Step",
        "amount": "Transaction Amount",
        "oldbalanceOrg": "Previous Origin Balance",
        "newbalanceOrig": "New Origin Balance",
        "oldbalanceDest": "Previous Destination Balance",
        "newbalanceDest": "New Destination Balance",
        "isFlaggedFraud": "System Fraud Flag",
        "previous_transaction_count": "Previous Transaction Count",
        "previous_amount": "Previous Transaction Amount",
        "amount_vs_previous": "Amount vs Previous Transaction",
        "balance_change": "Origin Balance Change",
        "type_CASH_IN": "Transaction Type: Cash In",
        "type_CASH_OUT": "Transaction Type: Cash Out",
        "type_DEBIT": "Transaction Type: Debit",
        "type_PAYMENT": "Transaction Type: Payment",
        "type_TRANSFER": "Transaction Type: Transfer",
    }

    return names.get(name, name)


def get_risk(probability):
    if probability >= 0.70:
        return "HIGH"

    if probability >= 0.30:
        return "MEDIUM"

    return "LOW"


def create_transaction(
    step,
    transaction_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    isFlaggedFraud,
    previous_transaction_count,
    previous_amount,
):
    amount_vs_previous = amount / (previous_amount + 1)
    balance_change = oldbalanceOrg - newbalanceOrig

    data = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFlaggedFraud": isFlaggedFraud,
        "previous_transaction_count": previous_transaction_count,
        "previous_amount": previous_amount,
        "amount_vs_previous": amount_vs_previous,
        "balance_change": balance_change,
        "type_CASH_IN": int(transaction_type == "CASH_IN"),
        "type_CASH_OUT": int(transaction_type == "CASH_OUT"),
        "type_DEBIT": int(transaction_type == "DEBIT"),
        "type_PAYMENT": int(transaction_type == "PAYMENT"),
        "type_TRANSFER": int(transaction_type == "TRANSFER"),
    }

    return pd.DataFrame([data]).reindex(
        columns=model_features,
        fill_value=0,
    )


def get_shap_values(transaction):
    values = explainer.shap_values(transaction)

    if isinstance(values, list):
        if len(values) > 1:
            return values[1][0]

        return values[0][0]

    values = np.asarray(values)

    if values.ndim == 3:
        if values.shape[2] == 2:
            return values[0, :, 1]

        return values[0, :, 0]

    return values[0]


def create_batch_features(batch_df):
    batch_input = pd.DataFrame()

    batch_input["step"] = batch_df["step"]
    batch_input["amount"] = batch_df["amount"]
    batch_input["oldbalanceOrg"] = batch_df["oldbalanceOrg"]
    batch_input["newbalanceOrig"] = batch_df["newbalanceOrig"]
    batch_input["oldbalanceDest"] = batch_df["oldbalanceDest"]
    batch_input["newbalanceDest"] = batch_df["newbalanceDest"]
    batch_input["isFlaggedFraud"] = batch_df["isFlaggedFraud"]

    if (
        "previous_transaction_count" in batch_df.columns
        and "previous_amount" in batch_df.columns
    ):
        batch_input["previous_transaction_count"] = (
            batch_df["previous_transaction_count"]
        )

        batch_input["previous_amount"] = batch_df["previous_amount"]

    elif "nameOrig" in batch_df.columns:
        history = batch_df[["nameOrig", "step", "amount"]].copy()
        history["_original_index"] = history.index
        history = history.sort_values(["nameOrig", "step"])

        history["previous_transaction_count"] = (
            history.groupby("nameOrig").cumcount()
        )

        history["previous_amount"] = (
            history.groupby("nameOrig")["amount"]
            .shift(1)
            .fillna(0)
        )

        history = history.sort_values("_original_index")

        batch_input["previous_transaction_count"] = (
            history["previous_transaction_count"].values
        )

        batch_input["previous_amount"] = (
            history["previous_amount"].values
        )

    else:
        batch_input["previous_transaction_count"] = 0
        batch_input["previous_amount"] = 0

    batch_input["amount_vs_previous"] = (
        batch_input["amount"]
        / (batch_input["previous_amount"] + 1)
    )

    batch_input["balance_change"] = (
        batch_input["oldbalanceOrg"]
        - batch_input["newbalanceOrig"]
    )

    for transaction_type in [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    ]:
        batch_input[f"type_{transaction_type}"] = (
            batch_df["type"] == transaction_type
        ).astype(int)

    return batch_input.reindex(
        columns=model_features,
        fill_value=0,
    )


# ============================================================
# UI HELPERS
# ============================================================

def html(markup):
    """Render HTML safely: strip indentation so Markdown never treats it as code."""
    flat = "".join(
        line.strip()
        for line in markup.strip().splitlines()
    )

    st.markdown(
        flat,
        unsafe_allow_html=True,
    )


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")

    r, g, b = (
        int(h[i : i + 2], 16)
        for i in (0, 2, 4)
    )

    return f"rgba({r},{g},{b},{alpha})"


def section(title, subtitle=""):
    sub = (
        f'<div class="section-sub">{subtitle}</div>'
        if subtitle
        else ""
    )

    html(
        f'<div class="section">'
        f'<div class="section-title">{title}</div>'
        f'{sub}'
        f'</div>'
    )


def kpi(label, value, color=None):
    style = (
        f' style="--kpi-color:{color}"'
        if color
        else ""
    )

    return (
        f'<div class="kpi"{style}>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"</div>"
    )


def gauge_svg(probability, color):
    """Semicircle gauge with markers at the 30% and 70% risk thresholds."""

    radius, cx, cy = 92, 120, 118
    length = math.pi * radius
    filled = max(length * probability, 0.01)

    ticks = ""

    for t in (0.30, 0.70):
        angle = math.pi * (1 - t)

        x1 = cx + (radius - 13) * math.cos(angle)
        y1 = cy - (radius - 13) * math.sin(angle)

        x2 = cx + (radius + 9) * math.cos(angle)
        y2 = cy - (radius + 9) * math.sin(angle)

        ticks += (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#3A4455" stroke-width="2" '
            f'stroke-linecap="round"/>'
        )

    path = (
        f"M {cx - radius} {cy} "
        f"A {radius} {radius} 0 0 1 {cx + radius} {cy}"
    )

    return (
        f'<svg width="240" height="146" '
        f'viewBox="0 0 240 146" '
        f'xmlns="http://www.w3.org/2000/svg">'

        f'<path d="{path}" fill="none" '
        f'stroke="rgba(255,255,255,0.07)" '
        f'stroke-width="12" '
        f'stroke-linecap="round"/>'

        f'<path d="{path}" fill="none" '
        f'stroke="{color}" '
        f'stroke-width="12" '
        f'stroke-linecap="round" '
        f'stroke-dasharray="{filled:.2f} {length:.2f}"/>'

        f"{ticks}"

        f'<text x="{cx}" y="{cy - 20}" '
        f'text-anchor="middle" '
        f'fill="{TEXT}" '
        f'font-family="Newsreader, Georgia, serif" '
        f'font-size="42" '
        f'letter-spacing="-1">'
        f'{probability:.1%}'
        f'</text>'

        f'<text x="{cx}" y="{cy + 2}" '
        f'text-anchor="middle" '
        f'fill="{MUTED}" '
        f'font-family="Instrument Sans, sans-serif" '
        f'font-size="12">'
        f'fraud probability'
        f'</text>'

        f"</svg>"
    )


def format_value(raw_name, value):
    if raw_name.startswith("type_"):
        return "Yes" if value else "No"

    if float(value).is_integer():
        return f"{value:,.0f}"

    return f"{value:,.2f}"


def contribution_rows(df, color, max_abs):
    if df.empty:
        return '<div class="empty">No strong contributors.</div>'

    rows = ""

    for _, row in df.iterrows():
        width = max(
            abs(row["SHAP"]) / max_abs * 100,
            3,
        )

        sign = "+" if row["SHAP"] > 0 else "−"

        rows += (
            '<div class="contrib">'

            '<div class="contrib-top">'

            f'<div>'
            f'<span class="contrib-name">{row["Feature"]}</span>'
            f'<span class="contrib-value">'
            f'{format_value(row["Raw"], row["Value"])}'
            f'</span>'
            f'</div>'

            f'<div class="contrib-score">'
            f'{sign}{abs(row["SHAP"]):.4f}'
            f'</div>'

            "</div>"

            f'<div class="bar">'
            f'<div style="width:{width:.1f}%;background:{color}"></div>'
            f'</div>'

            "</div>"
        )

    return rows


def prediction_table(df):
    """Display copy with a progress bar for probability; the source data is untouched."""

    view = df.copy()
    view["Fraud Probability"] = (
        view["Fraud Probability"] * 100
    )

    return view


PROBABILITY_COLUMN = {
    "Fraud Probability": st.column_config.ProgressColumn(
        "Fraud Probability",
        min_value=0,
        max_value=100,
        format="%.1f%%",
    )
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    html(
        """
<div class="side-brand">
    <div class="side-brand-mark">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
             stroke="#fff" stroke-width="2" stroke-linecap="round"
             stroke-linejoin="round">
            <path d="M12 2.5 20 5.5v5.7c0 4.7-3.2 8.5-8 10.3-4.8-1.8-8-5.6-8-10.3V5.5z"/>
            <path d="m8.8 12 2.2 2.2 4.2-4.4"/>
        </svg>
    </div>

    <div class="side-brand-name">
        FraudGuard
    </div>
</div>
"""
    )

    html(
        '<div class="side-title">Risk levels</div>'
    )

    for level, label, rng in (
        ("HIGH", "High", "70% and above"),
        ("MEDIUM", "Medium", "30% to 69%"),
        ("LOW", "Low", "Below 30%"),
    ):
        html(
            f'<div class="side-row">'
            f'<div class="name">'
            f'<span class="dot" '
            f'style="background:{RISK_COLORS[level]}"></span>'
            f'{label}'
            f'</div>'
            f'<div class="range">{rng}</div>'
            f"</div>"
        )

    html(
        '<div class="side-note">'
        "Scores come from a behavioral Random Forest. "
        "SHAP explains what pushed each score up or down."
        "</div>"
    )


# ============================================================
# HERO
# ============================================================

html(
    """
<div class="hero">
    <div class="hero-title">
        Detect risk. Understand why.
    </div>

    <div class="hero-text">
        FraudGuard scores transactions on behavior as well as amount,
        then shows exactly which factors drove each decision.
    </div>
</div>

<div class="capabilities">

    <div class="capability">
        <div class="capability-title">
            Behavioral analysis
        </div>

        <div class="capability-text">
            Looks beyond the amount to previous activity and balance changes.
        </div>
    </div>

    <div class="capability">
        <div class="capability-title">
            Explainable scores
        </div>

        <div class="capability-text">
            SHAP shows which factors pushed a prediction toward or away from fraud.
        </div>
    </div>

    <div class="capability">
        <div class="capability-title">
            Clear risk levels
        </div>

        <div class="capability-text">
            Every prediction is translated into a low, medium, or high rating.
        </div>
    </div>

</div>
"""
)


with st.expander("How FraudGuard works"):

    html(
        """
<div class="steps">

    <div class="step">
        <div class="step-no">1</div>

        <div class="step-title">
            Enter a transaction
        </div>

        <div class="step-text">
            Provide the transaction and account details.
        </div>
    </div>


    <div class="step">
        <div class="step-no">2</div>

        <div class="step-title">
            Build behavior features
        </div>

        <div class="step-text">
            Previous activity and balance changes are derived.
        </div>
    </div>


    <div class="step">
        <div class="step-no">3</div>

        <div class="step-title">
            Estimate probability
        </div>

        <div class="step-text">
            The Random Forest returns a fraud probability.
        </div>
    </div>


    <div class="step">
        <div class="step-no">4</div>

        <div class="step-title">
            Explain the result
        </div>

        <div class="step-text">
            SHAP highlights the factors behind the score.
        </div>
    </div>

</div>
"""
    )


st.write("")


investigation_tab, batch_tab = st.tabs(
    ["Transaction investigation", "Batch analytics"]
)


# ============================================================
# INDIVIDUAL INVESTIGATION
# ============================================================

with investigation_tab:

    section(
        "Investigate a transaction",
        "Enter the details to get an explainable fraud risk assessment.",
    )

    left, right = st.columns(
        2,
        gap="medium",
    )

    with left:

        with st.container(border=True):

            html(
                '<div class="card-title">Transaction</div>'
                '<div class="card-sub">'
                'What happened and how this account behaved before.'
                '</div>'
            )

            c1, c2 = st.columns(2)

            with c1:

                step = st.number_input(
                    "Transaction step",
                    min_value=1,
                    value=400,
                )

                amount = st.number_input(
                    "Transaction amount",
                    min_value=0.0,
                    value=50000.0,
                    step=1000.0,
                )

                previous_transaction_count = st.number_input(
                    "Previous transaction count",
                    min_value=0,
                    value=5,
                )

            with c2:

                transaction_type = st.selectbox(
                    "Transaction type",
                    [
                        "TRANSFER",
                        "CASH_OUT",
                        "PAYMENT",
                        "CASH_IN",
                        "DEBIT",
                    ],
                )

                isFlaggedFraud = st.selectbox(
                    "System fraud flag",
                    [0, 1],
                )

                previous_amount = st.number_input(
                    "Previous transaction amount",
                    min_value=0.0,
                    value=10000.0,
                    step=1000.0,
                )


    with right:

        with st.container(border=True):

            html(
                '<div class="card-title">Balances</div>'
                '<div class="card-sub">'
                'Origin and destination accounts, before and after.'
                '</div>'
            )

            c1, c2 = st.columns(2)

            with c1:

                oldbalanceOrg = st.number_input(
                    "Previous origin balance",
                    min_value=0.0,
                    value=60000.0,
                    step=1000.0,
                )

                newbalanceOrig = st.number_input(
                    "New origin balance",
                    min_value=0.0,
                    value=10000.0,
                    step=1000.0,
                )

            with c2:

                oldbalanceDest = st.number_input(
                    "Previous destination balance",
                    min_value=0.0,
                    value=5000.0,
                    step=1000.0,
                )

                newbalanceDest = st.number_input(
                    "New destination balance",
                    min_value=0.0,
                    value=55000.0,
                    step=1000.0,
                )


    st.write("")

    analyze = st.button(
        "Analyze transaction",
        type="primary",
        use_container_width=True,
    )


    if analyze:

        transaction = create_transaction(
            step,
            transaction_type,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest,
            isFlaggedFraud,
            previous_transaction_count,
            previous_amount,
        )

        probability = model.predict_proba(
            transaction
        )[0][1]

        risk = get_risk(probability)
        tone = RISK_COLORS[risk]

        badge_text, headline, advice = RISK_COPY[risk]

        shap_values = get_shap_values(transaction)

        shap_df = pd.DataFrame(
            {
                "Raw": model_features,
                "SHAP": shap_values,
                "Value": transaction.iloc[0].values,
            }
        )

        shap_df["Feature"] = shap_df["Raw"].apply(
            format_feature_name
        )

        positive = (
            shap_df[
                shap_df["SHAP"] > 0
            ]
            .sort_values(
                "SHAP",
                ascending=False,
            )
            .head(5)
        )

        negative = (
            shap_df[
                shap_df["SHAP"] < 0
            ]
            .sort_values(
                "SHAP",
                ascending=True,
            )
            .head(5)
        )

        amount_ratio = (
            amount / (previous_amount + 1)
        )

        balance_change = (
            oldbalanceOrg - newbalanceOrig
        )


        # ---------------- Result card ----------------

        st.write("")

        html(
            f"""
<div class="result"
     style="--tone:{tone};
            --tint:{hex_to_rgba(tone, 0.10)};
            --tint-strong:{hex_to_rgba(tone, 0.16)}">

    <div class="result-gauge">
        {gauge_svg(probability, tone)}
    </div>

    <div class="result-body">

        <span class="badge">
            <span class="dot"></span>
            {badge_text}
        </span>

        <div class="result-title">
            {headline}
        </div>

        <div class="result-text">
            {advice}
        </div>

        <div class="chips">

            <div class="chip">
                <div class="chip-label">
                    Type
                </div>

                <div class="chip-value">
                    {transaction_type.replace("_", " ").title()}
                </div>
            </div>


            <div class="chip">
                <div class="chip-label">
                    Amount
                </div>

                <div class="chip-value">
                    {amount:,.2f}
                </div>
            </div>


            <div class="chip">
                <div class="chip-label">
                    Amount vs previous
                </div>

                <div class="chip-value">
                    {amount_ratio:,.2f}x
                </div>
            </div>


            <div class="chip">
                <div class="chip-label">
                    Origin balance change
                </div>

                <div class="chip-value">
                    {balance_change:,.2f}
                </div>
            </div>

        </div>

    </div>

</div>
"""
        )


        # ---------------- Explanation ----------------

        section(
            "Why this score",
            "Bars show how strongly each factor moved the prediction. "
            "Values in grey are the inputs the model saw.",
        )

        max_abs = (
            float(shap_df["SHAP"].abs().max())
            or 1e-9
        )

        col1, col2 = st.columns(
            2,
            gap="medium",
        )

        with col1:

            with st.container(border=True):

                html(
                    '<div class="card-title">'
                    'Pushing toward fraud'
                    '</div>'
                    '<div class="card-sub">&nbsp;</div>'
                )

                html(
                    contribution_rows(
                        positive,
                        RISK_COLORS["HIGH"],
                        max_abs,
                    )
                )


        with col2:

            with st.container(border=True):

                html(
                    '<div class="card-title">'
                    'Pushing away from fraud'
                    '</div>'
                    '<div class="card-sub">&nbsp;</div>'
                )

                html(
                    contribution_rows(
                        negative,
                        RISK_COLORS["LOW"],
                        max_abs,
                    )
                )


        st.write("")


        with st.expander("Technical model explanation"):

            plot_df = (
                shap_df
                .assign(
                    Absolute=lambda x: x["SHAP"].abs()
                )
                .sort_values(
                    "Absolute",
                    ascending=False,
                )
                .head(10)
                .sort_values("SHAP")
            )

            fig, ax = new_fig(
                8.4,
                4.6,
            )

            bar_colors = [
                RISK_COLORS["HIGH"]
                if v > 0
                else RISK_COLORS["LOW"]
                for v in plot_df["SHAP"]
            ]

            ax.barh(
                plot_df["Feature"],
                plot_df["SHAP"],
                color=bar_colors,
                height=0.6,
            )

            ax.axvline(
                0,
                color="#3A4455",
                linewidth=1,
            )

            ax.grid(
                axis="x",
                visible=True,
            )

            ax.grid(
                axis="y",
                visible=False,
            )

            ax.set_xlabel(
                "SHAP contribution"
            )

            ax.set_title(
                "Top 10 feature contributions"
            )

            show_fig(fig)

            st.caption(
                "Positive values push the prediction toward fraud and negative "
                "values push it away. SHAP values are model contributions, "
                "not probability percentages."
            )


        with st.expander("Behavioral profile"):

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Previous transactions",
                previous_transaction_count,
            )

            c2.metric(
                "Previous amount",
                f"{previous_amount:,.2f}",
            )

            c3.metric(
                "Amount vs previous",
                f"{amount_ratio:.2f}x",
            )

            c4.metric(
                "Origin balance change",
                f"{balance_change:,.2f}",
            )


# ============================================================
# BATCH ANALYTICS
# ============================================================

with batch_tab:

    section(
        "Batch analytics",
        "Upload multiple transactions to assess risk and explore fraud patterns.",
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
    )

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
    ]

    if uploaded_file is None:
        pass

    else:

        batch_df = pd.read_csv(
            uploaded_file
        )

        missing = [
            c
            for c in required_columns
            if c not in batch_df.columns
        ]

        if missing:

            st.error(
                "Missing required columns: "
                + ", ".join(missing)
            )

        else:

            batch_input = create_batch_features(
                batch_df
            )

            probabilities = model.predict_proba(
                batch_input
            )[:, 1]

            batch_df["Fraud Probability"] = probabilities

            batch_df["Risk Level"] = [
                get_risk(p)
                for p in probabilities
            ]

            batch_df["Predicted Fraud"] = (
                probabilities >= 0.50
            ).astype(int)


            # ---------------- Overview ----------------

            total = len(batch_df)

            high = int(
                (
                    batch_df["Risk Level"]
                    == "HIGH"
                ).sum()
            )

            medium = int(
                (
                    batch_df["Risk Level"]
                    == "MEDIUM"
                ).sum()
            )

            low = int(
                (
                    batch_df["Risk Level"]
                    == "LOW"
                ).sum()
            )

            average_probability = (
                probabilities.mean()
            )


            section(
                "Risk overview"
            )

            html(
                '<div class="kpis">'
                + kpi(
                    "Transactions",
                    f"{total:,}",
                )
                + kpi(
                    "High risk",
                    f"{high:,}",
                    RISK_COLORS["HIGH"],
                )
                + kpi(
                    "Medium risk",
                    f"{medium:,}",
                    RISK_COLORS["MEDIUM"],
                )
                + kpi(
                    "Low risk",
                    f"{low:,}",
                    RISK_COLORS["LOW"],
                )
                + kpi(
                    "Average probability",
                    f"{average_probability:.2%}",
                    NEUTRAL,
                )
                + "</div>"
            )


            # ---------------- Charts, row 1 ----------------

            section(
                "Distribution"
            )

            c1, c2 = st.columns(
                2,
                gap="medium",
            )


            with c1:

                with st.container(border=True):

                    risk_counts = (
                        batch_df["Risk Level"]
                        .value_counts()
                        .reindex(
                            [
                                "HIGH",
                                "MEDIUM",
                                "LOW",
                            ],
                            fill_value=0,
                        )
                    )

                    fig, ax = new_fig()

                    bars = ax.bar(
                        [
                            i.title()
                            for i in risk_counts.index
                        ],
                        risk_counts.values,
                        color=[
                            RISK_COLORS[i]
                            for i in risk_counts.index
                        ],
                        width=0.5,
                    )

                    ax.grid(
                        axis="y",
                        visible=True,
                    )

                    ax.set_ylabel(
                        "Transactions"
                    )

                    ax.set_title(
                        "Transactions by risk level"
                    )

                    ax.bar_label(
                        bars,
                        fmt="%d",
                        padding=5,
                        color=TEXT,
                        fontsize=9,
                    )

                    ax.margins(
                        y=0.15
                    )

                    show_fig(fig)


            with c2:

                with st.container(border=True):

                    type_summary = (
                        batch_df
                        .groupby("type")[
                            "Fraud Probability"
                        ]
                        .mean()
                        .sort_values(
                            ascending=False
                        )
                    )

                    fig, ax = new_fig()

                    ax.bar(
                        type_summary.index,
                        type_summary.values,
                        color=NEUTRAL,
                        width=0.5,
                    )

                    ax.set_ylim(
                        0,
                        1,
                    )

                    ax.yaxis.set_major_formatter(
                        PercentFormatter(
                            1.0,
                            decimals=0,
                        )
                    )

                    ax.grid(
                        axis="y",
                        visible=True,
                    )

                    ax.set_ylabel(
                        "Average fraud probability"
                    )

                    ax.set_title(
                        "Average probability by transaction type"
                    )

                    show_fig(fig)


            # ---------------- Charts, row 2 ----------------

            c1, c2 = st.columns(
                2,
                gap="medium",
            )


            with c1:

                with st.container(border=True):

                    fig, ax = new_fig()

                    ax.hist(
                        probabilities,
                        bins=20,
                        range=(0, 1),
                        color=ACCENT,
                        alpha=0.9,
                        rwidth=0.9,
                    )

                    for threshold in (
                        0.30,
                        0.70,
                    ):
                        ax.axvline(
                            threshold,
                            color="#3A4455",
                            linestyle="--",
                            linewidth=1,
                        )

                    ax.xaxis.set_major_formatter(
                        PercentFormatter(
                            1.0,
                            decimals=0,
                        )
                    )

                    ax.grid(
                        axis="y",
                        visible=True,
                    )

                    ax.set_xlabel(
                        "Fraud probability"
                    )

                    ax.set_ylabel(
                        "Transactions"
                    )

                    ax.set_title(
                        "Fraud probability distribution"
                    )

                    show_fig(fig)


            with c2:

                with st.container(border=True):

                    fig, ax = new_fig()

                    point_colors = (
                        batch_df["Risk Level"]
                        .map(RISK_COLORS)
                    )

                    ax.scatter(
                        batch_df["amount"],
                        probabilities,
                        c=point_colors,
                        s=22,
                        alpha=0.75,
                        linewidths=0,
                    )

                    ax.set_ylim(
                        0,
                        1,
                    )

                    ax.yaxis.set_major_formatter(
                        PercentFormatter(
                            1.0,
                            decimals=0,
                        )
                    )

                    ax.xaxis.set_major_formatter(
                        FuncFormatter(
                            compact_number
                        )
                    )

                    ax.grid(
                        axis="y",
                        visible=True,
                    )

                    ax.set_xlabel(
                        "Transaction amount"
                    )

                    ax.set_ylabel(
                        "Fraud probability"
                    )

                    ax.set_title(
                        "Amount vs fraud probability"
                    )

                    show_fig(fig)


            # ---------------- Actual vs predicted ----------------

            if "isFraud" in batch_df.columns:

                section(
                    "Model evaluation",
                    "Predictions compared with the isFraud column, using a 50% threshold.",
                )

                actual = batch_df["isFraud"]
                predicted = batch_df["Predicted Fraud"]

                m1, m2, m3, m4 = st.columns(4)

                m1.metric(
                    "Accuracy",
                    f"{accuracy_score(actual, predicted):.2%}",
                )

                m2.metric(
                    "Precision",
                    f"{precision_score(actual, predicted, zero_division=0):.2%}",
                )

                m3.metric(
                    "Recall",
                    f"{recall_score(actual, predicted, zero_division=0):.2%}",
                )

                m4.metric(
                    "F1 score",
                    f"{f1_score(actual, predicted, zero_division=0):.2%}",
                )

                cm = confusion_matrix(
                    actual,
                    predicted,
                )

                st.write("")

                cm_col, _ = st.columns(
                    [1, 1.4]
                )

                with cm_col:

                    with st.container(border=True):

                        fig, ax = new_fig(
                            4.6,
                            3.8,
                        )

                        cmap = LinearSegmentedColormap.from_list(
                            "fg",
                            [
                                SURFACE_2,
                                ACCENT,
                            ],
                        )

                        ax.imshow(
                            cm,
                            cmap=cmap,
                        )

                        ax.grid(False)

                        ax.set_xticks(
                            range(cm.shape[1])
                        )

                        ax.set_yticks(
                            range(cm.shape[0])
                        )

                        ax.set_xticklabels(
                            [
                                "Legitimate",
                                "Fraud",
                            ][: cm.shape[1]]
                        )

                        ax.set_yticklabels(
                            [
                                "Legitimate",
                                "Fraud",
                            ][: cm.shape[0]]
                        )

                        ax.set_xlabel(
                            "Predicted"
                        )

                        ax.set_ylabel(
                            "Actual"
                        )

                        ax.set_title(
                            "Confusion matrix"
                        )

                        for i in range(
                            cm.shape[0]
                        ):
                            for j in range(
                                cm.shape[1]
                            ):
                                ax.text(
                                    j,
                                    i,
                                    f"{cm[i, j]:,}",
                                    ha="center",
                                    va="center",
                                    color="#FFFFFF",
                                    fontsize=12,
                                    fontweight="semibold",
                                )

                        show_fig(fig)


            # ---------------- High-risk transactions ----------------

            section(
                "High-risk transactions",
                "Sorted by fraud probability, highest first.",
            )

            high_risk_df = (
                batch_df[
                    batch_df["Risk Level"]
                    == "HIGH"
                ]
                .sort_values(
                    "Fraud Probability",
                    ascending=False,
                )
            )

            if len(high_risk_df) > 0:

                st.dataframe(
                    prediction_table(
                        high_risk_df
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config=PROBABILITY_COLUMN,
                )

            else:

                html(
                    '<div class="note">'
                    'No high-risk transactions detected in this file.'
                    '</div>'
                )


            st.write("")


            with st.expander(
                "View complete prediction results"
            ):

                st.dataframe(
                    prediction_table(
                        batch_df
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config=PROBABILITY_COLUMN,
                )


            st.write("")


            csv = batch_df.to_csv(
                index=False
            )

            st.download_button(
                "Download prediction results",
                csv,
                "fraudguard_predictions.csv",
                "text/csv",
            )