import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

from core.data_loader import load_csv, get_basic_info
from core.data_profiler import profile_dataset, detect_time_series
from core.visualizer import plot_correlation_heatmap, plot_boxplots
from core.cleaning_guide import get_cleaning_guidance
from core.train_test_guide import get_train_test_guidance
from core.auto_cleaner import auto_clean_dataframe
from core.custom_visualizer import generate_custom_plot
from core.trainer import run_automl_pipeline
from core.explainability import compute_shap_explanations

from llm_engine.prompts import problem_understanding_prompt
from llm_engine.llm_client import call_llm
from llm_engine.response_parser import parse_llm_response

from ui.style import apply_global_style

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI-Guided ML Mentor & AutoML Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_global_style()

# ===============================
# SESSION STATE
# ===============================
st.session_state.setdefault("df", None)
st.session_state.setdefault("target_col", None)
st.session_state.setdefault("problem_info", None)
st.session_state.setdefault("fullscreen_fig", None)
st.session_state.setdefault("automl_results", None)
st.session_state.setdefault("shap_fig", None)
st.session_state.setdefault("shap_table", None)

# ===============================
# HELPERS
# ===============================
def rotate_axis_labels(fig):
    for ax in fig.axes:
        ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()


def render_small(fig, width=440):
    rotate_axis_labels(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, width=width)


def render_fullscreen(fig):
    rotate_axis_labels(fig)
    fig.set_size_inches(12, 9)
    st.pyplot(fig)
    if st.button("❌ Close Fullscreen"):
        st.session_state.fullscreen_fig = None
        st.rerun()


def llm_plot_explanation(plot_type, features):
    prompt = f"""
You are a senior data scientist.
Plot type: {plot_type}
Features involved: {features}

Return EXACTLY two lines:
Line 1: How to read this plot (1 sentence)
Line 2: What insight this plot provides (1 sentence)

Rules: No bullets, plain text only.
"""
    return call_llm(prompt=prompt)


# ===============================
# FULLSCREEN MODE
# ===============================
if st.session_state.fullscreen_fig is not None:
    st.markdown("## 🔍 Fullscreen Graph View")
    render_fullscreen(st.session_state.fullscreen_fig)
    st.stop()

# ===============================
# HERO BANNER
# ===============================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🧠 AI-Guided Machine Learning Mentor & AutoML Suite</div>
        <div class="hero-subtitle">
            An end-to-end intelligent ML engineering workbench: automated exploratory data analysis, data leakage prevention, Bayesian hyperparameter optimization with Optuna, and TreeSHAP explainability.
        </div>
        <div class="badge-container">
            <span class="badge badge-blue">⚡ Optuna Bayesian Optimization</span>
            <span class="badge badge-purple">🔍 TreeSHAP Explainability</span>
            <span class="badge badge-emerald">🛡️ Data Leakage Safeguards</span>
            <span class="badge badge-blue">🤖 Fail-Safe AI Mentor</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# SIDEBAR CONTROLS
# ===============================
with st.sidebar:
    st.markdown("### ⚙️ Quick Navigation & Data")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], help="Upload any tabular CSV file to begin.")

    if uploaded_file:
        st.session_state.df = load_csv(uploaded_file)
        st.success(f"Loaded: `{uploaded_file.name}`")
        st.write(f"- Rows: **{st.session_state.df.shape[0]:,}**")
        st.write(f"- Columns: **{st.session_state.df.shape[1]}**")

        st.markdown("---")
        st.markdown("### 🎯 Target Selection")
        col_options = ["-- Select Target Column --"] + list(st.session_state.df.columns)
        def_idx = 0
        if st.session_state.target_col in col_options:
            def_idx = col_options.index(st.session_state.target_col)
        
        target_choice = st.selectbox("Predict Target (Y):", col_options, index=def_idx)
        if target_choice != "-- Select Target Column --":
            st.session_state.target_col = target_choice
        else:
            st.session_state.target_col = None

    st.markdown("---")
    st.markdown("### 💡 Project Mentor Status")
    if st.session_state.df is None:
        st.info("Upload a dataset to activate the mentor pipeline.")
    else:
        st.caption("✅ Dataset Ingested")
        st.caption(f"🎯 Target: `{st.session_state.target_col or 'Not set'}`")
        if st.session_state.automl_results:
            st.caption(f"🏆 Best Model Trained in {st.session_state.automl_results['elapsed_time']}s")

# ===============================
# MAIN CONTENT WORKFLOW (TABS)
# ===============================
if st.session_state.df is None:
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 3rem; margin-bottom: 12px;">📁</div>
            <h3 style="margin-bottom: 8px;">No Dataset Uploaded Yet</h3>
            <p style="color: #94A3B8; max-width: 500px; margin: 0 auto;">
                Upload a CSV file using the sidebar on the left to begin data exploration, leakage prevention checks, and AutoML optimization.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# Detect time series
time_info = detect_time_series(st.session_state.df, st.session_state)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Data & Health",
    "📊 EDA & Visuals",
    "🧹 Preprocessing & Split",
    "🚀 AutoML Benchmark",
    "🧠 Model Explainability"
])

# -------------------------------------------------------------
# TAB 1: DATA & HEALTH
# -------------------------------------------------------------
with tab1:
    st.markdown("### 📋 Dataset Preview & Quick Stats")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-badge'><div class='metric-val'>{st.session_state.df.shape[0]:,}</div><div class='metric-label'>Total Rows</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-badge'><div class='metric-val'>{st.session_state.df.shape[1]}</div><div class='metric-label'>Total Columns</div></div>", unsafe_allow_html=True)
    with c3:
        null_count = int(st.session_state.df.isnull().sum().sum())
        st.markdown(f"<div class='metric-badge'><div class='metric-val'>{null_count:,}</div><div class='metric-label'>Missing Values</div></div>", unsafe_allow_html=True)
    with c4:
        numeric_count = len(st.session_state.df.select_dtypes(include=['number']).columns)
        st.markdown(f"<div class='metric-badge'><div class='metric-val'>{numeric_count}</div><div class='metric-label'>Numeric Features</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.dataframe(st.session_state.df.head(10), use_container_width=True)

    st.markdown("### 🩺 Dataset Health & Outlier Profile")
    st.dataframe(profile_dataset(st.session_state.df), use_container_width=True)

    st.markdown("### 🎯 Problem Goal & Task Formulation")
    goal_input = st.text_input("State your project goal (optional, e.g. 'Predict churn based on activity'):", placeholder="e.g. Predict whether a transaction is fraudulent")
    if goal_input:
        info = get_basic_info(st.session_state.df)
        raw = call_llm(problem_understanding_prompt(goal_input, info), fallback_context=goal_input)
        st.session_state.problem_info = parse_llm_response(raw)
        st.info(f"**Mentor Task Deduction:** {st.session_state.problem_info.get('task_type', 'classification').upper()} | {st.session_state.problem_info.get('reasoning')}")

# -------------------------------------------------------------
# TAB 2: EDA & VISUALIZATIONS
# -------------------------------------------------------------
with tab2:
    st.markdown("### 📈 Automated Exploratory Visualizations")
    st.caption("Inspect distributions, correlations, and feature variability with AI commentary.")

    visuals = []
    heatmap = plot_correlation_heatmap(st.session_state.df)
    if heatmap:
        visuals.append(("Correlation Heatmap", "sns.heatmap(df.corr(), annot=True)", "numeric correlations", heatmap))

    for fig in plot_boxplots(st.session_state.df):
        visuals.append(("Boxplot Distribution", "sns.boxplot(x=df[column])", "single numeric feature", fig))

    if visuals:
        idx = st.slider("Select visualization index", 0, len(visuals) - 1, 0)
        title, code, feat_ctx, fig = visuals[idx]
        explanation = llm_plot_explanation(title, feat_ctx)
        lines = explanation.splitlines() if explanation else ["", ""]

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown(f"#### {title}")
            st.code(code, language="python")
            st.markdown("**📖 How to read this chart:**")
            st.write(lines[0] if len(lines) > 0 else "Examine the distribution and density spread.")
            st.markdown("**💡 Key engineering takeaway:**")
            st.write(lines[1] if len(lines) > 1 else "Detect potential skewness or collinearity.")
            if st.button("🔍 Expand Fullscreen", key=f"fs_auto_{idx}"):
                st.session_state.fullscreen_fig = fig
                st.rerun()

        with col_right:
            render_small(fig)

    st.markdown("---")
    st.markdown("### 🧪 Hypothesis-Driven Custom Plotter")
    col_p1, col_p2, col_p3 = st.columns([1, 1, 1])
    with col_p1:
        plot_type = st.selectbox("Graph Type", ["Histogram", "Boxplot", "Scatter Plot", "Line Plot", "Count Plot", "Correlation Heatmap"])
    cols = list(st.session_state.df.columns)
    with col_p2:
        if plot_type in ["Histogram", "Boxplot", "Count Plot"]:
            custom_feats = [st.selectbox("Feature", cols)]
        else:
            custom_feats = [st.selectbox("X-axis", cols), st.selectbox("Y-axis", cols, index=min(1, len(cols)-1))]

    custom_fig = generate_custom_plot(st.session_state.df, plot_type, custom_feats)
    col_cl, col_cr = st.columns([1, 1])
    with col_cl:
        st.markdown(f"#### Custom {plot_type}")
        st.code(f"generate_custom_plot(df, '{plot_type}', {custom_feats})", language="python")
        if st.button("🔍 Expand Fullscreen", key="fs_custom"):
            st.session_state.fullscreen_fig = custom_fig
            st.rerun()
    with col_cr:
        render_small(custom_fig)

# -------------------------------------------------------------
# TAB 3: PREPROCESSING & SPLIT
# -------------------------------------------------------------
with tab3:
    st.markdown("### 🛡️ Data Leakage & Temporal Checks")
    if time_info.get("is_time_series"):
        st.warning(f"⚠️ **Temporal ordering detected** in columns: `{time_info.get('datetime_columns')}`. A sequential time-series split is mandatory to avoid catastrophic future data leakage.")
    else:
        st.success("✅ No temporal ordering detected. Standard stratified or randomized cross-validation is safe.")

    st.markdown("### 🧹 Dataset-Aware Preprocessing Guidance")
    st.caption("Actionable, concrete code snippets dynamically tailored to your uploaded features.")

    auto_clean = st.checkbox("Download Cleaned Dataset (Auto-Impute, One-Hot & Scale)")
    if auto_clean:
        cleaned_df = auto_clean_dataframe(st.session_state.df)
        st.download_button("📥 Download Cleaned CSV", cleaned_df.to_csv(index=False), "cleaned_dataset.csv", "text/csv")
    else:
        guidance_steps = get_cleaning_guidance(st.session_state.df)
        for step in guidance_steps:
            with st.expander(f"🔹 {step['title']}", expanded=True):
                st.write(step["reason"])
                st.code(step["code"], language="python")

    st.markdown("### ✂️ Recommended Train-Test Split Architecture")
    task_for_split = st.session_state.problem_info.get("task_type", "classification") if st.session_state.problem_info else "classification"
    for s in get_train_test_guidance(task_for_split, time_info.get("is_time_series", False)):
        with st.expander(f"📌 {s['title']}", expanded=True):
            st.write(s["why"])
            st.code(s["code"], language="python")

# -------------------------------------------------------------
# TAB 4: AUTOML BENCHMARK
# -------------------------------------------------------------
with tab4:
    st.markdown("### 🚀 AutoML Benchmark Engine (Optuna Bayesian Optimization)")
    st.caption("Benchmarks an interpretable Baseline against a Bayesian-tuned ensemble, capped for fast CPU performance.")

    if not st.session_state.target_col:
        st.warning("👉 Please select a **Target Variable (Y)** in the sidebar on the left before launching AutoML.")
    else:
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            timeout_budget = st.slider("Time Budget (Seconds)", min_value=15, max_value=60, value=30, step=5)
        with ctrl2:
            trials_budget = st.slider("Max Optuna Trials", min_value=5, max_value=25, value=15, step=5)
        with ctrl3:
            inferred = "classification"
            if st.session_state.problem_info:
                inferred = st.session_state.problem_info.get("task_type", "classification")
            elif pd.api.types.is_numeric_dtype(st.session_state.df[st.session_state.target_col]) and st.session_state.df[st.session_state.target_col].nunique() > 10:
                inferred = "regression"
            task_choice = st.selectbox("Objective Type", ["classification", "regression"], index=0 if inferred == "classification" else 1)

        if st.button("🔥 Launch AutoML & Optimization", type="primary", use_container_width=True):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.text(msg)

            with st.spinner("Executing Baseline & Bayesian Optimization..."):
                results = run_automl_pipeline(
                    df=st.session_state.df,
                    target_col=st.session_state.target_col,
                    task_type=task_choice,
                    is_time_series=time_info.get("is_time_series", False),
                    max_trials=trials_budget,
                    timeout_seconds=timeout_budget,
                    progress_callback=update_progress
                )
                st.session_state.automl_results = results

                status_text.text("Extracting TreeSHAP explainability values...")
                shap_fig, shap_table = compute_shap_explanations(results["best_model"], results["X_test"])
                st.session_state.shap_fig = shap_fig
                st.session_state.shap_table = shap_table

            progress_bar.progress(1.0)
            status_text.text("Optimization and Explainability completed successfully!")
            st.rerun()

        if st.session_state.automl_results is not None:
            res = st.session_state.automl_results
            st.success(f"✨ Training complete in **{res['elapsed_time']}s** across **{res['trials_completed']} Optuna trials**.")
            if res["is_downsampled"]:
                st.info("ℹ️ Dataset was safely stratified/sampled to 10,000 rows to ensure zero memory exhaustion on free hosting.")

            st.markdown("#### 🏆 Performance Leaderboard")
            col_b, col_t = st.columns(2)
            with col_b:
                st.markdown("**Baseline Model (Linear/Logistic)**")
                st.json(res["baseline_metrics"])
            with col_t:
                st.markdown("**Optuna-Tuned Ensemble (Winner)**")
                st.json(res["tuned_metrics"])

            st.markdown("#### ⚙️ Optimal Hyperparameters Found")
            st.json(res["best_params"])

# -------------------------------------------------------------
# TAB 5: MODEL EXPLAINABILITY
# -------------------------------------------------------------
with tab5:
    st.markdown("### 🧠 Model Explainability (TreeSHAP)")
    st.caption("Inspect exact Shapley feature attributions and understand model decisions.")

    if st.session_state.shap_fig is None:
        st.info("Run the AutoML optimization in Tab 4 to generate live TreeSHAP explainability charts.")
    else:
        st.markdown("#### 🐝 Global Feature Attribution (Beeswarm Plot)")
        st.pyplot(st.session_state.shap_fig)

        if st.session_state.shap_table is not None:
            st.markdown("#### 📊 Ranked Feature Impact Leaderboard")
            st.dataframe(st.session_state.shap_table, use_container_width=True)
