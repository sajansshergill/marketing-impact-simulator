"""
Marketing Mix Model (MMM) & Brand Strategy Simulator - Full Dashboard
============================================================================
A comprehensive dashboard for MMM analysis, simulation, and strategic planning.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on path so `src` can be imported
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.preprocessing import TransformConfig, build_features
from src.mmm_model import MMMSpec, fit_ols_mmm, predict, contribution_decomposition
from src.simulation_engine import Scenario, simulate_revenue_uplift

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="MMM & Brand Strategy Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        padding: 0.5rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .divider {
        margin: 2rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING & MODEL TRAINING
# =============================================================================
@st.cache_data(ttl=3600)
def load_data_and_train():
    """Load data and train model - cached for performance"""
    csv_path = "data/processed/mmm_synth.csv"
    
    if not os.path.exists(csv_path):
        return None, None, None, None, None
    
    df_raw = pd.read_csv(csv_path, parse_dates=["week"])
    tcfg = TransformConfig()
    df = build_features(df_raw, tcfg)
    
    spec = MMMSpec()
    model, x_cols = fit_ols_mmm(df, spec)
    df["pred"] = predict(model, df, x_cols)
    
    # Calculate residuals
    df["residuals"] = df["revenue"] - df["pred"]
    df["residuals_pct"] = (df["residuals"] / df["revenue"]) * 100
    
    return df_raw, df, model, x_cols, tcfg


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
def render_sidebar():
    """Render sidebar navigation"""
    st.sidebar.image("https://img.icons8.com/color/96/000000/marketing.png", width=80)
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Go to",
        ["📈 Overview", "💰 Revenue Analysis", "📺 Channel Performance", 
         "🧪 Simulation Lab", "📋 Model Details", "📥 Data Export"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings")
    
    # Model refresh button
    if st.sidebar.button("🔄 Refresh Model", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Data info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Range:**")
    
    return page


# =============================================================================
# PAGE: OVERVIEW
# =============================================================================
def render_overview(df_raw, df, model, x_cols):
    """Main overview page with KPIs and summary"""
    
    st.markdown('<p class="main-header">📊 Marketing Mix Model & Brand Strategy Simulator</p>', 
                unsafe_allow_html=True)
    
    st.caption("Econometric MMM + Strategy Counterfactuals | Focus, Leadership, Budget Allocation")
    
    # Date range info
    date_range = f"{df_raw['week'].min().strftime('%Y-%m-%d')} to {df_raw['week'].max().strftime('%Y-%m-%d')}"
    st.markdown(f"**Data Period:** {date_range} ({len(df_raw)} weeks)")
    
    st.markdown('<p class="divider"></p>', unsafe_allow_html=True)
    
    # KPI Row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    total_revenue = df["revenue"].sum()
    avg_weekly = df["revenue"].mean()
    total_spend = df[["tv_spend", "search_spend", "social_spend", "display_spend"]].sum().sum()
    r_squared = model.rsquared
    mae = df["residuals"].abs().mean()
    
    with kpi1:
        st.metric("Total Revenue", f"${total_revenue/1e6:.1f}M", delta=None)
    with kpi2:
        st.metric("Avg Weekly Revenue", f"${avg_weekly/1e3:.0f}K", delta=None)
    with kpi3:
        st.metric("Total Ad Spend", f"${total_spend/1e6:.1f}M", delta=None)
    with kpi4:
        st.metric("R² Score", f"{r_squared:.3f}", delta_color="normal")
    with kpi5:
        st.metric("MAE", f"${mae:,.0f}", delta_color="inverse")
    
    # Charts Row
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.subheader("📈 Revenue: Actual vs Predicted")
        
        # Interactive plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["week"], y=df["revenue"],
            mode="lines", name="Actual Revenue",
            line=dict(color="#1f77b4", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df["week"], y=df["pred"],
            mode="lines", name="Predicted Revenue",
            line=dict(color="#ff7f0e", width=2, dash="dash")
        ))
        
        fig.update_layout(
            xaxis_title="Week",
            yaxis_title="Revenue ($)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.subheader("📊 Revenue Distribution")
        
        # Histogram
        fig_hist = px.histogram(
            df, x="revenue", nbins=30,
            marginal="box",
            color_discrete_sequence=["#1f77b4"]
        )
        fig_hist.update_layout(
            xaxis_title="Revenue",
            yaxis_title="Frequency",
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Model Performance Section
    st.markdown('<p class="divider"></p>', unsafe_allow_html=True)
    st.subheader("🎯 Model Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    rmse = np.sqrt((df["residuals"] ** 2).mean())
    mape = df["residuals_pct"].abs().mean()
    
    with perf_col1:
        st.metric("RMSE", f"${rmse:,.0f}")
    with perf_col2:
        st.metric("MAPE", f"{mape:.2f}%")
    with perf_col3:
        st.metric("Max Error", f"${df['residuals'].abs().max():,.0f}")
    with perf_col4:
        st.metric("Min Error", f"${df['residuals'].abs().min():,.0f}")
    
    # Residuals over time
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=df["week"], y=df["residuals"],
        mode="lines+markers",
        name="Residuals",
        line=dict(color="#d62728", width=1),
        marker=dict(size=4)
    ))
    fig_resid.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_resid.add_hline(y=mae*2, line_dash="dot", line_color="orange", annotation_text="±2 MAE")
    fig_resid.add_hline(y=-mae*2, line_dash="dot", line_color="orange", annotation_text="±2 MAE")
    
    fig_resid.update_layout(
        title="Residuals Over Time",
        xaxis_title="Week",
        yaxis_title="Residual ($)",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_resid, use_container_width=True)


# =============================================================================
# PAGE: REVENUE ANALYSIS
# =============================================================================
def render_revenue_analysis(df_raw, df, model, x_cols):
    """Detailed revenue analysis page"""
    
    st.markdown('<p class="main-header">💰 Revenue Analysis</p>', unsafe_allow_html=True)
    
    # Revenue components
    contrib = contribution_decomposition(model, df, x_cols)
    
    # Summary stats
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Total Revenue", f"${df['revenue'].sum()/1e6:.2f}M")
    with stat_col2:
        st.metric("Predicted Revenue", f"${contrib['predicted'].sum()/1e6:.2f}M")
    with stat_col3:
        st.metric("Actual - Predicted", f"${(df['revenue'].sum() - contrib['predicted'].sum())/1e3:.1f}K")
    with stat_col4:
        st.metric("R²", f"{model.rsquared:.4f}")
    
    st.markdown("---")
    
    # Contribution breakdown chart
    st.subheader("📊 Revenue Contribution by Factor")
    
    # Get average contribution per week
    contrib_avg = contrib.drop(columns=["predicted"]).mean()
    contrib_avg = contrib_avg.sort_values(ascending=True)
    
    fig_contrib = go.Figure(go.Bar(
        x=contrib_avg.values,
        y=contrib_avg.index,
        orientation="h",
        marker_color=px.colors.qualitative.Set2
    ))
    fig_contrib.update_layout(
        title="Average Weekly Revenue Contribution by Factor",
        xaxis_title="Average Contribution ($)",
        yaxis_title="Factor",
        height=400,
        margin=dict(l=150, r=40, t=60, b=40)
    )
    st.plotly_chart(fig_contrib, use_container_width=True)
    
    # Trend analysis
    st.markdown("---")
    st.subheader("📈 Revenue Trends & Seasonality")
    
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        # Seasonality effect
        fig_seas = go.Figure()
        fig_seas.add_trace(go.Scatter(
            x=df["week"], y=df["seasonality"],
            mode="lines", name="Seasonality",
            line=dict(color="#2ca02c", width=2),
            fill="tozeroy", fillcolor="rgba(44, 160, 44, 0.2)"
        ))
        fig_seas.update_layout(
            title="Seasonality Pattern",
            xaxis_title="Week",
            yaxis_title="Seasonality Index",
            height=350
        )
        st.plotly_chart(fig_seas, use_container_width=True)
    
    with trend_col2:
        # Holiday impact
        fig_hol = go.Figure()
        fig_hol.add_trace(go.Bar(
            x=df["week"], y=df["holiday_spike"],
            name="Holiday Spike",
            marker_color="#ff7f0e"
        ))
        fig_hol.update_layout(
            title="Holiday/Promo Impact",
            xaxis_title="Week",
            yaxis_title="Holiday Multiplier",
            height=350
        )
        st.plotly_chart(fig_hol, use_container_width=True)
    
    # Brand awareness impact
    st.markdown("---")
    st.subheader("🏷️ Brand Awareness Impact")
    
    fig_brand = px.scatter(
        df, x="brand_awareness", y="revenue",
        trendline="ols",
        title="Brand Awareness vs Revenue",
        color_discrete_sequence=["#1f77b4"]
    )
    fig_brand.update_layout(
        xaxis_title="Brand Awareness Index",
        yaxis_title="Revenue ($)",
        height=400
    )
    st.plotly_chart(fig_brand, use_container_width=True)


# =============================================================================
# PAGE: CHANNEL PERFORMANCE
# =============================================================================
def render_channel_performance(df_raw, df, model, x_cols):
    """Channel performance analysis"""
    
    st.markdown('<p class="main-header">📺 Channel Performance Analysis</p>', unsafe_allow_html=True)
    
    channels = ["tv_spend", "search_spend", "social_spend", "display_spend"]
    channel_names = {"tv_spend": "TV", "search_spend": "Search", 
                     "social_spend": "Social", "display_spend": "Display"}
    
    # Channel summary
    st.subheader("💵 Channel Spend Summary")
    
    spend_data = []
    for ch in channels:
        spend_data.append({
            "Channel": channel_names[ch],
            "Total Spend": df[ch].sum(),
            "Avg Weekly": df[ch].mean(),
            "Min Weekly": df[ch].min(),
            "Max Weekly": df[ch].max(),
            "Std Dev": df[ch].std()
        })
    
    spend_df = pd.DataFrame(spend_data)
    st.dataframe(
        spend_df.style.format({
            "Total Spend": "${:,.0f}",
            "Avg Weekly": "${:,.0f}",
            "Min Weekly": "${:,.0f}",
            "Max Weekly": "${:,.0f}",
            "Std Dev": "${:,.0f}"
        }),
        use_container_width=True
    )
    
    # Spend pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=[channel_names[ch] for ch in channels],
        values=[df[ch].sum() for ch in channels],
        hole=0.4,
        marker_colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    )])
    fig_pie.update_layout(title="Budget Allocation by Channel", height=400)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Spend over time
    st.markdown("---")
    st.subheader("📈 Channel Spend Over Time")
    
    fig_spend = go.Figure()
    for ch in channels:
        fig_spend.add_trace(go.Scatter(
            x=df["week"], y=df[ch],
            mode="lines", name=channel_names[ch],
            stackgroup="spend"
        ))
    fig_spend.update_layout(
        xaxis_title="Week",
        yaxis_title="Spend ($)",
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig_spend, use_container_width=True)
    
    # Channel effectiveness (saturation curves)
    st.markdown("---")
    st.subheader("📉 Channel Saturation Curves")
    
    sat_col1, sat_col2 = st.columns(2)
    
    with sat_col1:
        # TV saturation
        fig_tv = px.scatter(
            df, x="tv_spend", y="revenue",
            trendline="lowess",
            title="TV Spend vs Revenue (LOWESS)",
            color_discrete_sequence=["#1f77b4"]
        )
        fig_tv.update_layout(height=350)
        st.plotly_chart(fig_tv, use_container_width=True)
    
    with sat_col2:
        # Search saturation
        fig_search = px.scatter(
            df, x="search_spend", y="revenue",
            trendline="lowess",
            title="Search Spend vs Revenue (LOWESS)",
            color_discrete_sequence=["#ff7f0e"]
        )
        fig_search.update_layout(height=350)
        st.plotly_chart(fig_search, use_container_width=True)
    
    # ROI analysis
    st.markdown("---")
    st.subheader("💹 Channel ROI Analysis")
    
    contrib = contribution_decomposition(model, df, x_cols)
    
    roi_data = []
    for ch in channels:
        sat_col = f"{ch}_sat"
        if sat_col in contrib.columns:
            revenue_attr = contrib[sat_col].sum()
            spend = df[ch].sum()
            roi = (revenue_attr - spend) / spend * 100 if spend > 0 else 0
            roi_data.append({
                "Channel": channel_names[ch],
                "Total Spend": spend,
                "Attributed Revenue": revenue_attr,
                "ROI (%)": roi
            })
    
    roi_df = pd.DataFrame(roi_data)
    
    fig_roi = px.bar(
        roi_df, x="Channel", y="ROI (%)",
        color="ROI (%)",
        color_continuous_scale="RdYlGn",
        title="Return on Investment by Channel"
    )
    fig_roi.update_layout(height=400)
    st.plotly_chart(fig_roi, use_container_width=True)


# =============================================================================
# PAGE: SIMULATION LAB
# =============================================================================
def render_simulation_lab(df_raw, df, model, x_cols, tcfg):
    """Interactive simulation laboratory"""
    
    st.markdown('<p class="main-header">🧪 Strategy Simulation Lab</p>', unsafe_allow_html=True)
    st.caption("Test counterfactual scenarios: Focus, Leadership, Budget Reallocation")
    
    # Scenario parameters in sidebar
    st.subheader("⚙️ Scenario Configuration")
    
    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 1])
    
    with sim_col1:
        st.markdown("**🎯 Focus Strategy (Law of Focus)**")
        focus_delta = st.slider(
            "Focus Score Change", 
            min_value=-0.30, max_value=0.30, 
            value=0.10, step=0.01,
            help="Adjust the focus score to simulate more or less strategic focus"
        )
        
        st.markdown("**👑 Leadership (Law of Leadership)**")
        leader_mode = st.selectbox(
            "Category Leadership",
            ["No Change", "Force Leader", "Force Challenger"],
            help="Override category leadership status"
        )
        leader_override = None
        if leader_mode == "Force Leader":
            leader_override = 1
        elif leader_mode == "Force Challenger":
            leader_override = 0
    
    with sim_col2:
        st.markdown("**💰 Budget Reallocation**")
        st.caption("Percentage change in channel budget")
        
        tv_shift = st.slider("TV Spend", -0.50, 0.50, 0.00, 0.01)
        search_shift = st.slider("Search Spend", -0.50, 0.50, 0.00, 0.01)
        social_shift = st.slider("Social Spend", -0.50, 0.50, 0.00, 0.01)
        display_shift = st.slider("Display Spend", -0.50, 0.50, 0.00, 0.01)
    
    with sim_col3:
        st.markdown("**📅 Time Period for Simulation**")
        sim_weeks = st.slider(
            "Weeks to Simulate",
            min_value=4, max_value=len(df_raw),
            value=len(df_raw), step=4
        )
        
        st.markdown("**📊 Display Options**")
        show_baseline = st.checkbox("Show Baseline", value=True)
        show_scenario = st.checkbox("Show Scenario", value=True)
        show_uplift = st.checkbox("Show Uplift", value=True)
    
    # Create scenario
    scenario = Scenario(
        focus_delta=focus_delta,
        leader_override=leader_override,
        budget_shift={
            "tv_spend": tv_shift,
            "search_spend": search_shift,
            "social_spend": social_shift,
            "display_spend": display_shift,
        },
    )
    
    # Run simulation
    sim = simulate_revenue_uplift(df_raw, model, x_cols, tcfg, scenario)
    
    # Limit to selected weeks
    if sim_weeks < len(sim):
        sim = sim.tail(sim_weeks)
    
    st.markdown("---")
    
    # Results KPIs
    st.subheader("📊 Simulation Results")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_baseline = sim["baseline_pred"].sum()
    total_scenario = sim["scenario_pred"].sum()
    total_uplift = sim["uplift"].sum()
    uplift_pct = (total_uplift / total_baseline) * 100 if total_baseline > 0 else 0
    
    with kpi1:
        st.metric("Baseline Revenue", f"${total_baseline/1e6:.2f}M")
    with kpi2:
        st.metric("Scenario Revenue", f"${total_scenario/1e6:.2f}M",
                  delta=f"${total_uplift/1e3:.1f}K")
    with kpi3:
        st.metric("Total Uplift", f"${total_uplift/1e3:.1f}K",
                  delta=f"{uplift_pct:.2f}%", delta_color="normal")
    with kpi4:
        st.metric("Avg Weekly Uplift", f"${sim['uplift'].mean()/1e3:.1f}K")
    
    # Visualization
    st.subheader("📈 Revenue Comparison")
    
    chart_data = {}
    if show_baseline:
        chart_data["Baseline"] = sim.set_index("week")["baseline_pred"]
    if show_scenario:
        chart_data["Scenario"] = sim.set_index("week")["scenario_pred"]
    
    if chart_data:
        fig_rev = go.Figure()
        for name, data in chart_data.items():
            fig_rev.add_trace(go.Scatter(
                x=data.index, y=data.values,
                mode="lines", name=name,
                line=dict(width=2)
            ))
        fig_rev.update_layout(
            xaxis_title="Week",
            yaxis_title="Revenue ($)",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_rev, use_container_width=True)
    
    # Uplift visualization
    if show_uplift:
        st.subheader("📈 Uplift Over Time")
        
        fig_uplift = go.Figure()
        fig_uplift.add_trace(go.Bar(
            x=sim["week"], y=sim["uplift"],
            marker_color=sim["uplift"].apply(lambda x: "#2ca02c" if x > 0 else "#d62728"),
            name="Uplift"
        ))
        fig_uplift.add_hline(y=sim["uplift"].mean(), line_dash="dash", 
                            line_color="#1f77b4", annotation_text=f"Mean: ${sim['uplift'].mean():,.0f}")
        fig_uplift.update_layout(
            xaxis_title="Week",
            yaxis_title="Uplift ($)",
            height=350
        )
        st.plotly_chart(fig_uplift, use_container_width=True)
    
    # Scenario summary
    st.markdown("---")
    st.subheader("📋 Scenario Summary")
    
    summary_data = {
        "Parameter": ["Focus Change", "Leadership Override", "TV Shift", "Search Shift", 
                      "Social Shift", "Display Shift"],
        "Value": [f"{focus_delta:+.2f}", 
                  leader_mode if leader_override is not None else "No Change",
                  f"{tv_shift:+.0%}", f"{search_shift:+.0%}", 
                  f"{social_shift:+.0%}", f"{display_shift:+.0%}"]
    }
    st.table(pd.DataFrame(summary_data))
    
    # Download button
    st.markdown("---")
    st.download_button(
        label="📥 Download Simulation Results (CSV)",
        data=sim.to_csv(index=False).encode("utf-8"),
        file_name="simulation_results.csv",
        mime="text/csv",
        use_container_width=True
    )


# =============================================================================
# PAGE: MODEL DETAILS
# =============================================================================
def render_model_details(model, df, x_cols):
    """Model details and coefficients"""
    
    st.markdown('<p class="main-header">📋 Model Details</p>', unsafe_allow_html=True)
    
    # Model summary
    st.subheader("📊 OLS Regression Summary")
    
    with st.expander("View Full Model Summary", expanded=False):
        st.text(model.summary().as_text())
    
    # Coefficients table
    st.subheader("📈 Model Coefficients")
    
    coef_df = pd.DataFrame({
        "Coefficient": model.params,
        "Std Error": model.bse,
        "t-statistic": model.tvalues,
        "p-value": model.pvalues,
        "CI Lower (95%)": model.conf_int()[0],
        "CI Upper (95%)": model.conf_int()[1]
    })
    
    # Highlight significant coefficients
    def highlight_significance(val):
        if isinstance(val, float) and val < 0.05:
            return "background-color: #d4edda"
        return ""
    
    st.dataframe(
        coef_df.style.format("{:,.4f}").map(highlight_significance, subset=["p-value"]),
        use_container_width=True
    )
    
    # Coefficient visualization
    st.subheader("📊 Coefficient Plot")
    
    # Remove constant for visualization
    coef_plot = coef_df.drop(index="const", errors="ignore")
    
    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(
        x=coef_plot["Coefficient"],
        y=coef_plot.index,
        orientation="h",
        error_x=dict(
            type="data",
            array=coef_plot["Std Error"] * 1.96,
            visible=True
        ),
        marker_color=["#1f77b4" if p > 0.05 else "#2ca02c" for p in coef_plot["p-value"]]
    ))
    fig_coef.add_vline(x=0, line_dash="dash", line_color="red")
    fig_coef.update_layout(
        title="Model Coefficients (95% CI)",
        xaxis_title="Coefficient Value",
        yaxis_title="Variable",
        height=max(400, len(coef_plot) * 40)
    )
    st.plotly_chart(fig_coef, use_container_width=True)
    
    # Goodness of fit
    st.markdown("---")
    st.subheader("📏 Goodness of Fit Metrics")
    
    fit_col1, fit_col2, fit_col3, fit_col4 = st.columns(4)
    
    with fit_col1:
        st.metric("R-squared", f"{model.rsquared:.4f}")
    with fit_col2:
        st.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")
    with fit_col3:
        st.metric("F-statistic", f"{model.fvalue:.2f}")
    with fit_col4:
        st.metric("Prob (F-stat)", f"{model.f_pvalue:.2e}")
    
    # Additional diagnostics
    st.markdown("---")
    st.subheader("🔍 Model Diagnostics")
    
    diag_col1, diag_col2 = st.columns(2)
    
    with diag_col1:
        st.metric("AIC", f"{model.aic:.2f}")
    with diag_col2:
        st.metric("BIC", f"{model.bic:.2f}")


# =============================================================================
# PAGE: DATA EXPORT
# =============================================================================
def render_data_export(df_raw, df):
    """Data export and download page"""
    
    st.markdown('<p class="main-header">📥 Data Export</p>', unsafe_allow_html=True)
    
    st.subheader("📊 Available Data")
    
    # Raw data
    st.markdown("**Raw Data (Original Features)**")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.caption(f"Total rows: {len(df_raw)}")
    
    st.download_button(
        label="📥 Download Raw Data (CSV)",
        data=df_raw.to_csv(index=False).encode("utf-8"),
        file_name="mmm_raw_data.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Processed data
    st.markdown("**Processed Data (with Features)**")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Total rows: {len(df)}")
    
    st.download_button(
        label="📥 Download Processed Data (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="mmm_processed_data.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Data statistics
    st.subheader("📈 Data Statistics")
    
    st.dataframe(
        df.describe().style.format("{:,.2f}"),
        use_container_width=True
    )


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    """Main application entry point"""
    
    # Load data and train model
    df_raw, df, model, x_cols, tcfg = load_data_and_train()
    
    if df_raw is None:
        st.error("❌ Dataset not found!")
        st.info("Please run: `python -m src.run_model` to generate the data first.")
        return
    
    # Render sidebar and get current page
    page = render_sidebar()
    
    # Render current page
    if page == "📈 Overview":
        render_overview(df_raw, df, model, x_cols)
    elif page == "💰 Revenue Analysis":
        render_revenue_analysis(df_raw, df, model, x_cols)
    elif page == "📺 Channel Performance":
        render_channel_performance(df_raw, df, model, x_cols)
    elif page == "🧪 Simulation Lab":
        render_simulation_lab(df_raw, df, model, x_cols, tcfg)
    elif page == "📋 Model Details":
        render_model_details(model, df, x_cols)
    elif page == "📥 Data Export":
        render_data_export(df_raw, df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>📊 Marketing Mix Model & Brand Strategy Simulator</p>
        <p><small>Built with Streamlit | Powered by Statsmodels</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

