"""
Nassau Candy Distributor — Factory Reallocation & Shipping Optimization
Recommendation System
========================================================================
Streamlit dashboard. Run with:

    streamlit run app.py

On first run this trains the lead-time model and runs the full scenario
simulation (~10-20s); results are cached (st.cache_resource /
st.cache_data) so subsequent interactions are instant.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_pipeline import build_dataset, load_reference_data, route_level_table  # noqa: E402
from modeling import train_and_evaluate, NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402
from clustering import cluster_routes, congested_region_products  # noqa: E402
from simulation import simulate_reassignments  # noqa: E402
from optimization import score_scenarios, top_recommendations, portfolio_kpis, DEFAULT_WEIGHTS  # noqa: E402

st.set_page_config(
    page_title="Nassau Candy — Factory Reallocation & Shipping Optimization",
    page_icon="\U0001F36B",
    layout="wide",
)

# --------------------------------------------------------------------------
# Cached data / model / simulation layer
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning order data...")
def get_dataset():
    return build_dataset()


@st.cache_data(show_spinner=False)
def get_reference():
    return load_reference_data()


@st.cache_resource(show_spinner="Training lead-time prediction models...")
def get_model(_df):
    best_name, best_pipe, results, _ = train_and_evaluate(_df)
    return best_name, best_pipe, results


@st.cache_data(show_spinner="Running factory reassignment scenario simulation...")
def get_simulation(_df, _factories, _model_name):
    # _model_name only in the signature to key the cache; the pipe itself
    # is fetched fresh from the resource cache below
    best_name, model_pipe, _ = get_model(_df)
    return simulate_reassignments(_df, _factories, model_pipe)


@st.cache_data(show_spinner=False)
def get_order_counts(_df):
    return _df.groupby("Product Name", as_index=False).size().rename(columns={"size": "Orders"})


df = get_dataset()
factories, product_map = get_reference()
best_name, model_pipe, model_results = get_model(df)
sim = get_simulation(df, factories, best_name)
order_counts = get_order_counts(df)

# --------------------------------------------------------------------------
# Sidebar — user capabilities
# --------------------------------------------------------------------------
st.sidebar.title("\U0001F36B Nassau Candy")
st.sidebar.caption("Factory Reallocation & Shipping Optimization")

all_products = sorted(df["Product Name"].unique())
all_regions = sorted(df["Region"].unique())
all_ship_modes = sorted(df["Ship Mode"].unique())

st.sidebar.subheader("Filters")
sel_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)
sel_ship_modes = st.sidebar.multiselect("Ship Mode", all_ship_modes, default=all_ship_modes)
sel_product = st.sidebar.selectbox("Product (for simulator)", all_products)

st.sidebar.subheader("Optimization priority")
priority = st.sidebar.slider(
    "Speed \u2190\u2192 Profit", min_value=0, max_value=100, value=50,
    help="0 = optimize purely for fastest shipping, 100 = optimize purely for profit",
)
w_profit = priority / 100.0
w_lead = 1.0 - w_profit
weights = {"lead_time": w_lead * 0.9, "profit": w_profit * 0.9, "risk": 0.10}

filtered_df = df[df["Region"].isin(sel_regions) & df["Ship Mode"].isin(sel_ship_modes)]

scored = score_scenarios(sim, order_counts, weights=weights)
recs = top_recommendations(scored)
kpis = portfolio_kpis(scored, recs)

# --------------------------------------------------------------------------
# Header + KPI row
# --------------------------------------------------------------------------
st.title("Factory Reallocation & Shipping Optimization Recommendation System")
st.caption(
    "Decision intelligence for Nassau Candy Distributor: predicts shipping outcomes under "
    "different factory configurations, recommends product-to-factory reassignments, and "
    "balances shipping efficiency against profitability."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Lead Time Reduction", f"{kpis['Lead Time Reduction (%)']:.1f}%")
k2.metric("Profit Impact Stability", f"${kpis['Profit Impact Stability (std $)']:,.0f}")
k3.metric("Scenario Confidence", f"{kpis['Scenario Confidence Score']:.2f}")
k4.metric("Recommendation Coverage", f"{kpis['Recommendation Coverage (%)']:.0f}%")

st.divider()

tab_sim, tab_whatif, tab_recs, tab_risk = st.tabs(
    ["\U0001F3ED Factory Optimization Simulator", "\U0001F500 What-If Scenario Analysis",
     "\U0001F4CB Recommendation Dashboard", "\u26A0\uFE0F Risk & Impact Panel"]
)

# --------------------------------------------------------------------------
# TAB 1 — Factory Optimization Simulator
# --------------------------------------------------------------------------
with tab_sim:
    st.subheader(f"Predicted performance for: {sel_product}")
    prod_scored = scored[scored["Product Name"] == sel_product].sort_values("CompositeScore", ascending=False)
    current_factory = prod_scored.loc[prod_scored["IsCurrent"], "CandidateFactory"].iloc[0]
    st.caption(f"Currently produced at **{current_factory}**")

    display_cols = {
        "CandidateFactory": "Factory",
        "PredictedLeadTimeDays": "Predicted Lead Time (days)",
        "AvgDistanceMiles": "Avg Distance (mi)",
        "EstShippingCost": "Est. Shipping Cost ($)",
        "SimulatedProfit": "Simulated Profit ($)",
        "CompositeScore": "Composite Score",
    }
    view = prod_scored[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(view.style.format(precision=2), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Predicted lead time by factory**")
        st.bar_chart(view.set_index("Factory")["Predicted Lead Time (days)"])
    with c2:
        st.markdown("**Simulated profit by factory**")
        st.bar_chart(view.set_index("Factory")["Simulated Profit ($)"])

    st.markdown("##### Model performance (lead-time regressor)")
    st.caption(f"Best model selected: **{best_name}**")
    st.dataframe(pd.DataFrame(model_results).T, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 2 — What-If Scenario Analysis
# --------------------------------------------------------------------------
with tab_whatif:
    st.subheader("Current vs. Recommended Assignment")
    compare = recs[recs["Product Name"] == sel_product]
    if not compare.empty:
        row = compare.iloc[0]
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Current Factory", row["CurrentFactory"])
        cc2.metric(
            "Recommended Factory",
            row["CandidateFactory"],
            delta=f"{row['LeadTimeDeltaDays']:+.2f} days lead time",
            delta_color="inverse",
        )
        cc3.metric("Profit Impact", f"${row['ProfitDelta']:+,.2f}", f"{row['ProfitDeltaPct']:+.1f}%")

    st.markdown("##### Lead-time improvement across all products (recommended vs. current)")
    lt_chart = recs.set_index("Product Name")["LeadTimeDeltaDays"].sort_values()
    st.bar_chart(lt_chart, height=400)
    st.caption("Negative bars = faster shipping under the recommended reassignment.")

    st.markdown("##### Route clustering — consistently slow vs. fast routes")
    routes = route_level_table(filtered_df)
    clustered_routes = cluster_routes(routes)
    st.dataframe(
        clustered_routes.sort_values("AvgLeadTimeDays", ascending=False)[
            ["Factory", "Region", "Ship Mode", "Orders", "AvgLeadTimeDays", "AvgDistanceMiles",
             "AvgMarginPct", "RouteSpeedLabel"]
        ].style.format({"AvgLeadTimeDays": "{:.2f}", "AvgDistanceMiles": "{:.0f}", "AvgMarginPct": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("##### Congested region \u00d7 product combinations")
    st.dataframe(congested_region_products(filtered_df), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# TAB 3 — Recommendation Dashboard
# --------------------------------------------------------------------------
with tab_recs:
    st.subheader("Ranked Factory Reassignment Recommendations")
    st.caption("Ranked by simulated profit impact. Adjust the Speed \u2194 Profit slider in the sidebar to re-rank.")
    show = recs.copy()
    show["ProfitDelta"] = show["ProfitDelta"].round(2)
    st.dataframe(
        show.style.format({
            "LeadTimeDeltaDays": "{:+.2f}", "ProfitDelta": "${:+,.2f}", "ProfitDeltaPct": "{:+.1f}%",
            "CompositeScore": "{:.3f}", "Confidence": "{:.2f}",
        }).background_gradient(subset=["ProfitDelta"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )

    total_opportunity = recs.loc[recs["ProfitDelta"] > 0, "ProfitDelta"].sum()
    n_reassign = (recs["Recommendation"] != "Keep current factory").sum()
    st.success(
        f"**{n_reassign} of {len(recs)} products** have a positive-impact reassignment opportunity, "
        f"totaling an estimated **${total_opportunity:,.2f}** in simulated profit uplift."
    )

# --------------------------------------------------------------------------
# TAB 4 — Risk & Impact Panel
# --------------------------------------------------------------------------
with tab_risk:
    st.subheader("Risk & Impact Panel")
    low_conf = recs[recs["ConfidenceBand"] == "Low"]
    if not low_conf.empty:
        st.warning(
            f"**{len(low_conf)} recommendation(s) are Low confidence** (thin historical order volume). "
            "Validate with a small pilot before committing capacity."
        )
        st.dataframe(
            low_conf[["Product Name", "CurrentFactory", "CandidateFactory", "ProfitDelta", "Confidence"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No low-confidence recommendations under the current filters.")

    st.markdown("##### Profit-impact alerts (largest simulated losses if mis-assigned)")
    worst_case = scored.sort_values("ProfitDelta").head(10)
    st.dataframe(
        worst_case[["Product Name", "CandidateFactory", "ProfitDelta", "LeadTimeDeltaDays", "ConfidenceBand"]],
        use_container_width=True, hide_index=True,
    )

    st.markdown("##### Confidence distribution across all recommendations")
    st.bar_chart(recs["ConfidenceBand"].value_counts())

st.divider()
st.caption(
    "Note: Order Date / Ship Date fields in the source data contain a data-quality defect "
    "(apparent lead times of 2.5-4.5 years). Lead-time predictions in this app are generated "
    "from an engineered, business-realistic proxy (distance + ship mode + region) rather than "
    "the corrupted raw date difference. See the research paper for full methodology and caveats."
)
