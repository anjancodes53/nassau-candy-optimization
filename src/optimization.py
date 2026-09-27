"""
optimization.py
================
Optimization & Recommendation Logic.

Takes the scenario simulation output (simulation.py) and:
  1. Scores every candidate factory for every product on a blended
     objective (lead-time reduction, profit impact, risk).
  2. Attaches a Scenario Confidence Score based on how much historical
     order volume backs the recommendation (few orders -> low confidence;
     this matters a lot here since Sugar/Other-division products have as
     few as 3-10 historical orders vs. 1,800+ for Chocolate SKUs).
  3. Produces the top-N factory reassignment recommendation per product,
     plus the four portfolio-level KPIs required by the spec.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# relative importance of each objective when ranking candidate factories
DEFAULT_WEIGHTS = {"lead_time": 0.4, "profit": 0.5, "risk": 0.1}


def confidence_score(order_count: int) -> float:
    """0-1 score: saturates at 1.0 around 60+ historical orders, penalizes
    thin-data SKUs (e.g. the Sugar-division long tail with 3-10 orders)."""
    return float(min(1.0, np.log1p(order_count) / np.log1p(60)))


def confidence_band(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def score_scenarios(sim: pd.DataFrame, order_counts: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    df = sim.merge(order_counts, on="Product Name", how="left")
    df["Confidence"] = df["Orders"].fillna(0).astype(int).map(confidence_score)
    df["ConfidenceBand"] = df["Confidence"].map(confidence_band)

    # normalize lead-time delta (negative/improvement is good) and profit delta
    # (positive is good) 0-1 within each product, via groupby().transform so
    # the original columns (incl. "Product Name") are never dropped.
    lt = -df["LeadTimeDeltaDays"]  # improvement = positive after negation
    pf = df["ProfitDelta"]
    grp = df["Product Name"]

    lt_min, lt_max = lt.groupby(grp).transform("min"), lt.groupby(grp).transform("max")
    pf_min, pf_max = pf.groupby(grp).transform("min"), pf.groupby(grp).transform("max")

    df["LeadTimeScore"] = np.where(lt_max != lt_min, (lt - lt_min) / (lt_max - lt_min), 0.5)
    df["ProfitScore"] = np.where(pf_max != pf_min, (pf - pf_min) / (pf_max - pf_min), 0.5)
    df["RiskScore"] = df["Confidence"]  # higher confidence = lower risk
    df["CompositeScore"] = (
        weights["lead_time"] * df["LeadTimeScore"]
        + weights["profit"] * df["ProfitScore"]
        + weights["risk"] * df["RiskScore"]
    )
    return df


def top_recommendations(scored: pd.DataFrame, top_n: int = 1) -> pd.DataFrame:
    """Best candidate factory per product (excludes the current factory
    unless it is itself the best option, in which case 'no change' is
    the recommendation)."""
    ranked = scored.sort_values(["Product Name", "CompositeScore"], ascending=[True, False])
    recs = ranked.groupby("Product Name").head(top_n).copy()
    recs["Recommendation"] = np.where(
        recs["IsCurrent"], "Keep current factory", "Reassign to " + recs["CandidateFactory"]
    )
    return recs[[
        "Division", "Product Name", "CurrentFactory", "CandidateFactory", "Recommendation",
        "TotalUnits", "LeadTimeDeltaDays", "ProfitDelta", "ProfitDeltaPct",
        "CompositeScore", "Confidence", "ConfidenceBand",
    ]].sort_values("ProfitDelta", ascending=False)


def portfolio_kpis(scored: pd.DataFrame, recs: pd.DataFrame) -> dict:
    """The four KPIs called for in the project spec."""
    actionable = recs[~recs["Recommendation"].eq("Keep current factory")]
    lead_time_reduction_pct = (
        100 * (-actionable["LeadTimeDeltaDays"]).clip(lower=0).sum()
        / scored.loc[scored["IsCurrent"], "BaselineLeadTimeDays"].sum()
    ) if len(actionable) else 0.0

    profit_impact_stability = float(recs["ProfitDelta"].std()) if len(recs) > 1 else 0.0
    scenario_confidence = float(recs["Confidence"].mean())
    recommendation_coverage = float(len(recs) / scored["Product Name"].nunique())

    return {
        "Lead Time Reduction (%)": round(lead_time_reduction_pct, 2),
        "Profit Impact Stability (std $)": round(profit_impact_stability, 2),
        "Scenario Confidence Score": round(scenario_confidence, 3),
        "Recommendation Coverage (%)": round(100 * recommendation_coverage, 1),
    }


if __name__ == "__main__":
    from data_pipeline import build_dataset, load_reference_data
    from modeling import load_model
    from simulation import simulate_reassignments

    df = build_dataset()
    factories, _ = load_reference_data()
    model = load_model()
    sim = simulate_reassignments(df, factories, model)

    order_counts = df.groupby("Product Name", as_index=False).size().rename(columns={"size": "Orders"})
    scored = score_scenarios(sim, order_counts)
    recs = top_recommendations(scored)
    kpis = portfolio_kpis(scored, recs)

    recs.to_csv(ARTIFACTS_DIR / "recommendations.csv", index=False)
    print(recs.to_string(index=False))
    print("\nPortfolio KPIs:")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
