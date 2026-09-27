"""
simulation.py
=============
Scenario Simulation Engine.

For every product, simulates re-assigning it to each of the five factories
(the actual assigned factory plus the four alternates), recomputing:
  - the shipping distance to each region the product actually ships to
    (region volumes used as weights, so a product's "simulated distance" at
    a candidate factory reflects its real regional demand mix)
  - predicted lead time via the trained regression model (modeling.py)
  - an estimated shipping cost (distance- and weight-driven; manufacturing
    cost is held constant across factories -- see assumptions below)
  - the resulting profit impact vs. the current assignment

Modeling assumption (documented, since it is not in the source data):
    `Cost` in the raw data is manufacturing cost only ("Cost to manufacture"),
    not shipping cost. We estimate shipping cost as:
        shipping_cost = BASE_HANDLING_PER_ORDER
                       + DistanceMiles * SHIP_COST_PER_MILE_PER_UNIT * Units
    Manufacturing cost is assumed comparable across factories for a given
    product (same recipe/BOM); only the shipping-cost component changes
    under reassignment. This is flagged as a limitation/extension point in
    the research paper.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

from data_pipeline import STATE_CENTROIDS, haversine_miles

BASE_HANDLING_PER_ORDER = 0.75
SHIP_COST_PER_MILE_PER_UNIT = 0.0035

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def _factory_lookup(factories: pd.DataFrame) -> dict:
    return {row.Factory: (row.Latitude, row.Longitude) for row in factories.itertuples()}


def _region_centroid(df: pd.DataFrame) -> dict:
    """Average customer lat/lon actually observed per Region (more accurate
    than a single state centroid since a Region spans many states)."""
    return (
        df.groupby("Region")[["CustLat", "CustLon"]].mean().apply(tuple, axis=1).to_dict()
    )


def build_product_region_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per product, per region: order volume, units, current profit/cost --
    the demand mix used to weight simulated distances."""
    g = df.groupby(["Division", "Product Name", "Region"], as_index=False).agg(
        Orders=("Row ID", "count"),
        Units=("Units", "sum"),
        Sales=("Sales", "sum"),
        Cost=("Cost", "sum"),
        GrossProfit=("Gross Profit", "sum"),
    )
    return g


def simulate_reassignments(df: pd.DataFrame, factories: pd.DataFrame, model_pipe) -> pd.DataFrame:
    from modeling import NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa

    fac_coords = _factory_lookup(factories)
    region_coords = _region_centroid(df)
    profile = build_product_region_profile(df)

    current_factory = df.drop_duplicates("Product Name")[["Product Name", "Factory", "Division"]].set_index(
        "Product Name"
    )["Factory"].to_dict()

    rows = []
    for product, prod_rows in profile.groupby("Product Name"):
        division = prod_rows["Division"].iloc[0]
        cur_factory = current_factory.get(product)
        total_units = prod_rows["Units"].sum()
        total_orders = prod_rows["Orders"].sum()
        total_profit_actual = prod_rows["GrossProfit"].sum()
        total_cost_actual = prod_rows["Cost"].sum()
        avg_unit_cost = total_cost_actual / total_units if total_units else 0

        for factory_name, (flat, flon) in fac_coords.items():
            weighted_dist = 0.0
            est_ship_cost = 0.0
            for _, r in prod_rows.iterrows():
                rlat, rlon = region_coords.get(r["Region"], (np.nan, np.nan))
                if np.isnan(rlat):
                    continue
                dist = haversine_miles(flat, flon, rlat, rlon)
                weighted_dist += dist * r["Units"]
                est_ship_cost += BASE_HANDLING_PER_ORDER * r["Orders"] + dist * SHIP_COST_PER_MILE_PER_UNIT * r["Units"]
            avg_dist = weighted_dist / total_units if total_units else np.nan

            # dominant ship mode & region for this product, used as the
            # categorical context fed to the lead-time model
            dominant_region = prod_rows.sort_values("Units", ascending=False)["Region"].iloc[0]
            pred_input = pd.DataFrame(
                [{
                    "DistanceMiles": avg_dist,
                    "Units": total_units / max(total_orders, 1),
                    "UnitCost": avg_unit_cost,
                    "Product Name": product,
                    "Factory": factory_name,
                    "Region": dominant_region,
                    "Ship Mode": "Standard Class",
                    "Division": division,
                }]
            )
            pred_lead_time = float(model_pipe.predict(pred_input)[0])

            simulated_profit = total_profit_actual - est_ship_cost if factory_name == cur_factory else (
                (total_cost_actual + total_profit_actual) - total_cost_actual - est_ship_cost
            )
            # ^ simplification: revenue - mfg cost - shipping cost, mfg cost held constant
            revenue = total_cost_actual + total_profit_actual
            simulated_profit = revenue - total_cost_actual - est_ship_cost

            rows.append({
                "Division": division,
                "Product Name": product,
                "CurrentFactory": cur_factory,
                "CandidateFactory": factory_name,
                "IsCurrent": factory_name == cur_factory,
                "TotalUnits": total_units,
                "AvgDistanceMiles": round(avg_dist, 1),
                "PredictedLeadTimeDays": round(pred_lead_time, 2),
                "EstShippingCost": round(est_ship_cost, 2),
                "SimulatedProfit": round(simulated_profit, 2),
            })

    result = pd.DataFrame(rows)
    baseline = result[result["IsCurrent"]][["Product Name", "PredictedLeadTimeDays", "SimulatedProfit"]].rename(
        columns={"PredictedLeadTimeDays": "BaselineLeadTimeDays", "SimulatedProfit": "BaselineProfit"}
    )
    result = result.merge(baseline, on="Product Name", how="left")
    result["LeadTimeDeltaDays"] = result["PredictedLeadTimeDays"] - result["BaselineLeadTimeDays"]
    result["ProfitDelta"] = result["SimulatedProfit"] - result["BaselineProfit"]
    result["ProfitDeltaPct"] = np.where(
        result["BaselineProfit"] != 0, 100 * result["ProfitDelta"] / result["BaselineProfit"].abs(), np.nan
    )
    return result


if __name__ == "__main__":
    from data_pipeline import build_dataset, load_reference_data
    from modeling import load_model

    df = build_dataset()
    factories, _ = load_reference_data()
    model = load_model()
    sim = simulate_reassignments(df, factories, model)
    sim.to_csv(ARTIFACTS_DIR / "simulation_results.csv", index=False)
    print(sim.sort_values(["Product Name", "ProfitDelta"], ascending=[True, False]).head(20).to_string(index=False))
    print(f"\nSaved {len(sim)} simulated scenarios to artifacts/simulation_results.csv")
