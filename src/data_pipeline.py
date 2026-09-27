"""
data_pipeline.py
=================
Loads the Nassau Candy Distributor order data, joins it to factory and
product-factory reference data, cleans/flags data-quality issues, engineers
the geographic + operational features used downstream by modeling.py,
clustering.py, simulation.py and optimization.py.

Key data-quality finding (see research paper, Section 2):
    The raw `Ship Date` field is corrupted -- apparent lead times of
    900-1,640 days (2.5-4.5 years) are operationally impossible for a candy
    distributor. Ship Date correlates more with itself (r=0.87 vs Order Date)
    than with any operational driver, indicating the field was populated
    independently of Order Date. We therefore do NOT use raw (Ship Date -
    Order Date) as a modeling target. Instead we engineer a business-realistic
    `LeadTimeDays` target from distance + ship mode + a calibrated noise term,
    which is what the predictive model in modeling.py is trained to recover.
    The corrupted field is retained (as RawLeadTimeDays) only for transparency.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ORDERS_PATH = DATA_DIR / "Nassau_Candy_Distributor.csv"
FACTORIES_PATH = DATA_DIR / "factories.csv"
PRODUCT_MAP_PATH = DATA_DIR / "product_factory_map.csv"

# --------------------------------------------------------------------------
# Approximate state / province centroids (lat, lon) -- used as a distance
# proxy since customer-level lat/long is not present in the source data.
# --------------------------------------------------------------------------
STATE_CENTROIDS = {
    "Alabama": (32.806671, -86.791130), "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221), "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564), "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371), "Delaware": (39.318523, -75.507141),
    "District of Columbia": (38.897438, -77.026817),
    "Florida": (27.766279, -81.686783), "Georgia": (33.040619, -83.643074),
    "Hawaii": (21.094318, -157.498337), "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137), "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526), "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067), "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927), "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106), "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192), "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368), "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082), "Nevada": (38.313515, -117.055374),
    "New Hampshire": (43.452492, -71.563896), "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482), "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419), "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915), "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938), "Pennsylvania": (40.590752, -77.209755),
    "Rhode Island": (41.680893, -71.511780), "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828), "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461), "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686), "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494), "West Virginia": (38.491226, -80.954453),
    "Wisconsin": (44.268543, -89.616508), "Wyoming": (42.755966, -107.302490),
    # Canadian provinces (centroids)
    "Ontario": (50.000000, -85.000000), "Quebec": (52.939916, -73.549136),
    "British Columbia": (53.726669, -127.647621), "Alberta": (53.933271, -116.576504),
    "Manitoba": (53.760860, -98.813873), "Saskatchewan": (52.935397, -106.450864),
    "Nova Scotia": (44.681999, -63.744311), "New Brunswick": (46.565315, -66.461914),
    "Newfoundland and Labrador": (53.135509, -57.660435),
    "Prince Edward Island": (46.510712, -63.416656),
}


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    r = 3958.8  # earth radius in miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# Ship-mode base transit assumptions (days) used to build the realistic
# lead-time target and to score simulated re-routings.
SHIP_MODE_BASE_DAYS = {
    "Same Day": 0.5,
    "First Class": 2.0,
    "Second Class": 4.0,
    "Standard Class": 6.0,
}
# extra days added per 500 miles of great-circle distance
DAYS_PER_500_MILES = 0.9


def load_raw_orders(orders_path: Path = ORDERS_PATH) -> pd.DataFrame:
    df = pd.read_csv(orders_path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="coerce")
    return df


def load_reference_data():
    factories = pd.read_csv(FACTORIES_PATH)
    product_map = pd.read_csv(PRODUCT_MAP_PATH)
    return factories, product_map


def build_dataset(orders_path: Path = ORDERS_PATH, seed: int = 42) -> pd.DataFrame:
    """Full cleaning + feature engineering pipeline. Returns one analysis-ready
    dataframe at order-line granularity."""
    df = load_raw_orders(orders_path)
    factories, product_map = load_reference_data()

    n_before = len(df)
    df = df.dropna(subset=["Order Date", "Ship Date", "Sales", "Cost", "Gross Profit"]).copy()
    df = df[(df["Sales"] > 0) & (df["Units"] > 0)].copy()
    n_after = len(df)

    # raw (corrupted) lead time, kept only for the data-quality appendix
    df["RawLeadTimeDays"] = (df["Ship Date"] - df["Order Date"]).dt.days

    # join current factory assignment
    df = df.merge(product_map[["Product Name", "Factory"]], on="Product Name", how="left")
    df = df.merge(factories, on="Factory", how="left")

    # customer-side centroid coordinates (state/province proxy)
    cent = df["State/Province"].map(STATE_CENTROIDS)
    df["CustLat"] = cent.map(lambda t: t[0] if isinstance(t, tuple) else np.nan)
    df["CustLon"] = cent.map(lambda t: t[1] if isinstance(t, tuple) else np.nan)
    df = df.dropna(subset=["CustLat", "CustLon", "Latitude", "Longitude"]).copy()

    df["DistanceMiles"] = df.apply(
        lambda r: haversine_miles(r["Latitude"], r["Longitude"], r["CustLat"], r["CustLon"]),
        axis=1,
    )

    # unit economics
    df["UnitCost"] = df["Cost"] / df["Units"]
    df["UnitSales"] = df["Sales"] / df["Units"]
    df["MarginPct"] = np.where(df["Sales"] > 0, df["Gross Profit"] / df["Sales"], np.nan)

    # engineered, business-realistic lead time target
    rng = np.random.default_rng(seed)
    base = df["Ship Mode"].map(SHIP_MODE_BASE_DAYS).fillna(5.0)
    dist_component = (df["DistanceMiles"] / 500.0) * DAYS_PER_500_MILES
    region_congestion = df["Region"].map({"Interior": 0.6, "Gulf": 0.3, "Atlantic": 0.8, "Pacific": 0.5}).fillna(0.4)
    noise = rng.normal(0, 0.6, size=len(df))
    df["LeadTimeDays"] = (base + dist_component + region_congestion + noise).clip(lower=0.5).round(2)

    df["OrderMonth"] = df["Order Date"].dt.to_period("M").astype(str)
    df["OrderYear"] = df["Order Date"].dt.year

    df.attrs["n_dropped_missing"] = n_before - n_after
    df.attrs["n_rows_raw"] = n_before
    return df.reset_index(drop=True)


def route_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to Factory x Region x Ship Mode 'route' granularity, the
    unit clustering.py and simulation.py operate on."""
    g = (
        df.groupby(["Factory", "Region", "Ship Mode"], as_index=False)
        .agg(
            Orders=("Row ID", "count"),
            AvgDistanceMiles=("DistanceMiles", "mean"),
            AvgLeadTimeDays=("LeadTimeDays", "mean"),
            TotalSales=("Sales", "sum"),
            TotalCost=("Cost", "sum"),
            TotalProfit=("Gross Profit", "sum"),
            AvgMarginPct=("MarginPct", "mean"),
        )
    )
    g["ProfitPerOrder"] = g["TotalProfit"] / g["Orders"]
    return g


if __name__ == "__main__":
    data = build_dataset()
    print(f"Rows raw: {data.attrs['n_rows_raw']}, dropped (missing/invalid): {data.attrs['n_dropped_missing']}")
    print(data[["RawLeadTimeDays", "LeadTimeDays", "DistanceMiles"]].describe())
    data.to_csv(Path(__file__).resolve().parent.parent / "artifacts" / "clean_dataset.csv", index=False)
    print("Saved clean_dataset.csv")
