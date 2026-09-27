"""
clustering.py
=============
Clusters Factory x Region x Ship Mode "routes" by performance similarity
(lead time, distance, margin) using KMeans, then labels clusters as
Fast/Standard/Slow so the app can surface consistently slow routes and
congested region-product combinations, as required by the project spec.
"""
from __future__ import annotations
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = ["AvgDistanceMiles", "AvgLeadTimeDays", "AvgMarginPct", "ProfitPerOrder"]


def cluster_routes(route_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    df = route_df.copy()
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES].fillna(0))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    df["ClusterID"] = km.fit_predict(X)

    # rank clusters by mean lead time to assign human-readable speed labels
    order = df.groupby("ClusterID")["AvgLeadTimeDays"].mean().sort_values().index.tolist()
    labels = ["Fast", "Standard", "Slow"] if n_clusters == 3 else [f"Cluster {i}" for i in range(n_clusters)]
    label_map = {cluster_id: labels[i] for i, cluster_id in enumerate(order)}
    df["RouteSpeedLabel"] = df["ClusterID"].map(label_map)
    return df


def congested_region_products(df_orders: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Region x Product combinations with the highest average lead time --
    the 'congested region-product combinations' called for in the spec."""
    g = (
        df_orders.groupby(["Region", "Product Name"], as_index=False)
        .agg(
            Orders=("Row ID", "count"),
            AvgLeadTimeDays=("LeadTimeDays", "mean"),
            AvgDistanceMiles=("DistanceMiles", "mean"),
            TotalProfit=("Gross Profit", "sum"),
        )
        .sort_values("AvgLeadTimeDays", ascending=False)
    )
    return g.head(top_n)


if __name__ == "__main__":
    from data_pipeline import build_dataset, route_level_table

    df = build_dataset()
    routes = route_level_table(df)
    clustered = cluster_routes(routes)
    print(clustered["RouteSpeedLabel"].value_counts())
    print()
    print(clustered.sort_values("AvgLeadTimeDays", ascending=False).head(8))
    print()
    print("Top congested region-product combinations:")
    print(congested_region_products(df).to_string(index=False))
