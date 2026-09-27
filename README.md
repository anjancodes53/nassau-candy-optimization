# Nassau Candy Distributor — Factory Reallocation & Shipping Optimization Recommendation System

Decision-intelligence system that predicts shipping outcomes under different
factory configurations, recommends which products should be reassigned to
alternative factories, and balances shipping efficiency against profitability.

🚀 **[Open Live Project](https://cpx7fkm74dotggv9zj72vz.streamlit.app)**

## Project structure

```
nassau/
├── app.py                     # Streamlit dashboard (entry point)
├── requirements.txt
├── data/
│   ├── Nassau_Candy_Distributor.csv   # raw order-level data (10,194 rows)
│   ├── factories.csv                  # 5 factories + lat/lon
│   └── product_factory_map.csv        # current product -> factory assignment
├── src/
│   ├── data_pipeline.py       # cleaning, geo-distance, feature engineering
│   ├── modeling.py            # lead-time regressors (LR / RF / GBR) + selection
│   ├── clustering.py          # route clustering (Fast/Standard/Slow) + congestion
│   ├── simulation.py          # scenario simulation engine (reassignment)
│   └── optimization.py        # scoring, ranking, confidence, portfolio KPIs
└── artifacts/                 # generated on run: model, metrics, recommendations
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the dashboard

```bash
streamlit run app.py
```

First launch trains the model and runs the full scenario simulation
(~10-20 seconds); everything is cached afterward.

## Run the pipeline standalone (no UI)

Each module can also be run directly to inspect intermediate output:

```bash
cd src
python data_pipeline.py     # cleans data, prints data-quality summary
python modeling.py          # trains/evaluates models, saves best one
python clustering.py        # route clusters + congested combinations
python simulation.py        # full reassignment scenario grid
python optimization.py      # ranked recommendations + portfolio KPIs
```

## Important data-quality note

The raw `Order Date` / `Ship Date` fields are corrupted (apparent lead times
of 900-1,642 days — 2.5 to 4.5 years — which is operationally impossible).
This is documented in `data_pipeline.py` and in the research paper. All
lead-time predictions in this project are built on an engineered,
business-realistic proxy target (`LeadTimeDays`, driven by distance + ship
mode + region), not the corrupted raw date difference.

## Key modeling assumption

The `Cost` field in the source data is manufacturing cost only ("cost to
manufacture"), not shipping cost — shipping cost is not present in the raw
data at all. `simulation.py` estimates shipping cost from distance and
order weight (units) using a documented rate assumption, and holds
manufacturing cost constant across factories for a given product. See the
research paper, Section 4, for the full assumption set and suggested
extensions (e.g. factory-specific COGS, real freight-rate cards).

## Deliverables

- `Nassau_Candy_Research_Paper.docx` — EDA, methodology, insights, recommendations
- `Nassau_Candy_Executive_Summary.docx` — 1-2 page summary for government stakeholders
- `app.py` (this repo) — live Streamlit analytics dashboard
