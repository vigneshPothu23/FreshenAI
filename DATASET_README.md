# FreshSense AI — Dataset Package

Chennai perishable-inventory dataset. Five linked tables plus a documented
data-quality issue log and a reference cleaning pipeline.

Reproducible: `seed = 42` (generation), `seed = 7` (defect injection).

---

## Folder layout

```
data/
├── raw/          ← START HERE. Sprint 1 input. Deliberately messy.
├── clean/        ← Ground truth. Use to score your own cleaning.
├── cleaned/      ← Output of clean_pipeline.py. Reference solution.
├── data_quality_issue_log.csv
├── cleaning_report.csv
└── scripts/
    ├── generate_data.py    regenerate clean/ from scratch
    ├── make_raw.py         re-inject the 28 defects
    └── clean_pipeline.py   raw/ → cleaned/  (Sprint 1 reference)
```

**Load `raw/` in your application.** The messiness is the point — cleaning it
visibly is where Sprint 1 marks come from.

---

## Files

| File | Rows | Cols | Nulls | Dupes | Purpose |
|---|---|---|---|---|---|
| `raw/inventory_raw.csv` | 1,482 | 46 | 764 | 17 | Current stock batches — the central entity |
| `raw/sales_history_raw.csv` | 7,915 | 11 | 646 | 40 | 180 days × 45 products — the time series |
| `raw/orders_raw.csv` | 909 | 27 | 909 | 15 | Transactions — interaction matrix + trust signals |
| `raw/buyers_raw.csv` | 60 | 15 | 3 | 0 | Hotels, caterers, cloud kitchens, messes, NGOs |
| `raw/sellers_raw.csv` | 120 | 11 | 13 | 0 | Ratings, dispute rates, FSSAI status, coordinates |

After cleaning: inventory 1,425 · sales 8,100 (gap-filled to a complete daily
grid) · orders 869.

> Note on `orders` nulls: ~790 of these are `Buyer_Rating_Given` on recent
> orders — **right-censored, not missing**. The buyer has not rated yet. Do not
> impute these; exclude them and flag with `Awaiting_Rating`.

---

## Which sprint uses what

| Sprint | Primary file | What it gives you |
|---|---|---|
| 1 EDA | all five | 28 documented defects; waste by category/zone; breach-vs-spoilage |
| 2 Forecasting | `sales_history` | 180 days, weekly seasonality, Chennai temperature, festival spikes |
| 2 Spoilage | `inventory` | `Age_Ratio`, `Temp_Breach_Hours`, `Humidity_Pct` → `Is_Spoiled` |
| 3 Recommendation | `inventory` + `buyers` + `sellers` | lat/long for distance, ratings, buyer constraints |
| 3 Collaborative filtering | `orders` | buyer × product matrix |
| 4 RAG | — | **Not in this package.** Use real FSSAI/FAO documents. |
| 5 Multi-Agent | all five | live matching, replanning, order creation |
| 6 Monitoring | `orders` + `inventory` | predicted vs actual spoilage; dispute rates |
| 7 Dashboard | `orders` | GMV, savings, waste prevented |

---

## Key columns

### `inventory_raw.csv`
Batch identity and history: `Inventory_ID`, `Seller_ID`, `Product_Name`,
`Category`, `Arrival_Date`, `Expiry_Date`, `Shelf_Life_Days`,
`Stock_Age_Days`, `Age_Ratio`, `Days_To_Expiry`

Environment (drives spoilage): `Storage_Type`, `Storage_Temperature_C`,
`Ambient_Temp_C`, `Humidity_Pct`, **`Temp_Breach_Hours`**,
`Perishability_Level`

Commercial: `Cost_Price`, `Selling_Price`, `Discount_Pct`, `Discounted_Price`,
`Quantity_Available`, `Surplus_Quantity`

Geography: `Zone`, `Latitude`, `Longitude` (20 real Chennai zones)

Targets: **`Is_Spoiled`**, `Spoilage_Risk_Score`, `Quality_Grade` (A/B/C),
`Waste_Quantity`, `Waste_Value_INR`

### `sales_history_raw.csv`
`Date`, `Product_Name`, `Category`, `Units_Sold`, `Avg_Selling_Price`,
`Ambient_Temp_C`, `Humidity_Pct`, `Is_Weekend`, `Is_Festival`, `Day_Of_Week`

### `orders_raw.csv`
`Order_ID`, `Order_Date`, `Buyer_ID`, `Seller_ID`, `Product_Name`,
`Quantity_Ordered`, `Unit_Price_Paid`, `MRP_Unit_Price`, `Order_Value_INR`,
`Buyer_Savings_INR`, `Distance_Km`, `Match_Source`, `Dispute_Raised`,
`Waste_Prevented_Kg`

---

## The 28 data-quality defects (Sprint 1)

Every defect has a documented real-world cause. Full list in
`data_quality_issue_log.csv`. The ones worth demonstrating:

| Defect | Rows | Cause | Correct fix |
|---|---|---|---|
| **Humidity missing** | 345 | Ambient-storage shops have no sensor — **MNAR** | Impute by `Storage_Type` group + add `Has_Humidity_Sensor` flag. **A global mean is wrong here.** |
| Temp logger missing | 29 | Non-FSSAI shops lack data loggers — MAR | Impute by `Store_Type` + `Storage_Type` median |
| Seller rating missing | 71 | Sellers onboarded <50 days — no completed orders | Cold-start prior, not zero |
| **False zeros in sales** | 120 | Shop shut, logged `0` instead of `null` | Detect vs true zeros. Filling with 0 under-predicts the forecast. |
| Entire dates missing | 5 days | Server outage + power failure | Reindex to a complete daily grid — Prophet-style models need continuity |
| Duplicate Date+Product | 40 | ETL ran twice | `drop_duplicates` — otherwise demand doubles silently |
| Exact duplicate rows | 28 | Double-tap on slow 3G | `drop_duplicates()` |
| Near-duplicate listings | 18 | Same stock listed twice | Dedupe on `[Seller_ID, Product_Name, Arrival_Date]` |
| Mixed date formats | 160 | Android vs iOS serialisation | `format='mixed', dayfirst=True` — verify no silent swaps |
| Price as text | 55 | `Rs.45.20` / `₹45.20` from legacy export | Regex strip → float. Column loads as `object` until fixed. |
| `Is_Spoiled` = Yes/Y/No/N | 90 | Two app versions | Normalise to 0/1 — otherwise 4 classes |
| Zone spelling variants | 105 | Free-text entry (`T.Nagar`/`TNagar`/`t nagar`) | Mapping dict before any `groupby` |
| Cost price ×100 | 7 | Paise entered as rupees | Detect via `Cost_Price > Selling_Price` |
| Impossible quantities | 12 | 9999, 0, negative — kg/gram confusion | Clip to plausible range |
| Sensor spikes | 9 | IoT disconnect sentinel (−40, 999) | Domain bounds −5…45 °C, then impute |
| Negative sales | 22 | Returns booked as negative | Clip or net out |

**The one to demo:** the 120 false zeros. A team that fills them with `0` gets
a forecast that under-predicts. A team that spots them as disguised nulls
doesn't. Call it out explicitly.

---

## Data provenance

Required for your submission. Reproduce this table in your README.

| Field group | Status | Source / formula |
|---|---|---|
| Chennai zone coordinates | **Observed** | Real lat/long for 20 Chennai localities |
| Ambient temp, humidity | **Derived** | Chennai climate model: `30.5 + 5·sin(2π(doy−105)/365)`, humidity peaks in NE monsoon |
| Festival flags | **Observed** | Tamil calendar — Puthandu (14 Apr), Eid, Maha Shivaratri |
| `Age_Ratio` | **Derived** | `Stock_Age_Days / Shelf_Life_Days` |
| `Days_To_Expiry` | **Derived** | `Expiry_Date − today` |
| `Discount_Pct` | **Derived** | Decay curve: 3d→30%, 2d→40%, 1d→55%, 0d→70% |
| `Temp_Breach_Hours` | **Simulated** | Poisson load-shedding model, rate scales with summer |
| `Is_Spoiled` | **Simulated** | `0.55·age_ratio·perishability + 0.22·breach + 0.13·humidity + 0.10·temp`, threshold 0.62 |
| `Units_Sold` | **Simulated** | `base × trend × weekend × festival × seasonal × noise` |
| Shelf life, prices | **Observed** | Real Chennai retail reference values |
| Buyers, sellers, orders | **Simulated** | Seeded generators; real business-type distributions |

**Nothing is fabricated.** Every field is either arithmetic on other fields or
produced by a documented generative process whose parameters are in
`generate_data.py`.

---

## Verified signal (why your models will find something)

```
                    Is_Spoiled=No   Is_Spoiled=Yes
Age_Ratio                   0.637            1.239
Temp_Breach_Hours           1.907            2.311
Spoilage rate: 27%
```

Seasonality in `sales_history`: weekend uplift 210 vs 164 units · festival
uplift 238 vs 173 · watermelon demand correlates **0.73** with ambient
temperature.

Business totals: ₹6.96L buyer savings across 894 orders · 12.1% dispute rate ·
5.13 km average match distance.

---

## Regenerating

```bash
python scripts/generate_data.py   # → clean/   (seed 42)
python scripts/make_raw.py        # → raw/     (seed 7, injects 28 defects)
python scripts/clean_pipeline.py  # raw/ → cleaned/ + cleaning_report.csv
```

Deterministic. Same seeds, same bytes.

---

## What is deliberately NOT here

**No RAG corpus.** Sprint 4 needs *external, unstructured* documents —
FSSAI storage and licensing guidance, FAO post-harvest manuals, WHO food-safety
sheets, your own platform policy. Vector-searching your own CSV rows is a
database query with extra latency, and a reviewer will spot it. Collect 6–12
real documents and record source + licence per document.
