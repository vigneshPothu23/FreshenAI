"""Fixtures for the repository test suite.

The database is built from the project's own ``schema.sql`` and seeded with a
small deterministic dataset. Building from the real schema is the point: it is
what catches a repository whose SQL has drifted from the tables it queries,
which is exactly the class of defect Milestone 2 found. The seed is small and
hand-written so the suite runs without depending on the pipeline having been
executed first.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from freshsense.db.database import Database  # noqa: E402
from freshsense.db.repository import (BuyerRepository,  # noqa: E402
                                      ChatRepository, DocumentRepository,
                                      InventoryRepository, MetricsRepository,
                                      MonitoringRepository, OrdersRepository,
                                      PartyRepository, PredictionRepository,
                                      RecommendationRepository,
                                      SalesRepository, SellerRepository)

SCHEMA = _ROOT / "database" / "schema.sql"

ZONES = ["T Nagar", "Adyar", "Velachery"]
PRODUCTS = [
    ("Tomato", "Vegetables", "kg", 5, "High", "Ambient"),
    ("Paneer", "Dairy", "kg", 5, "Very High", "Chilled"),
    ("Potato", "Vegetables", "kg", 21, "Low", "Ambient"),
]


def _seed(db: Database) -> None:
    """Insert a deterministic dataset covering every mapped table.

    Shapes matter more than volume here. The seed deliberately includes an
    expired batch, a sold-out batch and an order with a NULL ``batch_id``,
    because each exercises a branch a uniformly healthy dataset would never
    reach.
    """
    with db.connect() as conn:
        for index, (name, category, unit, shelf, perish, storage) in enumerate(PRODUCTS):
            conn.execute(
                "INSERT INTO items (item_id, name, category, unit, "
                "shelf_life_days, perishability_level, storage_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (index + 1, name, category, unit, shelf, perish, storage),
            )

        for index in range(1, 6):
            conn.execute(
                "INSERT INTO sellers (seller_id, seller_name, store_type, zone, "
                "latitude, longitude, seller_rating, total_orders, "
                "dispute_rate_pct, fssai_verified, onboarded_date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"SEL{index:03d}", f"Store {index}", "Grocery",
                 ZONES[index % len(ZONES)], 13.0 + index / 100, 80.2 + index / 100,
                 3.0 + index / 10, index * 4, index * 1.5, index % 2,
                 "2026-01-01"),
            )

        for index in range(1, 5):
            conn.execute(
                "INSERT INTO buyers (buyer_id, buyer_name, buyer_type, zone, "
                "latitude, longitude, avg_order_value, price_sensitivity, "
                "max_distance_km, min_acceptable_grade, accepts_split_order, "
                "buyer_rating, total_orders) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"BUY{index:03d}", f"Kitchen {index}",
                 "Restaurant" if index % 2 else "Caterer",
                 ZONES[index % len(ZONES)], 13.0 + index / 100, 80.2 + index / 100,
                 1000.0 * index, 0.5, 10.0, "B", 1, 4.0, index * 3),
            )

        # 12 active batches, plus one expired and one sold out.
        for index in range(1, 13):
            name, category, unit, shelf, perish, storage = PRODUCTS[index % 3]
            days_to_expiry = index % 6            # 0..5, so some are at risk
            conn.execute(
                """
                INSERT INTO batches
                    (batch_id, item_id, seller_id, product_name, category, unit,
                     zone, latitude, longitude, quantity_initial,
                     quantity_available, expiry_date, shelf_life_days,
                     stock_age_days, days_to_expiry, age_ratio, storage_type,
                     storage_temperature_c, humidity_pct, temp_breach_hours,
                     perishability_level, perishability_score,
                     storage_risk_score, cost_price, mrp, discount_pct,
                     effective_price, daily_avg_sales, quality_grade, status,
                     is_spoiled, environment_stress_index, action_priority,
                     inventory_value, capital_at_risk, projected_surplus)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (f"BATCH{index:03d}", (index % 3) + 1, f"SEL{(index % 5) + 1:03d}",
                 name, category, unit, ZONES[index % len(ZONES)],
                 13.0 + index / 100, 80.2 + index / 100,
                 float(index * 10), float(index * 10), "2030-01-01", shelf,
                 shelf - days_to_expiry, days_to_expiry,
                 round((shelf - days_to_expiry) / shelf, 3), storage,
                 8.0, 70.0, 0.5, perish, 3, 2, 20.0, 40.0, 10.0, 36.0, 5.0,
                 "A" if days_to_expiry > 2 else "C", "active", 0,
                 0.3, 5.0, float(index * 360), float(index * 200), 2.0),
            )

        conn.execute(
            "INSERT INTO batches (batch_id, item_id, seller_id, product_name, "
            "category, unit, zone, quantity_available, expiry_date, "
            "shelf_life_days, days_to_expiry, status, quality_grade, "
            "effective_price, action_priority, inventory_value, capital_at_risk) "
            "VALUES ('BATCH-EXP', 1, 'SEL001', 'Tomato', 'Vegetables', 'kg', "
            "'Adyar', 8.0, '2020-01-01', 5, -3, 'expired', 'C', 20.0, 9.0, "
            "160.0, 100.0)"
        )
        conn.execute(
            "INSERT INTO batches (batch_id, item_id, seller_id, product_name, "
            "category, unit, zone, quantity_available, expiry_date, "
            "shelf_life_days, days_to_expiry, status, quality_grade, "
            "effective_price) "
            "VALUES ('BATCH-OUT', 2, 'SEL002', 'Paneer', 'Dairy', 'kg', "
            "'Adyar', 0.0, '2030-01-01', 5, 4, 'sold_out', 'A', 90.0)"
        )

        # Complete daily grid: every product on every day.
        for day in range(1, 11):
            for name, category, unit, *_ in PRODUCTS:
                conn.execute(
                    "INSERT INTO daily_item_stats (stat_date, product_name, "
                    "category, unit, units_sold, avg_selling_price, revenue, "
                    "ambient_temp_c, humidity_pct, is_weekend, is_festival, "
                    "day_of_week) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (f"2026-06-{day:02d}", name, category, unit,
                     float(10 * day), 40.0, float(400 * day), 31.0, 72.0,
                     int(day % 7 in (0, 6)), 0, "Monday"),
                )

        for index in range(1, 9):
            conn.execute(
                """
                INSERT INTO orders
                    (order_id, order_date, buyer_id, seller_id, batch_id,
                     product_name, category, quantity, unit, unit_price, mrp,
                     order_value, buyer_savings, distance_km, quality_grade,
                     match_source, payment_status, dispute_raised,
                     waste_prevented_kg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (f"ORD{index:03d}", f"2026-06-{index:02d}",
                 f"BUY{(index % 4) + 1:03d}", f"SEL{(index % 5) + 1:03d}",
                 f"BATCH{index:03d}", PRODUCTS[index % 3][0],
                 PRODUCTS[index % 3][1], float(index * 2), "kg", 36.0, 40.0,
                 float(index * 72), float(index * 8), 3.5, "A", "Manual",
                 "Paid", index % 4 == 0, float(index)),
            )

        # An order whose batch was never listed: a NULL foreign key the schema
        # permits, and a branch cancel() must handle without restoring stock.
        conn.execute(
            "INSERT INTO orders (order_id, order_date, buyer_id, seller_id, "
            "batch_id, product_name, quantity, unit_price, order_value, "
            "payment_status) VALUES ('ORD-NOBATCH', '2026-06-09', 'BUY001', "
            "'SEL001', NULL, 'Tomato', 5.0, 36.0, 180.0, 'Paid')"
        )


@pytest.fixture(scope="session")
def seeded_db(tmp_path_factory) -> Database:
    """A schema-built, seeded database, created once per session."""
    if not SCHEMA.is_file():
        pytest.skip(f"schema.sql not found at {SCHEMA}")
    path = tmp_path_factory.mktemp("freshsense") / "seed.db"
    database = Database(path)
    database.initialise(SCHEMA)
    _seed(database)
    return database


@pytest.fixture()
def db(seeded_db, tmp_path) -> Database:
    """A private copy per test.

    Copied rather than rolled back, so a test that leaves the database in a
    surprising state cannot leak into the next one.
    """
    private = tmp_path / "case.db"
    shutil.copyfile(seeded_db.db_path, private)
    return Database(private)


@pytest.fixture()
def inventory(db) -> InventoryRepository:
    return InventoryRepository(db)


@pytest.fixture()
def sales(db) -> SalesRepository:
    return SalesRepository(db)


@pytest.fixture()
def orders(db) -> OrdersRepository:
    return OrdersRepository(db)


@pytest.fixture()
def buyers(db) -> BuyerRepository:
    return BuyerRepository(db)


@pytest.fixture()
def sellers(db) -> SellerRepository:
    return SellerRepository(db)


@pytest.fixture()
def parties(db) -> PartyRepository:
    return PartyRepository(db)


@pytest.fixture()
def documents(db) -> DocumentRepository:
    return DocumentRepository(db)


@pytest.fixture()
def chat(db) -> ChatRepository:
    return ChatRepository(db)


@pytest.fixture()
def metrics(db) -> MetricsRepository:
    return MetricsRepository(db)


@pytest.fixture()
def predictions(db) -> PredictionRepository:
    return PredictionRepository(db)


@pytest.fixture()
def recommendations(db) -> RecommendationRepository:
    return RecommendationRepository(db)


@pytest.fixture()
def monitoring(db) -> MonitoringRepository:
    return MonitoringRepository(db)