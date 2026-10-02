"""Repository layer: schema, CRUD, search, pagination and integration tests.

Grouped by what a failure would mean:

``TestSchemaAlignment``  every repository addresses a table that exists
``TestSmoke``            every read path runs against seeded data
``TestCrud``             create, read, update and delete round-trip
``TestSearchFilters``    filters actually restrict, and restrict correctly
``TestPagination``       a page reports the total it was drawn from
``TestTransactions``     read-then-write is atomic and never loses stock
``TestForeignKeys``      writes cannot orphan a row
``TestBusinessQueries``  aggregates agree with raw SQL
``TestIntegration``      repositories agree with each other

Regression tests name the defect they guard in the docstring: a test named for
its symptom outlives the memory of why it was written.
"""

from __future__ import annotations

import pandas as pd
import pytest

from freshsense.db.repository import (BaseRepository, BuyerRepository,
                                      ChatRepository, DocumentRepository,
                                      InventoryRepository, MetricsRepository,
                                      MonitoringRepository, OrdersRepository,
                                      Page, PredictionRepository,
                                      RecommendationRepository,
                                      SalesRepository, SellerRepository)
from freshsense.exceptions import DatabaseError

ALL_REPOSITORIES = [
    InventoryRepository, SalesRepository, OrdersRepository, BuyerRepository,
    SellerRepository, DocumentRepository, ChatRepository, MetricsRepository,
    PredictionRepository, RecommendationRepository, MonitoringRepository,
]


# ══════════════════════════════════════════════════════════════════════════
class TestSchemaAlignment:
    """Every repository must address a table the schema actually defines."""

    @pytest.mark.parametrize("repository_class", ALL_REPOSITORIES)
    def test_table_exists(self, db, repository_class):
        """Regression: three repositories targeted non-existent tables.

        ``inventory``, ``sales`` and ``metrics`` belong to the analytical
        vocabulary the pipeline emits; the database normalises them into
        ``batches``, ``daily_item_stats`` and ``model_metrics``. Every call on
        those classes raised ``no such table``.
        """
        repository = repository_class(db)
        assert repository.exists(), (
            f"{repository_class.__name__}.TABLE = {repository.TABLE!r} "
            f"does not exist in schema.sql"
        )

    @pytest.mark.parametrize("repository_class", ALL_REPOSITORIES)
    def test_count_runs(self, db, repository_class):
        assert repository_class(db).count() >= 0

    def test_inventory_targets_batches(self, inventory):
        assert inventory.TABLE == "batches"

    def test_sales_targets_daily_item_stats(self, sales):
        assert sales.TABLE == "daily_item_stats"

    def test_metrics_targets_model_metrics(self, metrics):
        assert metrics.TABLE == "model_metrics"

    def test_document_search_uses_a_real_column(self, documents):
        """Regression: search() filtered on documents.content, which does not
        exist. Chunk text lives in chunks.text."""
        assert "content" not in documents.columns()
        documents.search("anything")          # must not raise


# ══════════════════════════════════════════════════════════════════════════
class TestSmoke:
    """Every read path executes against seeded data."""

    def test_inventory_reads(self, inventory):
        assert inventory.count() == 14
        assert len(inventory.active()) == 12
        assert not inventory.load_dataframe().empty
        assert not inventory.by_category().empty
        assert not inventory.by_zone().empty
        assert inventory.kpi_summary()["total_batches"] == 12

    def test_sales_reads(self, sales):
        assert sales.count() == 30
        assert len(sales.products()) == 3
        assert not sales.series().empty
        assert not sales.totals_by_day().empty
        assert not sales.top_products().empty

    def test_order_reads(self, orders):
        assert orders.count() == 9
        assert not orders.recent().empty
        assert not orders.interaction_matrix().empty
        assert orders.business_kpis()["orders"] > 0

    def test_party_reads(self, buyers, sellers, parties):
        assert len(buyers.load_dataframe()) == 4
        assert len(sellers.load_dataframe()) == 5
        assert len(parties.buyers()) == 4
        assert len(parties.sellers()) == 5
        assert parties.buyer("BUY001") is not None
        assert parties.seller("SEL001") is not None

    def test_empty_tables_return_frames_not_errors(self, documents, chat,
                                                   metrics, predictions,
                                                   recommendations, monitoring):
        """An unused table must return an empty frame, never raise."""
        for frame in (documents.all_documents(), chat.history(),
                      metrics.all_metrics(), metrics.latest(),
                      predictions.recent(), predictions.outcomes(),
                      recommendations.recent(), monitoring.events(),
                      monitoring.llm_stats(), monitoring.agent_stats()):
            assert isinstance(frame, pd.DataFrame)

    def test_distinct_rejects_arbitrary_columns(self, inventory):
        """The column name is interpolated into SQL, so it must be whitelisted."""
        with pytest.raises(ValueError):
            inventory.distinct("batch_id; DROP TABLE batches")

    def test_distinct_returns_values(self, inventory):
        assert set(inventory.distinct("zone")) <= {"T Nagar", "Adyar", "Velachery"}


# ══════════════════════════════════════════════════════════════════════════
class TestCrud:
    """Create, read, update and delete round-trip correctly."""

    def test_batch_lifecycle(self, inventory):
        batch_id = inventory.create({
            "product_name": "Tomato", "category": "Vegetables", "unit": "kg",
            "quantity_available": 40.0, "expiry_date": "2030-01-01",
            "status": "active", "effective_price": 30.0, "days_to_expiry": 4,
        })
        assert inventory.get(batch_id) is not None

        assert inventory.update(batch_id, {"discount_pct": 35.0}) == 1
        assert inventory.get(batch_id)["discount_pct"] == 35.0

        assert inventory.adjust_quantity(batch_id, -15.0) == 25.0
        assert inventory.delete(batch_id) == 1
        assert inventory.get(batch_id) is None

    def test_insert_rejects_a_record_with_no_known_column(self, documents):
        """A frame in the wrong column vocabulary must fail loudly.

        Silently writing zero columns is how the pre-Milestone-1 loader
        produced ``CREATE TABLE x ()``.
        """
        with pytest.raises(DatabaseError, match="shares none of its columns"):
            documents._insert({"nonsense": 1, "also_wrong": 2})

    def test_update_ignores_unknown_columns(self, inventory):
        """An unknown key is dropped with a warning, never written."""
        inventory.update("BATCH001", {"not_a_column": 1, "discount_pct": 12.0})
        batch = inventory.get("BATCH001")
        assert batch["discount_pct"] == 12.0
        assert "not_a_column" not in batch

    def test_update_with_only_unknown_columns_changes_nothing(self, buyers):
        assert buyers.update("BUY001", {"not_a_column": 1}) == 0

    def test_order_lifecycle(self, orders):
        order_id = orders.create({
            "buyer_id": "BUY001", "seller_id": "SEL001",
            "product_name": "Tomato", "quantity": 4.0, "unit_price": 36.0,
            "order_value": 144.0, "payment_status": "Paid",
        })
        assert orders.get(order_id) is not None
        assert orders.update(order_id, {"payment_status": "Refunded"}) == 1
        assert orders.get(order_id)["payment_status"] == "Refunded"
        assert orders.delete(order_id) == 1
        assert orders.get(order_id) is None

    def test_buyer_lifecycle(self, buyers):
        buyers.upsert({"buyer_id": "B-NEW", "buyer_name": "Test Kitchen",
                       "buyer_type": "Restaurant", "zone": "Guindy"})
        assert buyers.get("B-NEW")["buyer_name"] == "Test Kitchen"
        assert buyers.update("B-NEW", {"max_distance_km": 12.0}) == 1
        assert buyers.get("B-NEW")["max_distance_km"] == 12.0
        assert buyers.delete("B-NEW") == 1
        assert buyers.get("B-NEW") is None

    def test_seller_lifecycle(self, sellers):
        sellers.upsert({"seller_id": "S-NEW", "seller_name": "Test Store",
                        "zone": "Adyar", "seller_rating": 4.0})
        assert sellers.get("S-NEW")["seller_name"] == "Test Store"
        assert sellers.update("S-NEW", {"seller_rating": 4.8}) == 1
        assert sellers.get("S-NEW")["seller_rating"] == 4.8
        assert sellers.delete("S-NEW") == 1

    def test_upsert_replaces_rather_than_duplicating(self, db, sellers):
        before = db.row_count("sellers")
        for name in ("First", "Second"):
            sellers.upsert({"seller_id": "S-DUP", "seller_name": name})
        assert db.row_count("sellers") == before + 1
        assert sellers.get("S-DUP")["seller_name"] == "Second"

    def test_upsert_requires_a_natural_key(self, sellers):
        with pytest.raises(DatabaseError, match="no 'seller_id'"):
            sellers.upsert({"seller_name": "Anonymous"})

    def test_update_cannot_change_the_primary_key(self, sellers):
        sellers.upsert({"seller_id": "S-PK", "seller_name": "Fixed"})
        sellers.update("S-PK", {"seller_id": "S-MOVED", "zone": "Porur"})
        assert sellers.get("S-PK")["zone"] == "Porur"
        assert sellers.get("S-MOVED") is None

    def test_daily_stat_upsert_corrects_not_duplicates(self, db, sales):
        """Regression: the natural key is (stat_date, product_name).

        A duplicated product-day is the exact defect Sprint 1 cleaning removes;
        the write path must not put one back.
        """
        before = db.row_count("daily_item_stats")
        sales.upsert_stat({"stat_date": "2026-06-01", "product_name": "Tomato",
                           "units_sold": 999.0, "category": "Vegetables"})
        assert db.row_count("daily_item_stats") == before
        assert sales.get("2026-06-01", "Tomato")["units_sold"] == 999.0

    def test_daily_stat_requires_both_key_parts(self, sales):
        with pytest.raises(DatabaseError, match="natural key"):
            sales.upsert_stat({"units_sold": 5.0})

    def test_daily_stat_delete(self, sales):
        assert sales.delete_stat("2026-06-01", "Tomato") == 1
        assert sales.get("2026-06-01", "Tomato") is None

    def test_document_and_chunk_lifecycle(self, documents):
        document_id = documents.add_document(source="test", title="Storage Guide")
        documents.add_chunks([
            {"document_id": document_id, "chunk_index": 0, "section": "Dairy",
             "text": "Paneer must be held between 2 and 4 degrees.",
             "token_count": 9},
        ])
        assert documents.stats() == {"documents": 1, "chunks": 1}
        assert not documents.chunks(document_id).empty

        found = documents.search("paneer")
        assert len(found) == 1 and found.title[0] == "Storage Guide"
        assert not documents.search("Storage").empty      # title match

        documents.delete_document(document_id)
        assert documents.stats()["documents"] == 0

    def test_chat_lifecycle(self, chat):
        chat.append(session_id="S1", role="user", content="hello")
        chat.append(session_id="S1", role="assistant", content="hi",
                    latency_ms=12, sources=["doc"])
        chat.append(session_id="S2", role="user", content="other")

        assert len(chat.history()) == 3
        assert len(chat.history("S1")) == 2
        assert list(chat.conversation("S1").content) == ["hello", "hi"]
        assert len(chat.sessions()) == 2

        assert chat.delete_session("S2") == 1
        chat.clear()
        assert chat.count() == 0

    def test_metrics_lifecycle(self, metrics):
        metrics.record("forecaster", {"mape": 8.1, "rmse": 18.3,
                                      "algorithm": "ridge"})
        assert metrics.count() == 2          # the string is skipped, not coerced
        assert not metrics.latest().empty
        assert len(metrics.for_model("forecaster")) == 2
        assert len(metrics.history("forecaster", "mape")) == 1
        assert len(metrics.latest_per_metric("forecaster")) == 2
        assert metrics.delete_for_model("forecaster") == 2

    def test_metrics_skips_non_numeric_values(self, metrics):
        """A metric column that accepted a string would break every aggregate."""
        assert metrics.record("m", {"note": "text", "flag": True}) == 0

    def test_business_metric_upsert(self, metrics):
        metrics.record_business_metric("2026-06-01", "gmv", 1000.0)
        metrics.record_business_metric("2026-06-01", "gmv", 2000.0)
        frame = metrics.business_metrics("gmv")
        assert len(frame) == 1 and frame.metric_value[0] == 2000.0

    def test_recommendation_lifecycle(self, db, recommendations):
        recommendations.save_many([{
            "rec_type": "inventory", "entity_type": "batch",
            "entity_id": "BATCH001", "score": 0.9, "rank": 1,
            "rationale": "test", "algorithm": "unit-test",
        }])
        rec_id = int(db.scalar(
            "SELECT MAX(recommendation_id) FROM recommendations"))

        recommendations.dismiss(rec_id, reason="declined")
        assert "dismissed" in db.query_one(
            "SELECT rationale FROM recommendations WHERE recommendation_id = ?",
            (rec_id,))["rationale"]

        recommendations.accept(rec_id)
        assert recommendations.acceptance_rate() == 100.0
        assert recommendations.delete(rec_id) == 1

    def test_prune_keeps_accepted_recommendations(self, db, recommendations):
        """Accepted rows are the outcome record monitoring scores against.

        Pruning them would inflate the acceptance rate of whatever remains.
        """
        recommendations.save_many([
            {"rec_type": "inventory", "entity_type": "batch", "entity_id": "A",
             "score": 0.5, "rank": 1},
            {"rec_type": "inventory", "entity_type": "batch", "entity_id": "B",
             "score": 0.5, "rank": 2},
        ])
        ids = db.query("SELECT recommendation_id FROM recommendations "
                       "ORDER BY recommendation_id").recommendation_id.tolist()
        recommendations.accept(int(ids[0]))
        db.execute("UPDATE recommendations SET shown_at = "
                   "datetime('now', '-90 days')")

        recommendations.prune(keep_days=30)
        survivors = db.query("SELECT accepted_at FROM recommendations")
        assert len(survivors) == 1 and survivors.accepted_at.notna().all()


# ══════════════════════════════════════════════════════════════════════════
class TestSearchFilters:
    """Filters restrict, and restrict correctly."""

    def test_search_by_zone(self, inventory):
        rows = inventory.search(zone="Adyar")
        assert not rows.empty and (rows.zone == "Adyar").all()

    def test_search_by_product_is_case_insensitive(self, inventory):
        assert len(inventory.search(product="TOMATO")) == len(
            inventory.search(product="tomato"))

    def test_search_combines_filters(self, inventory):
        broad = len(inventory.search(zone="Adyar"))
        narrow = len(inventory.search(zone="Adyar", product="Tomato"))
        assert narrow <= broad

    def test_search_active_only_excludes_expired_and_sold_out(self, inventory):
        ids = set(inventory.search().batch_id)
        assert "BATCH-EXP" not in ids and "BATCH-OUT" not in ids

    def test_expiring_products_excludes_already_expired(self, inventory):
        """Regression: the query had no lower bound, so it mixed stock that can
        still be sold with stock that must be written off."""
        rows = inventory.expiring_products(days=2)
        assert (rows.days_to_expiry >= 0).all()
        assert "BATCH-EXP" not in set(rows.batch_id)

    def test_low_stock_excludes_sold_out(self, inventory):
        """Regression: a sold-out batch sits below every threshold, so it
        swamped the result with rows needing no action."""
        rows = inventory.low_stock(threshold=100)
        assert (rows.quantity_available > 0).all()
        assert "BATCH-OUT" not in set(rows.batch_id)

    def test_expired_returns_only_expired(self, inventory):
        rows = inventory.expired()
        assert (rows.days_to_expiry < 0).all()
        assert "BATCH-EXP" in set(rows.batch_id)

    def test_order_filters(self, orders):
        assert (orders.by_buyer("BUY001").buyer_id == "BUY001").all()
        assert (orders.by_seller("SEL001").seller_id == "SEL001").all()

    def test_sales_series_filters_by_product(self, sales):
        assert (sales.series(product="Tomato").product_name == "Tomato").all()

    def test_sales_series_window_is_relative_to_the_data(self, sales):
        """Regression: a window measured from today returns nothing for a
        dataset loaded weeks after it was captured."""
        assert not sales.series(days=3).empty

    def test_dimension_search(self, buyers, sellers):
        assert (buyers.search(zone="Adyar").zone == "Adyar").all()
        assert not sellers.search(name_like="store").empty

    def test_verified_sellers_only(self, sellers):
        assert (sellers.verified().fssai_verified == 1).all()


# ══════════════════════════════════════════════════════════════════════════
class TestPagination:
    """A page must report the total it was drawn from."""

    def test_page_reports_total_not_page_size(self, inventory):
        """Regression: rows without a total make a paged view lie — fifty of six
        hundred looks identical to all fifty that exist."""
        page = inventory.search_page(page=1, page_size=5)
        assert isinstance(page, Page)
        assert len(page.rows) == 5
        assert page.total == inventory.count_matching()
        assert page.total > 5 and page.is_truncated

    def test_pages_do_not_overlap_and_cover_everything(self, inventory):
        first = inventory.search_page(page=1, page_size=5)
        seen: set[str] = set()
        for number in range(1, first.pages + 1):
            rows = inventory.search_page(page=number, page_size=5).rows
            ids = set(rows.batch_id)
            assert not (ids & seen), f"page {number} overlaps an earlier page"
            seen |= ids
        assert len(seen) == first.total

    def test_count_matches_the_filter_it_pages(self, inventory):
        for filters in ({}, {"zone": "Adyar"}, {"product": "Tomato"},
                        {"grade": "C"}, {"max_days_to_expiry": 2}):
            page = inventory.search_page(page_size=2, **filters)
            assert page.total == inventory.count_matching(**filters)

    def test_page_beyond_the_end_is_empty_not_an_error(self, inventory):
        page = inventory.search_page(page=999, page_size=5)
        assert page.rows.empty and page.total > 0 and not page.has_next

    def test_page_navigation_flags(self, inventory):
        first = inventory.search_page(page=1, page_size=5)
        assert first.has_next and not first.has_previous
        last = inventory.search_page(page=first.pages, page_size=5)
        assert last.has_previous and not last.has_next

    def test_page_size_is_capped(self, inventory):
        assert inventory.search_page(page_size=10_000).page_size == 500

    def test_page_number_is_floored(self, inventory):
        assert inventory.search_page(page=0).page == 1

    @pytest.mark.parametrize("repository_fixture,default_order", [
        ("inventory", "action_priority DESC"),
        ("sales", "stat_date DESC"),
        ("orders", "order_date DESC"),
        ("buyers", "buyer_name ASC"),
        ("sellers", "seller_rating DESC"),
    ])
    def test_order_by_is_whitelisted(self, request, repository_fixture,
                                     default_order):
        """ORDER BY cannot be parameterised, so it is interpolated into SQL and
        must never accept unchecked input."""
        repository = request.getfixturevalue(repository_fixture)
        assert not repository.search_page(order_by=default_order).rows.empty
        with pytest.raises(ValueError, match="order_by must be one of"):
            repository.search_page(order_by="1; DROP TABLE batches")

    def test_paginated_dimensions(self, buyers, sellers):
        assert buyers.search_page(page_size=2).total == 4
        assert sellers.search_page(page_size=2).total == 5

    def test_page_summary_is_readable(self, inventory):
        assert "of" in inventory.search_page(page=1, page_size=5).summary()


# ══════════════════════════════════════════════════════════════════════════
class TestTransactions:
    """Read-then-write is atomic and never loses stock."""

    def test_adjust_quantity_is_atomic(self, inventory):
        """Regression: read and write used separate connections.

        Two concurrent sales could both read the old value and both write the
        same new one. Under Streamlit, which re-runs the script on every
        interaction, concurrent writers are normal rather than exceptional.
        """
        start = float(inventory.get("BATCH001")["quantity_available"])
        for _ in range(5):
            inventory.adjust_quantity("BATCH001", -1.0)
        assert inventory.get("BATCH001")["quantity_available"] == start - 5

    def test_adjust_quantity_never_goes_negative(self, inventory):
        inventory.adjust_quantity("BATCH001", -1_000_000.0)
        batch = inventory.get("BATCH001")
        assert batch["quantity_available"] == 0.0
        assert batch["status"] == "sold_out"

    def test_adjust_quantity_preserves_expired_status(self, inventory):
        """Selling stock down does not make expired stock fresh again."""
        inventory.adjust_quantity("BATCH-EXP", -1.0)
        assert inventory.get("BATCH-EXP")["status"] == "expired"

    def test_adjust_quantity_rejects_unknown_batch(self, inventory):
        with pytest.raises(DatabaseError, match="no batch"):
            inventory.adjust_quantity("NO-SUCH-BATCH", -1.0)

    def test_cancel_returns_stock_and_keeps_history(self, db, orders):
        """Cancellation is not deletion. The order is a historical fact the
        recommender learned from and monitoring scored."""
        order = orders.get("ORD001")
        before = float(db.scalar(
            "SELECT quantity_available FROM batches WHERE batch_id = ?",
            (order["batch_id"],)))

        restored = orders.cancel("ORD001", reason="test")

        assert restored == before + float(order["quantity"])
        assert orders.get("ORD001")["payment_status"] == "Cancelled"

    def test_cancel_is_idempotent(self, orders):
        orders.cancel("ORD001")
        assert orders.cancel("ORD001") == 0.0

    def test_cancel_handles_a_null_batch_reference(self, orders):
        """An order whose batch was never listed is still a real order; there is
        simply no stock to restore."""
        assert orders.cancel("ORD-NOBATCH") == 0.0
        assert orders.get("ORD-NOBATCH")["payment_status"] == "Cancelled"

    def test_cancel_rejects_unknown_order(self, orders):
        with pytest.raises(DatabaseError, match="no order"):
            orders.cancel("NO-SUCH-ORDER")


# ══════════════════════════════════════════════════════════════════════════
class TestForeignKeys:
    """Writes cannot orphan a row."""

    def test_delete_seller_refuses_while_referenced(self, sellers):
        """A bare FOREIGN KEY error names neither the seller nor the blocker."""
        with pytest.raises(DatabaseError, match="still reference it"):
            sellers.delete("SEL001")
        assert sellers.get("SEL001") is not None

    def test_delete_buyer_refuses_while_referenced(self, buyers):
        with pytest.raises(DatabaseError, match="still reference it"):
            buyers.delete("BUY001")

    def test_force_delete_cascades_and_leaves_no_orphan(self, db, sellers):
        sellers.delete("SEL001", force=True)
        assert db.scalar(
            "SELECT COUNT(*) FROM batches WHERE seller_id = 'SEL001'") == 0
        assert db.scalar(
            "SELECT COUNT(*) FROM orders WHERE seller_id = 'SEL001'") == 0
        assert sellers.get("SEL001") is None

    def test_delete_batch_refuses_while_ordered(self, inventory):
        with pytest.raises(DatabaseError, match="still reference it"):
            inventory.delete("BATCH001")

    def test_force_delete_batch_cascades(self, db, inventory):
        inventory.delete("BATCH001", force=True)
        assert db.scalar(
            "SELECT COUNT(*) FROM orders WHERE batch_id = 'BATCH001'") == 0

    def test_dependants_reports_blockers(self, sellers):
        blocking = sellers.dependants("SEL001")
        assert blocking["batches"] > 0 or blocking["orders"] > 0

    def test_no_orphans_in_the_seeded_database(self, orders):
        check = orders.orphan_check()
        assert (check.status == "ok").all(), check.to_string(index=False)

    def test_record_outcome_rejects_unknown_prediction(self, predictions):
        """Regression: a null predicted value silently became 0.0, recording an
        unscoreable prediction as a maximal miss."""
        with pytest.raises(DatabaseError, match="no such prediction"):
            predictions.record_outcome(999_999, 1.0)

    def test_prediction_outcome_round_trip(self, predictions):
        prediction_id = predictions.record(
            entity_type="batch", entity_id="BATCH001",
            prediction_type="spoilage_risk", value=0.8, model_name="test")
        assert len(predictions.unscored()) == 1

        predictions.record_outcome(prediction_id, 1.0)
        outcomes = predictions.outcomes("spoilage_risk")
        assert len(outcomes) == 1
        assert outcomes.predicted[0] == 0.8
        assert round(float(outcomes.error[0]), 3) == 0.2
        assert predictions.unscored().empty

    def test_document_delete_cascades_to_chunks(self, db, documents):
        document_id = documents.add_document(source="s", title="t")
        documents.add_chunks([{"document_id": document_id, "chunk_index": 0,
                               "text": "body"}])
        documents.delete_document(document_id)
        assert db.row_count("chunks") == 0


# ══════════════════════════════════════════════════════════════════════════
class TestBusinessQueries:
    """Aggregates agree with raw SQL."""

    def test_kpi_summary_matches_raw_sql(self, db, inventory):
        kpis = inventory.kpi_summary()
        assert int(kpis["total_batches"]) == int(db.scalar(
            "SELECT COUNT(*) FROM batches WHERE status = 'active' "
            "AND quantity_available > 0"))

    def test_kpi_summary_sees_expired_batches(self, inventory):
        """Regression: the active-only query cannot see expired stock, so
        reporting a confident zero would hide the write-off."""
        assert inventory.kpi_summary()["expired_count"] == 1

    def test_category_and_zone_totals_reconcile(self, inventory):
        total = inventory.kpi_summary()["inventory_value"]
        assert round(inventory.by_category().value.sum(), 2) == round(total, 2)
        assert round(inventory.by_zone().value.sum(), 2) == round(total, 2)

    def test_order_gmv_matches_raw_aggregate(self, db, orders):
        kpis = orders.business_kpis(days=36_500)
        expected = float(db.scalar(
            "SELECT COALESCE(SUM(order_value), 0) FROM orders"))
        assert round(kpis["gmv"], 2) == round(expected, 2)

    def test_business_kpis_reports_whether_the_window_applied(self, orders):
        """Regression: an empty window silently returned all-time figures.

        All-time numbers under a seven-day label overstate recent performance,
        and nothing in the response revealed the substitution.
        """
        assert orders.business_kpis(days=36_500)["window_applied"] == 1.0
        assert orders.business_kpis(days=0)["window_applied"] == 0.0

    def test_daily_revenue_view(self, orders):
        assert not orders.daily_revenue().empty

    def test_sales_coverage_detects_a_complete_grid(self, sales):
        coverage = sales.coverage()
        assert coverage["is_complete_grid"] is True
        assert coverage["expected_rows"] == coverage["rows"] == 30

    def test_sales_coverage_detects_a_gap(self, sales):
        """A model fitted on a series with silent gaps mis-attributes weekly
        seasonality, so the gap must be visible before training."""
        sales.delete_stat("2026-06-05", "Tomato")
        assert sales.coverage()["is_complete_grid"] is False

    def test_buyer_segmentation(self, buyers):
        assert set(buyers.by_type().buyer_type) == {"Restaurant", "Caterer"}
        assert not buyers.top_by_spend().empty

    def test_seller_performance_join(self, sellers):
        frame = sellers.performance()
        assert len(frame) == 5 and frame.active_batches.sum() > 0

    def test_top_products_ordered_by_revenue(self, sales):
        revenue = sales.top_products().revenue.tolist()
        assert revenue == sorted(revenue, reverse=True)


# ══════════════════════════════════════════════════════════════════════════
class TestIntegration:
    """Repositories agree with each other and with the schema's own links."""

    def test_batches_reference_only_known_sellers(self, db, inventory, sellers):
        referenced = set(db.query(
            "SELECT DISTINCT seller_id FROM batches").seller_id)
        assert referenced <= set(sellers.load_dataframe().seller_id)

    def test_item_foreign_key_resolves(self, inventory):
        """A null item_id would leave the product catalogue unjoinable."""
        joined = inventory.with_item()
        assert not joined.empty and joined.item_id.notna().all()

    def test_interaction_matrix_covers_transacting_buyers(self, db, orders):
        expected = int(db.scalar(
            "SELECT COUNT(DISTINCT buyer_id) FROM orders"))
        assert len(orders.interaction_matrix()) == expected

    def test_inventory_and_sales_share_a_product_vocabulary(self, inventory,
                                                            sales):
        assert set(sales.products()) <= set(inventory.distinct("product_name"))

    def test_monitoring_round_trip(self, monitoring):
        monitoring.log_event(component="test", action="ping", latency_ms=5)
        monitoring.log_llm_call(provider="stub", model="x", latency_ms=3)
        monitoring.start_run("RUN-T", "a query")
        monitoring.log_steps("RUN-T", [{"agent_name": "planner",
                                        "action": "plan"}])
        monitoring.finish_run("RUN-T", final_output="done", latency_ms=9)

        assert not monitoring.events().empty
        assert not monitoring.component_stats().empty
        assert not monitoring.llm_stats().empty
        assert len(monitoring.steps("RUN-T")) == 1
        assert monitoring.runs().status[0] == "completed"
        assert not monitoring.agent_stats().empty

    def test_monitoring_prune(self, db, monitoring):
        monitoring.log_event(component="old", action="a")
        db.execute("UPDATE system_logs SET created_at = "
                   "datetime('now', '-200 days')")
        monitoring.log_event(component="new", action="b")
        assert monitoring.prune(keep_days=90) == 1
        assert monitoring.count() == 1

    def test_every_repository_shares_one_connection(self, db):
        """Injecting a Database must reach every repository, or a test would
        silently exercise the production file instead of its fixture."""
        for repository_class in ALL_REPOSITORIES:
            assert repository_class(db).db is db

    def test_base_repository_helpers(self, db):
        base = BaseRepository(db)
        assert base.query("SELECT 1 AS n").n[0] == 1
        assert base.query_one("SELECT 1 AS n")["n"] == 1
        assert base.scalar("SELECT 1") == 1