"""Sprint 5 — the recommendation agent.

Exposes the Sprint 3 :class:`RecommendationEngine` to the orchestration layer.
Every score, price and match here is produced by that engine; this module only
assembles its inputs, translates column vocabularies at the database boundary,
persists the output and renders a narrative.

Two boundary problems it owns:

**Column vocabulary.** The database is snake_case; the engine and the trained
models expect the Sprint 1 pipeline's title-case frame. Passing DB rows in
unmapped does not raise — pandas simply reports missing columns as absent and
the engine falls back to defaults, producing plausible but meaningless output.
:func:`to_pipeline_frame` makes the translation explicit and reuses the model
mapping already defined by the inventory agent rather than restating it.

**Missing forecasts.** Every action that needs forward demand degrades to the
stored daily average when the forecast artefact is absent. The planner's
recovery path sets ``use_historical_average`` explicitly for this reason, and
the narrative always states which basis was used — a recommendation whose
provenance is unclear is not actionable.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from freshsense.agents.base import AgentResult, BaseAgent
from freshsense.agents.forecast_agent import DB_TO_PIPELINE_SALES
from freshsense.agents.inventory_agent import DB_TO_MODEL_COLUMNS, to_model_frame
from freshsense.agents.state import AgentState, AgentTask
from freshsense.config import SETTINGS
from freshsense.db.repository import (InventoryRepository, OrderRepository,
                                      PartyRepository,
                                      RecommendationRepository,
                                      SalesRepository)
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)

#: Full database -> Sprint 1 pipeline column map. Extends the model-feature
#: subset the inventory agent already defines, so the shared entries have one
#: definition.
DB_TO_PIPELINE_COLUMNS: dict[str, str] = {
    **DB_TO_MODEL_COLUMNS,
    "seller_id": "Seller_ID",
    "seller_name": "Seller_Name",
    "product_name": "Product_Name",
    "category": "Category",
    "brand": "Brand",
    "zone": "Zone",
    "unit": "Unit",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "cost_price": "Cost_Price",
    "mrp": "Selling_Price",
    "discount_pct": "Discount_Pct",
    "effective_price": "Effective_Price",
    "quality_grade": "Quality_Grade",
    "status": "Status",
    "expiry_date": "Expiry_Date",
    "environment_stress_index": "Environment_Stress_Index",
    "action_priority": "Action_Priority",
    "inventory_value": "Inventory_Value",
    "capital_at_risk": "Capital_At_Risk",
    "projected_surplus": "Projected_Surplus",
    "waste_quantity": "Waste_Quantity",
}

#: Order-log columns the engine's collaborative and affinity models read.
DB_TO_PIPELINE_ORDERS: dict[str, str] = {
    "buyer_id": "Buyer_ID",
    "seller_id": "Seller_ID",
    "product_name": "Product_Name",
    "category": "Category",
    "quantity": "Quantity_Ordered",
    "order_value": "Order_Value_INR",
    "buyer_savings": "Buyer_Savings_INR",
    "order_date": "Order_Date",
    "distance_km": "Distance_Km",
    "quality_grade": "Quality_Grade",
}

#: Buyer and seller dimension columns.
DB_TO_PIPELINE_PARTIES: dict[str, str] = {
    "buyer_id": "Buyer_ID",
    "buyer_name": "Buyer_Name",
    "buyer_type": "Buyer_Type",
    "seller_id": "Seller_ID",
    "seller_name": "Seller_Name",
    "zone": "Zone",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "avg_order_value": "Avg_Order_Value",
    "price_sensitivity": "Price_Sensitivity",
    "max_distance_km": "Max_Distance_Km",
    "min_acceptable_grade": "Min_Acceptable_Grade",
    "accepts_split_order": "Accepts_Split_Order",
    "seller_rating": "Seller_Rating",
    "dispute_rate_pct": "Dispute_Rate_Pct",
    "fssai_verified": "FSSAI_Verified",
}


def to_pipeline_frame(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename database columns to the vocabulary the engine expects."""
    present = {db: pipe for db, pipe in mapping.items() if db in frame.columns}
    return frame.rename(columns=present)


class RecommendationAgent(BaseAgent):
    """Turns inventory state into priced, ranked, explained recommendations."""

    name = "recommendation"
    description = (
        "Recommends what to do with stock: pricing, buyer and seller matching, "
        "restocking quantities and a prioritised daily action list."
    )
    supported_actions = (
        "recommend_actions", "recommend_pricing", "recommend_buyers",
        "match_sellers", "recommend_restocking",
    )

    def __init__(
        self,
        *,
        inventory: InventoryRepository | None = None,
        parties: PartyRepository | None = None,
        orders: OrderRepository | None = None,
        sales: SalesRepository | None = None,
        recommendations: RecommendationRepository | None = None,
    ) -> None:
        super().__init__()
        self._inventory = inventory
        self._parties = parties
        self._orders = orders
        self._sales = sales
        self._recommendations = recommendations
        self._engine: Any = None
        self._spoilage: Any = None
        self._spoilage_loaded = False
        self.currency = SETTINGS.currency

    # ── lazily-resolved dependencies ──────────────────────────────────
    @property
    def inventory(self) -> InventoryRepository:
        if self._inventory is None:
            self._inventory = InventoryRepository()
        return self._inventory

    @property
    def parties(self) -> PartyRepository:
        if self._parties is None:
            self._parties = PartyRepository()
        return self._parties

    @property
    def orders(self) -> OrderRepository:
        if self._orders is None:
            self._orders = OrderRepository()
        return self._orders

    @property
    def sales(self) -> SalesRepository:
        if self._sales is None:
            self._sales = SalesRepository()
        return self._sales

    @property
    def recommendations(self) -> RecommendationRepository:
        if self._recommendations is None:
            self._recommendations = RecommendationRepository()
        return self._recommendations

    @property
    def engine(self) -> Any:
        """The Sprint 3 engine, constructed once against the order history.

        Deferred because building it fits a collaborative-filtering matrix over
        the whole order log — work an agent should not do merely to appear on a
        roster.
        """
        if self._engine is None:
            from freshsense.recommendation import RecommendationEngine

            order_log = to_pipeline_frame(
                self.orders.recent(limit=5000), DB_TO_PIPELINE_ORDERS
            )
            self._engine = RecommendationEngine(orders=order_log)
            LOG.info("Recommendation engine ready: %s", self._engine.summary())
        return self._engine

    @property
    def spoilage(self) -> Any:
        if not self._spoilage_loaded:
            self._spoilage_loaded = True
            try:
                from freshsense.models.spoilage import get_spoilage_model

                self._spoilage = get_spoilage_model()
            except Exception as exc:                     # pragma: no cover
                LOG.warning("Spoilage model unavailable: %s", exc)
                self._spoilage = None
        return self._spoilage

    # ── input assembly ────────────────────────────────────────────────
    def _batches(self, **filters: Any) -> pd.DataFrame:
        """Active batches in the engine's column vocabulary, seller-enriched."""
        frame = to_pipeline_frame(
            self.inventory.search(**filters), DB_TO_PIPELINE_COLUMNS
        )
        return self._attach_seller_attributes(frame)

    def _attach_seller_attributes(self, batches: pd.DataFrame) -> pd.DataFrame:
        """Join seller reputation onto the batch frame.

        The ``batches`` table stores no seller reputation — it is a property of
        the seller, not of a batch, and duplicating it per row would let the two
        drift apart. But the engine's reliability sub-score reads
        ``Seller_Rating`` off each candidate row, and its column lookups supply
        a scalar default when the column is missing rather than a Series, which
        fails downstream. Joining it here keeps normalisation intact while
        giving the engine exactly the frame shape it expects.
        """
        if batches.empty or "Seller_ID" not in batches.columns:
            return batches

        sellers = to_pipeline_frame(self.parties.sellers(), DB_TO_PIPELINE_PARTIES)
        if sellers.empty:
            defaults = {"Seller_Rating": 3.5, "Dispute_Rate_Pct": 10.0,
                        "FSSAI_Verified": 0}
            for column, value in defaults.items():
                if column not in batches.columns:
                    batches[column] = value
            return batches

        columns = [
            c for c in ("Seller_ID", "Seller_Rating", "Dispute_Rate_Pct",
                        "FSSAI_Verified", "Seller_Name")
            if c in sellers.columns
        ]
        merged = batches.merge(
            sellers[columns], on="Seller_ID", how="left", suffixes=("", "_seller")
        )
        # A batch from an unregistered seller gets neutral reputation rather
        # than a null that would silently zero the reliability sub-score.
        merged["Seller_Rating"] = pd.to_numeric(
            merged.get("Seller_Rating"), errors="coerce"
        ).fillna(3.5)
        merged["Dispute_Rate_Pct"] = pd.to_numeric(
            merged.get("Dispute_Rate_Pct"), errors="coerce"
        ).fillna(10.0)
        merged["FSSAI_Verified"] = pd.to_numeric(
            merged.get("FSSAI_Verified"), errors="coerce"
        ).fillna(0).astype(int)
        return merged

    def _risk_map(self, batches: pd.DataFrame) -> dict[str, float]:
        """Modelled spoilage risk keyed by batch id, empty when unavailable.

        The engine already falls back to ``Environment_Stress_Index`` per row
        when a batch is absent from this map, so returning an empty dict is a
        correct degradation rather than a failure.
        """
        model = self.spoilage
        if model is None or batches.empty:
            return {}
        try:
            scores = model.predict_proba(to_model_frame(batches))
            return {
                str(bid): float(score)
                for bid, score in zip(batches["Inventory_ID"], scores)
            }
        except Exception as exc:
            LOG.warning("Risk scoring failed (%s); the engine will use the "
                        "stress index", exc)
            return {}

    def _forecast_map(
        self, batches: pd.DataFrame, *, use_historical_average: bool = False
    ) -> tuple[dict[str, float], str]:
        """Two-day demand per product, plus the basis actually used.

        Returns ``(mapping, basis)`` where basis is ``"forecast model"`` or
        ``"historical daily average"``. The caller states the basis in its
        narrative, so a reader always knows what the numbers rest on.
        """
        if not use_historical_average:
            try:
                from freshsense.models.forecasting import get_forecast_service

                service = get_forecast_service()
                if service.is_ready:
                    series = to_pipeline_frame(
                        self.sales.series(), DB_TO_PIPELINE_SALES
                    )
                    if not series.empty:
                        frame = service.forecast_all(series, horizon=2)
                        if not frame.empty:
                            return (
                                dict(zip(frame["product_name"],
                                         frame["forecast_units"])),
                                "forecast model",
                            )
            except Exception as exc:
                LOG.warning("Forecast unavailable (%s); using historical averages",
                            exc)

        if batches.empty or "Daily_Avg_Sales" not in batches.columns:
            return {}, "historical daily average"
        averages = (
            batches.groupby("Product_Name")["Daily_Avg_Sales"].mean() * 2
        ).round(1)
        return averages.to_dict(), "historical daily average"

    def _persist(self, recommendations: list[Any]) -> None:
        """Store recommendations so acceptance rate becomes measurable."""
        if not recommendations:
            return
        try:
            self.recommendations.save_many([r.as_dict() for r in recommendations])
        except Exception as exc:                         # pragma: no cover
            LOG.debug("Could not persist recommendations: %s", exc)

    def _money(self, value: float) -> str:
        return f"{self.currency}{float(value):,.0f}"

    def _resolve_buyer(
        self, buyer_id: str | None, zone: str | None, *, relax: bool = False
    ) -> dict[str, Any]:
        """Load a buyer profile, or synthesise one anchored on a zone.

        A seller search issued from the operator console has no buyer behind it,
        so a neutral profile is constructed from the zone centroid. It is
        reported as synthetic in the narrative rather than passed off as a real
        buyer's stated constraints.
        """
        if buyer_id:
            record = self.parties.buyer(str(buyer_id))
            if record:
                profile = {
                    DB_TO_PIPELINE_PARTIES.get(k, k): v for k, v in record.items()
                }
                if relax:
                    profile["Max_Distance_Km"] = float(
                        SETTINGS.recommendation.get("max_distance_km", 15)
                    )
                    profile["Min_Acceptable_Grade"] = "C"
                return profile

        latitude, longitude = 13.0827, 80.2707          # Chennai centre
        if zone:
            centroid = self.inventory.search(zone=zone, active_only=True)
            if not centroid.empty and "latitude" in centroid.columns:
                latitude = float(centroid["latitude"].mean())
                longitude = float(centroid["longitude"].mean())

        return {
            "Buyer_ID": "SYNTHETIC",
            "Buyer_Name": f"Operator search{f' ({zone})' if zone else ''}",
            "Buyer_Type": "Operator",
            "Zone": zone or "Chennai",
            "Latitude": latitude,
            "Longitude": longitude,
            "Max_Distance_Km": float(
                SETTINGS.recommendation.get("max_distance_km", 15)
            ) * (2.0 if relax else 1.0),
            "Min_Acceptable_Grade": "C",
            "Accepts_Split_Order": 1,
            "Avg_Order_Value": 0.0,
            "Price_Sensitivity": 0.5,
        }

    # ── actions ───────────────────────────────────────────────────────
    def action_recommend_actions(
        self, state: AgentState, task: AgentTask
    ) -> AgentResult:
        """Prioritised action list: what to do with which batch, and why."""
        top_n = int(task.param("top_n", SETTINGS.recommendation.get("top_n", 8)))
        use_average = bool(task.param("use_historical_average", False))

        # Prefer the batches an earlier agent already surfaced, so the two
        # agents cannot disagree about which stock is under discussion.
        batch_ids = list(state.fact("at_risk_batch_ids", []) or [])
        batches = self._batches(zone=task.param("zone"),
                                product=task.param("product"))
        if batch_ids and not batches.empty:
            focused = batches[batches["Inventory_ID"].astype(str).isin(batch_ids)]
            if not focused.empty:
                batches = focused

        if batches.empty:
            return self.empty("There is no active stock to recommend actions for.",
                              tool_called="RecommendationEngine.inventory")

        risks = self._risk_map(batches)
        forecasts, basis = self._forecast_map(
            batches, use_historical_average=use_average
        )

        results = self.engine.inventory.recommend(
            batches, risk_scores=risks, forecasts=forecasts, top_n=top_n
        )
        if not results:
            return self.empty("No batch currently warrants action.",
                              tool_called="RecommendationEngine.inventory")

        self._persist(results)

        top = results[0]
        counts: dict[str, int] = {}
        for item in results:
            action = str(item.payload.get("action", "monitor"))
            counts[action] = counts.get(action, 0) + 1
        mix = ", ".join(f"{n} × {a.replace('_', ' ')}"
                        for a, n in sorted(counts.items(), key=lambda kv: -kv[1]))
        exposure = sum(float(r.payload.get("capital_at_risk", 0)) for r in results)

        narrative = (
            f"{len(results)} prioritised action(s) covering {self._money(exposure)} "
            f"of capital ({mix}). Highest priority: "
            f"{top.payload.get('product_name')} — "
            f"{str(top.payload.get('action', '')).replace('_', ' ')} at "
            f"{top.payload.get('risk_band', 'unknown').lower()} risk. "
            f"{top.rationale} Demand basis: {basis}."
        )

        return self.ok(
            narrative,
            facts={
                "recommended_actions": counts,
                "top_action": top.payload.get("action"),
                "top_product": top.payload.get("product_name"),
                "recommendation_exposure": round(exposure, 2),
                "demand_basis": basis,
            },
            artifacts={"inventory_recommendations": results},
            tool_called="RecommendationEngine.inventory.recommend",
        )

    def action_recommend_pricing(
        self, state: AgentState, task: AgentTask
    ) -> AgentResult:
        """Decay-curve pricing adjusted for risk and surplus, floor enforced."""
        product = task.param("product")
        batch_ids = list(
            task.param("batch_ids")
            or state.fact("matched_batch_ids", [])
            or state.fact("at_risk_batch_ids", [])
            or []
        )
        top_n = int(task.param("top_n", 10))

        batches = self._batches(product=product, zone=task.param("zone"))
        if batch_ids and not batches.empty:
            focused = batches[batches["Inventory_ID"].astype(str).isin(batch_ids)]
            if not focused.empty:
                batches = focused

        if batches.empty:
            scope = f" for {product}" if product else ""
            return self.empty(f"No active stock found{scope} to price.",
                              tool_called="RecommendationEngine.pricing")

        risks = self._risk_map(batches)
        forecasts, basis = self._forecast_map(batches)

        priced: list[dict[str, Any]] = []
        for _, row in batches.iterrows():
            batch_id = str(row["Inventory_ID"])
            quote = self.engine.pricing.recommend(
                row,
                risk=risks.get(batch_id),
                forecast_demand_2d=forecasts.get(str(row["Product_Name"])),
            )
            priced.append({
                "batch_id": batch_id,
                "product_name": row["Product_Name"],
                "zone": row.get("Zone"),
                "quantity": float(row.get("Quantity_Available", 0)),
                "days_to_expiry": int(row.get("Days_To_Expiry", 0)),
                **quote,
            })

        frame = pd.DataFrame(priced)
        changes = frame[frame["pricing_action"] != "hold"]
        frame = frame.reindex(
            frame["discount_delta"].abs().sort_values(ascending=False).index
        ).head(top_n)

        if changes.empty:
            return self.ok(
                f"All {len(priced)} batch(es) are already priced correctly "
                f"against the decay curve; no change recommended. "
                f"Demand basis: {basis}.",
                facts={"pricing_changes": 0, "pricing_reviewed": len(priced)},
                artifacts={"pricing_quotes": frame},
                tool_called="RecommendationEngine.pricing.recommend",
            )

        uplift = float(
            (changes["recommended_price"] - changes["current_price"]).mul(
                changes["quantity"]
            ).sum()
        )
        headline = frame.iloc[0]

        narrative = (
            f"{len(changes)} of {len(priced)} batch(es) are mispriced against the "
            f"decay curve. Largest adjustment: {headline['product_name']} from "
            f"{headline['current_discount_pct']:.0f}% to "
            f"{headline['recommended_discount_pct']:.0f}% off "
            f"({self.currency}{headline['current_price']:.0f} → "
            f"{self.currency}{headline['recommended_price']:.0f}). "
            f"Net revenue effect across all changes: {self._money(uplift)}. "
            f"Demand basis: {basis}."
        )

        return self.ok(
            narrative,
            facts={
                "pricing_changes": int(len(changes)),
                "pricing_reviewed": len(priced),
                "pricing_revenue_effect": round(uplift, 2),
                "demand_basis": basis,
            },
            artifacts={"pricing_quotes": frame},
            tool_called="RecommendationEngine.pricing.recommend",
        )

    def action_recommend_buyers(
        self, state: AgentState, task: AgentTask
    ) -> AgentResult:
        """Rank buyers for a batch by proximity, purchase history and capacity."""
        product = task.param("product")
        batch_id = task.param("batch_id")
        top_n = int(task.param("top_n", 5))

        batches = self._batches(product=product, zone=task.param("zone"))
        if batches.empty:
            return self.empty(
                f"No active {product or 'stock'} to find buyers for.",
                tool_called="RecommendationEngine.buyer",
            )

        if batch_id:
            selected = batches[batches["Inventory_ID"].astype(str) == str(batch_id)]
            batch = selected.iloc[0] if not selected.empty else batches.iloc[0]
        else:
            # Default to the batch under most pressure — the one a human would
            # have picked anyway.
            batch = batches.sort_values("Action_Priority", ascending=False).iloc[0]

        buyers = to_pipeline_frame(self.parties.buyers(), DB_TO_PIPELINE_PARTIES)
        if buyers.empty:
            return self.empty("No buyers are registered on the platform.",
                              tool_called="PartyRepository.buyers")

        results = self.engine.buyer.recommend(batch, buyers, top_n=top_n)
        if not results:
            return self.empty(
                f"No buyer accepts grade {batch.get('Quality_Grade')} "
                f"{batch.get('Product_Name')} within their delivery radius.",
                tool_called="RecommendationEngine.buyer.recommend",
            )

        self._persist(results)

        top = results[0]
        repeat = sum(1 for r in results if r.payload.get("bought_before"))
        narrative = (
            f"{len(results)} buyer(s) match {batch.get('Quantity_Available'):.0f} "
            f"{batch.get('Unit', 'kg')} of grade {batch.get('Quality_Grade')} "
            f"{batch.get('Product_Name')} expiring in "
            f"{int(batch.get('Days_To_Expiry', 0))} day(s). "
            f"Best lead: {top.payload.get('buyer_name')} "
            f"({top.payload.get('buyer_type')}) — {top.rationale}. "
            f"{repeat} of them have bought this product before."
        )

        return self.ok(
            narrative,
            facts={
                "buyer_matches": len(results),
                "top_buyer_id": top.target_id,
                "top_buyer_name": top.payload.get("buyer_name"),
                "repeat_buyers": repeat,
                "sourced_batch_id": str(batch["Inventory_ID"]),
            },
            artifacts={"buyer_recommendations": results},
            tool_called="RecommendationEngine.buyer.recommend",
        )

    def action_match_sellers(
        self, state: AgentState, task: AgentTask
    ) -> AgentResult:
        """Find sellers who can fill a requirement, splitting the order if needed."""
        product = task.param("product") or state.fact("product")
        if not product:
            return self.fail(
                "no product specified — say which item you want to source"
            )

        quantity = float(task.param("quantity") or state.fact("quantity") or 10.0)
        zone = task.param("zone") or state.fact("zone")
        relax = bool(task.param("relax_constraints", False))

        buyer = self._resolve_buyer(task.param("buyer_id"), zone, relax=relax)
        inventory = self._batches()
        if inventory.empty:
            return self.empty("There is no active stock to match against.",
                              tool_called="RecommendationEngine.seller")

        sellers = to_pipeline_frame(self.parties.sellers(), DB_TO_PIPELINE_PARTIES)
        if weights := task.param("weights"):
            self.engine.set_weights(dict(weights))

        # The engine caps the search at min(buyer radius, configured ceiling),
        # so raising only the buyer's radius has no effect. Relaxation must lift
        # the ceiling too, or the planner's "widen the radius" recovery is inert.
        seller_config = self.engine.seller.config
        original_ceiling = seller_config.get("max_distance_km")
        if relax:
            seller_config["max_distance_km"] = float(original_ceiling or 15.0) * 2.5
        try:
            plan = self.engine.seller.fulfilment_plan(
                product=str(product), quantity=quantity, buyer=buyer,
                inventory=inventory, sellers=sellers,
            )
        finally:
            if relax:
                seller_config["max_distance_km"] = original_ceiling

        if plan["status"] == "unfilled":
            constraint = (
                f" within {buyer['Max_Distance_Km']:.0f} km of "
                f"{buyer.get('Zone', 'Chennai')}"
            )
            return self.empty(
                f"No seller can supply {quantity:.0f} unit(s) of {product}"
                f"{constraint} at grade "
                f"{buyer.get('Min_Acceptable_Grade', 'C')} or better.",
                tool_called="RecommendationEngine.seller.fulfilment_plan",
            )

        lines = plan["lines"]
        best = lines[0]
        synthetic = buyer["Buyer_ID"] == "SYNTHETIC"

        narrative = (
            f"{plan['status'].capitalize()} fill for {plan['filled']:.0f} of "
            f"{plan['requested']:.0f} unit(s) of {product} "
            f"across {plan['n_sellers']} seller(s) at {self._money(plan['cost'])}, "
            f"saving {self._money(plan['savings'])} against MRP. "
            f"Best line: {best['seller_name']} — {best['quantity']:.0f} "
            f"{best['unit']} at {self.currency}{best['unit_price']:.0f} "
            f"({best['discount_pct']:.0f}% off), {best['distance_km']:.1f} km away, "
            f"grade {best['quality_grade']}, {best['days_to_expiry']}d shelf life."
        )
        if plan["status"] == "partial":
            shortfall = plan["requested"] - plan["filled"]
            narrative += (
                f" {shortfall:.0f} unit(s) remain unfilled — no further seller "
                f"holds {product} within reach at an acceptable grade."
            )
        if relax:
            ceiling = float(original_ceiling or 15.0) * 2.5
            narrative += (
                f" Constraints were relaxed: search radius widened to "
                f"{ceiling:.0f} km and the grade floor dropped to C."
            )
        if synthetic:
            narrative += (
                f" Matched against a neutral profile centred on "
                f"{buyer.get('Zone')}, not a registered buyer's own constraints."
            )

        return self.ok(
            narrative,
            facts={
                "fulfilment_status": plan["status"],
                "quantity_filled": plan["filled"],
                "fulfilment_cost": plan["cost"],
                "fulfilment_savings": plan["savings"],
                "sellers_involved": plan["n_sellers"],
                "top_seller_id": best.get("seller_id"),
                "synthetic_buyer": synthetic,
            },
            artifacts={"fulfilment_plan": plan,
                       "seller_candidates": plan.get("candidates")},
            tool_called="RecommendationEngine.seller.fulfilment_plan",
        )

    def action_recommend_restocking(
        self, state: AgentState, task: AgentTask
    ) -> AgentResult:
        """Reorder quantities from forward demand against current cover."""
        cover_days = int(task.param("cover_days", 3))
        top_n = int(task.param("top_n", 8))

        batches = self._batches()
        if batches.empty:
            return self.empty("No inventory on record to compute cover against.",
                              tool_called="RecommendationEngine.restocking")

        forecasts, basis = self._forecast_map(batches)
        if not forecasts:
            return self.empty(
                "Neither a forecast nor a historical average is available, so "
                "reorder quantities cannot be computed.",
                tool_called="RecommendationEngine.restocking",
            )

        forecast_frame = pd.DataFrame(
            [{"product_name": k, "forecast_units": v} for k, v in forecasts.items()]
        )
        results = self.engine.restocking.recommend(
            batches, forecast_frame, cover_days=cover_days, top_n=top_n
        )
        if not results:
            return self.empty(
                f"Every product has at least {cover_days} day(s) of cover; "
                f"nothing needs reordering.",
                tool_called="RecommendationEngine.restocking.recommend",
            )

        self._persist(results)

        total_cost = sum(float(r.payload.get("estimated_cost", 0)) for r in results)
        top = results[0]
        high_waste = [
            r.entity_id for r in results
            if float(r.payload.get("waste_rate", 0)) > 0.15
        ]

        narrative = (
            f"{len(results)} product(s) need reordering for {cover_days}-day "
            f"cover, at an estimated {self._money(total_cost)}. "
            f"Most urgent: {top.entity_id} — order "
            f"{top.payload.get('reorder_qty'):.0f} "
            f"{top.payload.get('unit', 'kg')} against "
            f"{top.payload.get('days_of_cover'):.1f} day(s) of cover. "
            f"Demand basis: {basis}."
        )
        if high_waste:
            narrative += (
                f" Orders were reduced for {', '.join(high_waste[:3])} because "
                f"their recent waste rate is high — reordering into a spoilage "
                f"problem makes it worse."
            )

        return self.ok(
            narrative,
            facts={
                "restock_products": [r.entity_id for r in results],
                "restock_cost": round(total_cost, 2),
                "restock_count": len(results),
                "high_waste_products": high_waste,
                "demand_basis": basis,
            },
            artifacts={"restocking_recommendations": results},
            tool_called="RecommendationEngine.restocking.recommend",
        )


__all__ = [
    "RecommendationAgent", "to_pipeline_frame", "DB_TO_PIPELINE_COLUMNS",
    "DB_TO_PIPELINE_ORDERS", "DB_TO_PIPELINE_PARTIES",
]