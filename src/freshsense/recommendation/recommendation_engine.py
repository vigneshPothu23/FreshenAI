"""Sprint 3 — recommendation engine.

Five recommendation types share one transparent scoring core:

============  ==========================================================
Type          Question answered
============  ==========================================================
inventory     Which batches need action first, and what action?
buyer         Which buyers should be offered this batch?
seller        Which sellers can fill this buyer's requirement?
pricing       What price should this batch carry right now?
restocking    What should be reordered, and how much?
============  ==========================================================

Nothing here is a black box. The multi-criteria score decomposes into five
named sub-scores whose weights are configurable and exposed as live sliders in
the UI, and every recommendation carries a plain-language rationale.

A popularity baseline is included deliberately: a recommender that cannot beat
"suggest the most-ordered item" is not a recommender.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from freshsense.config import SETTINGS
from freshsense.data.features import (decay_discount, enforce_price_floor,
                                      haversine_km, risk_label)
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)

GRADE_RANK = {"A": 3, "B": 2, "C": 1}


# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Recommendation:
    """A single scored, explained recommendation."""

    rec_type: str
    entity_type: str
    entity_id: str
    score: float
    rank: int = 0
    rationale: str = ""
    target_type: str = ""
    target_id: str = ""
    algorithm: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rec_type": self.rec_type,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "score": round(float(self.score), 4),
            "rank": self.rank,
            "rationale": self.rationale,
            "algorithm": self.algorithm,
            "payload": self.payload,
        }


def _normalise_weights(weights: dict[str, float] | None) -> dict[str, float]:
    base = dict(SETTINGS.recommendation.get("weights", {}))
    merged = {**base, **(weights or {})}
    total = sum(merged.values()) or 1.0
    return {k: v / total for k, v in merged.items()}


# ══════════════════════════════════════════════════════════════════════════
class MatchingScorer:
    """The shared multi-criteria scoring core.

    Five sub-scores, each normalised to 0–1 with higher meaning better:

    * ``price``       — cheaper is better, min-max scaled within the candidate set
    * ``distance``    — closer is better, relative to the buyer's radius
    * ``freshness``   — more remaining shelf life is better
    * ``reliability`` — seller rating, dispute rate and FSSAI status combined
    * ``quantity``    — can the batch fill the requested amount in one pickup

    Min-max scaling of price is deliberate: what matters to a buyer is how a
    candidate compares to the alternatives actually on offer, not its absolute
    rupee value.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = _normalise_weights(weights)

    def score(
        self,
        candidates: pd.DataFrame,
        *,
        quantity_required: float,
        max_distance_km: float,
    ) -> pd.DataFrame:
        """Attach sub-scores, a composite score and a rationale to each candidate."""
        if candidates.empty:
            return candidates

        out = candidates.copy()
        price = pd.to_numeric(out["Effective_Price"], errors="coerce").fillna(0.0)
        span = max(float(price.max() - price.min()), 1e-9)

        out["s_price"] = (1 - (price - price.min()) / span).round(4)
        out["s_distance"] = (
            1 - pd.to_numeric(out.get("Distance_Km", 0), errors="coerce").fillna(0)
            / max(max_distance_km, 1e-9)
        ).clip(0, 1).round(4)
        out["s_freshness"] = (
            pd.to_numeric(out["Days_To_Expiry"], errors="coerce").fillna(0).clip(0, 5) / 5
        ).round(4)

        rating = pd.to_numeric(out.get("Seller_Rating", 3.5), errors="coerce").fillna(3.5)
        disputes = pd.to_numeric(out.get("Dispute_Rate_Pct", 10), errors="coerce").fillna(10)
        fssai = pd.to_numeric(out.get("FSSAI_Verified", 0), errors="coerce").fillna(0)
        out["s_reliability"] = (
            0.55 * ((rating - 2.5) / 2.5).clip(0, 1)
            + 0.30 * (1 - disputes / 35.0).clip(0, 1)
            + 0.15 * fssai.clip(0, 1)
        ).clip(0, 1).round(4)

        available = pd.to_numeric(out["Quantity_Available"], errors="coerce").fillna(0)
        out["s_quantity"] = np.where(
            available >= quantity_required, 1.0,
            available / max(quantity_required, 1e-9),
        ).round(4)

        w = self.weights
        out["match_score"] = (
            w["price"] * out["s_price"]
            + w["distance"] * out["s_distance"]
            + w["freshness"] * out["s_freshness"]
            + w["reliability"] * out["s_reliability"]
            + w["quantity"] * out["s_quantity"]
        ).round(4)

        out["why"] = [self._rationale(row) for _, row in out.iterrows()]
        return out.sort_values("match_score", ascending=False)

    def _rationale(self, row: pd.Series) -> str:
        """Name the two components that contributed most to this row's score."""
        w = self.weights
        contributions = {
            "price": w["price"] * row["s_price"],
            "distance": w["distance"] * row["s_distance"],
            "freshness": w["freshness"] * row["s_freshness"],
            "reliability": w["reliability"] * row["s_reliability"],
            "quantity": w["quantity"] * row["s_quantity"],
        }
        currency = SETTINGS.currency
        labels = {
            "price": f"{currency}{row['Effective_Price']:.0f}/{row.get('Unit', 'kg')} "
                     f"({row.get('Discount_Pct', 0):.0f}% off)",
            "distance": f"{row.get('Distance_Km', 0):.1f} km away",
            "freshness": f"{int(row['Days_To_Expiry'])}d shelf life left",
            "reliability": f"{float(row.get('Seller_Rating', 0)):.1f}★ seller",
            "quantity": f"{float(row['Quantity_Available']):.0f} "
                        f"{row.get('Unit', 'kg')} in stock",
        }
        top = sorted(contributions.items(), key=lambda kv: -kv[1])[:2]
        return " · ".join(labels[name] for name, _ in top)

    def breakdown(self, candidates: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """Long-form weighted contributions — feeds the score-decomposition chart."""
        columns = ["s_price", "s_distance", "s_freshness", "s_reliability", "s_quantity"]
        subset = candidates.head(top_n)
        if subset.empty:
            return pd.DataFrame(columns=["candidate", "component", "weighted"])

        rows: list[dict[str, Any]] = []
        for _, row in subset.iterrows():
            for column in columns:
                component = column.removeprefix("s_")
                rows.append({
                    "candidate": str(row.get("Seller_Name", row.get("Inventory_ID", "")))[:28],
                    "component": component,
                    "sub_score": float(row[column]),
                    "weighted": round(float(row[column]) * self.weights[component], 4),
                })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
class PricingRecommender:
    """Dynamic decay pricing, adjusted for risk and surplus pressure.

    The decay curve sets the base. Two adjustments follow, both bounded:
    modelled spoilage risk pushes the discount up, and surplus relative to
    forecast demand pushes it up further. Scarcity pulls it back down. The
    cost-price floor is always enforced last.
    """

    def __init__(self) -> None:
        self.config = SETTINGS.pricing

    def recommend(
        self,
        batch: pd.Series | dict[str, Any],
        *,
        risk: float | None = None,
        forecast_demand_2d: float | None = None,
    ) -> dict[str, Any]:
        """Return a priced recommendation with a full rationale trail."""
        get = batch.get
        mrp = float(get("Selling_Price") or get("mrp") or 0)
        cost = float(get("Cost_Price") or get("cost_price") or 0)
        quantity = float(get("Quantity_Available") or get("quantity_available") or 0)
        days_to_expiry = int(get("Days_To_Expiry") or get("days_to_expiry") or 0)
        shelf_life = int(get("Shelf_Life_Days") or get("shelf_life_days") or 5)
        current_discount = float(get("Discount_Pct") or get("discount_pct") or 0)
        current_price = float(get("Effective_Price") or get("effective_price") or mrp)

        if risk is None:
            risk = float(get("Environment_Stress_Index") or 0.3)
        if forecast_demand_2d is None:
            forecast_demand_2d = float(get("Expected_Demand_2D") or 0)

        base = decay_discount(days_to_expiry, shelf_life)
        reasons = [f"{days_to_expiry} day(s) of shelf life remaining "
                   f"→ base discount {base}%"]

        risk_adjustment = 12.0 * max(0.0, risk - 0.5) * 2
        if risk_adjustment > 1:
            reasons.append(f"spoilage risk {risk:.0%} → +{risk_adjustment:.0f}%")

        surplus_ratio = max(0.0, (quantity - forecast_demand_2d) / max(quantity, 1e-9))
        surplus_adjustment = 10.0 * surplus_ratio
        if surplus_adjustment > 1:
            reasons.append(f"stock exceeds 2-day forecast demand "
                           f"→ +{surplus_adjustment:.0f}%")

        scarcity_adjustment = -8.0 if forecast_demand_2d > quantity else 0.0
        if scarcity_adjustment:
            reasons.append("demand exceeds stock → discount reduced 8%")

        discount = float(np.clip(
            base + risk_adjustment + surplus_adjustment + scarcity_adjustment,
            float(self.config.get("min_discount", 5)),
            float(self.config.get("max_discount", 75)),
        ))
        price, discount = enforce_price_floor(mrp, cost, discount)
        if cost and price <= cost * float(self.config.get("cost_price_floor_ratio", 0.85)) + 0.01:
            reasons.append("capped at the cost-price floor")

        delta = discount - current_discount
        return {
            "current_price": round(current_price, 2),
            "current_discount_pct": round(current_discount, 1),
            "recommended_price": price,
            "recommended_discount_pct": round(discount, 1),
            "price_delta": round(price - current_price, 2),
            "discount_delta": round(delta, 1),
            "pricing_action": ("increase_discount" if delta > 4
                               else "reduce_discount" if delta < -4 else "hold"),
            "expected_revenue": round(price * min(quantity, max(forecast_demand_2d, 0) or quantity), 2),
            "rationale": "; ".join(reasons),
            "risk_band": risk_label(risk),
        }


# ══════════════════════════════════════════════════════════════════════════
class InventoryRecommender:
    """Ranks batches by urgency and prescribes a concrete next action."""

    def __init__(self, pricing: PricingRecommender | None = None) -> None:
        self.pricing = pricing or PricingRecommender()
        self.config = SETTINGS.recommendation

    def recommend(
        self,
        inventory: pd.DataFrame,
        *,
        risk_scores: dict[str, float] | None = None,
        forecasts: dict[str, float] | None = None,
        top_n: int | None = None,
    ) -> list[Recommendation]:
        """Produce a prioritised action list for the operator."""
        if inventory.empty:
            return []

        top_n = int(top_n or self.config.get("top_n", 8))
        risk_scores = risk_scores or {}
        forecasts = forecasts or {}
        recommendations: list[Recommendation] = []

        for _, row in inventory.iterrows():
            batch_id = str(row.get("Inventory_ID") or row.get("batch_id") or "")
            product = str(row.get("Product_Name") or row.get("product_name") or "")
            risk = float(risk_scores.get(batch_id, row.get("Environment_Stress_Index", 0.3)))
            demand = float(forecasts.get(product, row.get("Expected_Demand_2D", 0)))
            quantity = float(row.get("Quantity_Available", 0))
            days = int(row.get("Days_To_Expiry", 0))

            pricing = self.pricing.recommend(
                row, risk=risk, forecast_demand_2d=demand
            )
            surplus = max(0.0, quantity - demand)

            # Urgency blends modelled risk with time pressure and oversupply.
            urgency = float(np.clip(
                0.50 * risk
                + 0.30 * (1 - min(max(days, 0), 3) / 3)
                + 0.20 * (surplus / max(quantity, 1e-9)),
                0, 1,
            ))

            action, guidance = self._decide_action(
                risk=risk, days=days, surplus=surplus, quantity=quantity,
                discount_delta=pricing["discount_delta"],
                grade=str(row.get("Quality_Grade", "A")),
            )

            recommendations.append(Recommendation(
                rec_type="inventory",
                entity_type="batch",
                entity_id=batch_id,
                score=round(urgency, 4),
                rationale=guidance,
                algorithm="urgency_blend(risk, time_pressure, surplus)",
                payload={
                    "product_name": product,
                    "seller_id": row.get("Seller_ID"),
                    "zone": row.get("Zone"),
                    "quantity_available": round(quantity, 1),
                    "unit": row.get("Unit", "kg"),
                    "days_to_expiry": days,
                    "risk_score": round(risk, 4),
                    "risk_band": risk_label(risk),
                    "quality_grade": row.get("Quality_Grade"),
                    "forecast_demand_2d": round(demand, 1),
                    "projected_surplus": round(surplus, 1),
                    "capital_at_risk": round(float(row.get("Capital_At_Risk", 0)), 2),
                    **pricing,
                    "action": action,
                },
            ))

        recommendations.sort(key=lambda r: -r.score)
        for rank, recommendation in enumerate(recommendations[:top_n], start=1):
            recommendation.rank = rank
        return recommendations[:top_n]

    @staticmethod
    def _decide_action(
        *, risk: float, days: int, surplus: float, quantity: float,
        discount_delta: float, grade: str,
    ) -> tuple[str, str]:
        """Map the situation onto one concrete action plus its justification."""
        if days < 0:
            return ("dispose_or_donate",
                    "Past expiry. Route to composting or an NGO partner and record "
                    "the write-off so the waste analytics stay accurate.")
        if risk >= 0.70 or days == 0:
            return ("urgent_clearance",
                    f"Critical risk ({risk:.0%}) with {days} day(s) left. Apply the "
                    f"maximum discount now and push a notification to buyers who "
                    f"saved this store.")
        if grade == "C":
            return ("route_to_processing",
                    "Grade C stock suits caterers and processing kitchens rather "
                    "than retail-facing buyers. Target that segment directly.")
        if discount_delta > 4:
            return ("increase_discount",
                    f"Current pricing sits above the decay curve for {days} day(s) "
                    f"remaining. Raising the discount by {discount_delta:.0f} points "
                    f"restores the expected sell-through rate.")
        if surplus > 0 and quantity > 0 and surplus / quantity > 0.4:
            return ("list_surplus",
                    f"About {surplus:.0f} units will remain unsold at the current "
                    f"rate. List the surplus for bulk buyers before it ages further.")
        if quantity <= float(SETTINGS.features.get("low_stock_threshold", 5)):
            return ("restock",
                    "Stock is low and demand is steady. Reorder or mark sold out to "
                    "avoid failed pickups.")
        return ("monitor",
                f"Pricing and risk are both healthy at {risk:.0%}. No action needed "
                f"today; recheck tomorrow morning.")


# ══════════════════════════════════════════════════════════════════════════
class BuyerRecommender:
    """Matches a batch to the buyers most likely to take it.

    Combines constraint satisfaction (distance radius, minimum acceptable grade)
    with revealed preference from the order history. A buyer who has bought this
    product before is a materially stronger lead than one who has not.
    """

    def __init__(self, orders: pd.DataFrame | None = None) -> None:
        self.orders = orders if orders is not None else pd.DataFrame()
        self._affinity = self._build_affinity()

    def _build_affinity(self) -> pd.DataFrame:
        """Buyer × product purchase-quantity matrix from the order log."""
        if self.orders.empty or "Buyer_ID" not in self.orders.columns:
            return pd.DataFrame()
        return self.orders.pivot_table(
            index="Buyer_ID", columns="Product_Name",
            values="Quantity_Ordered", aggfunc="sum", fill_value=0,
        )

    def recommend(
        self,
        batch: pd.Series | dict[str, Any],
        buyers: pd.DataFrame,
        *,
        top_n: int = 5,
    ) -> list[Recommendation]:
        if buyers.empty:
            return []

        product = str(batch.get("Product_Name") or batch.get("product_name") or "")
        grade = str(batch.get("Quality_Grade") or "A")
        batch_lat = float(batch.get("Latitude") or 13.0827)
        batch_lon = float(batch.get("Longitude") or 80.2707)
        quantity = float(batch.get("Quantity_Available") or 0)
        batch_id = str(batch.get("Inventory_ID") or batch.get("batch_id") or "")

        candidates = buyers.copy()
        candidates["distance_km"] = haversine_km(
            batch_lat, batch_lon,
            candidates["Latitude"].to_numpy(), candidates["Longitude"].to_numpy(),
        ).round(2)

        # Hard constraints first — a buyer outside the radius or below the grade
        # threshold is not a weak match, it is not a match at all.
        radius = pd.to_numeric(candidates["Max_Distance_Km"], errors="coerce").fillna(10)
        candidates = candidates[candidates["distance_km"] <= radius]
        if candidates.empty:
            return []

        grade_ok = candidates["Min_Acceptable_Grade"].map(
            lambda g: GRADE_RANK.get(str(g), 1) <= GRADE_RANK.get(grade, 1)
        )
        candidates = candidates[grade_ok]
        if candidates.empty:
            return []

        max_radius = float(radius.max() or 10)
        proximity = (1 - candidates["distance_km"] / max_radius).clip(0, 1)

        if not self._affinity.empty and product in self._affinity.columns:
            history = self._affinity[product].reindex(candidates["Buyer_ID"]).fillna(0)
            affinity = (history / max(history.max(), 1e-9)).to_numpy()
        else:
            affinity = np.zeros(len(candidates))

        capacity = (
            pd.to_numeric(candidates["Avg_Order_Value"], errors="coerce").fillna(0)
            / max(float(candidates["Avg_Order_Value"].max() or 1), 1e-9)
        ).clip(0, 1)
        price_fit = pd.to_numeric(
            candidates["Price_Sensitivity"], errors="coerce"
        ).fillna(0.5)

        candidates["score"] = (
            0.35 * proximity.to_numpy()
            + 0.30 * affinity
            + 0.20 * capacity.to_numpy()
            + 0.15 * price_fit.to_numpy()
        ).round(4)

        candidates = candidates.sort_values("score", ascending=False).head(top_n)

        recommendations: list[Recommendation] = []
        for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
            bought_before = bool(
                not self._affinity.empty
                and product in self._affinity.columns
                and self._affinity.get(product, pd.Series(dtype=float))
                    .get(row["Buyer_ID"], 0) > 0
            )
            reasons = [f"{row['distance_km']:.1f} km away"]
            if bought_before:
                reasons.append(f"has ordered {product} before")
            reasons.append(f"{row.get('Buyer_Type', 'buyer')} segment")

            recommendations.append(Recommendation(
                rec_type="buyer",
                entity_type="batch",
                entity_id=batch_id,
                target_type="buyer",
                target_id=str(row["Buyer_ID"]),
                score=float(row["score"]),
                rank=rank,
                rationale=" · ".join(reasons),
                algorithm="proximity + purchase affinity + capacity + price fit",
                payload={
                    "buyer_name": row.get("Buyer_Name"),
                    "buyer_type": row.get("Buyer_Type"),
                    "zone": row.get("Zone"),
                    "distance_km": float(row["distance_km"]),
                    "avg_order_value": float(row.get("Avg_Order_Value", 0)),
                    "bought_before": bought_before,
                    "product_name": product,
                    "quantity_available": quantity,
                },
            ))
        return recommendations


# ══════════════════════════════════════════════════════════════════════════
class SellerRecommender:
    """Finds the best sellers to fill a buyer's stated requirement.

    Supports split fulfilment across sellers when a single seller cannot cover
    the requested quantity and the buyer permits it.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.scorer = MatchingScorer(weights)
        self.config = SETTINGS.recommendation

    def recommend(
        self,
        *,
        product: str,
        quantity: float,
        buyer: pd.Series | dict[str, Any],
        inventory: pd.DataFrame,
        sellers: pd.DataFrame | None = None,
        top_n: int = 5,
    ) -> tuple[list[Recommendation], pd.DataFrame]:
        """Return ranked seller recommendations plus the scored candidate frame."""
        if inventory.empty:
            return [], pd.DataFrame()

        max_distance = min(
            float(buyer.get("Max_Distance_Km") or 10),
            float(self.config.get("max_distance_km", 15)),
        )
        min_grade = str(buyer.get("Min_Acceptable_Grade") or "C")
        buyer_lat = float(buyer.get("Latitude") or 13.0827)
        buyer_lon = float(buyer.get("Longitude") or 80.2707)

        candidates = inventory[
            inventory["Product_Name"].astype(str).str.lower() == product.lower()
        ].copy()
        candidates = candidates[
            (candidates["Quantity_Available"] > 0)
            & (candidates["Days_To_Expiry"] >= 0)
            & (candidates.get("Status", "active") == "active")
        ]
        if candidates.empty:
            return [], pd.DataFrame()

        candidates = candidates[
            candidates["Quality_Grade"].map(lambda g: GRADE_RANK.get(str(g), 1))
            >= GRADE_RANK.get(min_grade, 1)
        ]
        if candidates.empty:
            return [], pd.DataFrame()

        candidates["Distance_Km"] = haversine_km(
            buyer_lat, buyer_lon,
            candidates["Latitude"].to_numpy(), candidates["Longitude"].to_numpy(),
        ).round(2)
        candidates = candidates[candidates["Distance_Km"] <= max_distance]
        if candidates.empty:
            return [], pd.DataFrame()

        if sellers is not None and not sellers.empty:
            columns = [c for c in ("Seller_ID", "Dispute_Rate_Pct", "FSSAI_Verified")
                       if c in sellers.columns]
            candidates = candidates.merge(sellers[columns], on="Seller_ID", how="left",
                                          suffixes=("", "_seller"))

        scored = self.scorer.score(
            candidates, quantity_required=quantity, max_distance_km=max_distance
        ).head(top_n)

        recommendations = [
            Recommendation(
                rec_type="seller",
                entity_type="buyer",
                entity_id=str(buyer.get("Buyer_ID", "")),
                target_type="batch",
                target_id=str(row.get("Inventory_ID", "")),
                score=float(row["match_score"]),
                rank=rank,
                rationale=str(row["why"]),
                algorithm="weighted multi-criteria (price, distance, freshness, "
                          "reliability, quantity)",
                payload={
                    "seller_id": row.get("Seller_ID"),
                    "seller_name": str(row.get("Seller_Name", "")).strip(),
                    "zone": row.get("Zone"),
                    "product_name": product,
                    "quantity_available": float(row["Quantity_Available"]),
                    "unit": row.get("Unit", "kg"),
                    "unit_price": float(row["Effective_Price"]),
                    "mrp": float(row.get("Selling_Price", 0)),
                    "discount_pct": float(row.get("Discount_Pct", 0)),
                    "distance_km": float(row["Distance_Km"]),
                    "days_to_expiry": int(row["Days_To_Expiry"]),
                    "quality_grade": row.get("Quality_Grade"),
                    "line_total": round(
                        float(row["Effective_Price"])
                        * min(quantity, float(row["Quantity_Available"])), 2
                    ),
                    "sub_scores": {
                        "price": float(row["s_price"]),
                        "distance": float(row["s_distance"]),
                        "freshness": float(row["s_freshness"]),
                        "reliability": float(row["s_reliability"]),
                        "quantity": float(row["s_quantity"]),
                    },
                },
            )
            for rank, (_, row) in enumerate(scored.iterrows(), start=1)
        ]
        return recommendations, scored

    def fulfilment_plan(
        self,
        *,
        product: str,
        quantity: float,
        buyer: pd.Series | dict[str, Any],
        inventory: pd.DataFrame,
        sellers: pd.DataFrame | None = None,
        allow_split: bool | None = None,
    ) -> dict[str, Any]:
        """Build a concrete fulfilment plan, splitting across sellers if permitted."""
        if allow_split is None:
            allow_split = bool(int(buyer.get("Accepts_Split_Order", 1) or 1))

        recommendations, scored = self.recommend(
            product=product, quantity=quantity, buyer=buyer,
            inventory=inventory, sellers=sellers, top_n=8,
        )
        if not recommendations:
            return {
                "product": product, "requested": quantity, "filled": 0.0,
                "status": "unfilled", "lines": [], "cost": 0.0, "savings": 0.0,
            }

        lines: list[dict[str, Any]] = []
        remaining = float(quantity)

        single = [r for r in recommendations
                  if r.payload["quantity_available"] >= quantity]
        if single:
            best = single[0]
            lines.append(self._line(best, quantity))
            remaining = 0.0
        elif allow_split:
            for recommendation in recommendations:
                if remaining <= 0:
                    break
                take = min(recommendation.payload["quantity_available"], remaining)
                if take <= 0:
                    continue
                lines.append(self._line(recommendation, take))
                remaining -= take
        else:
            best = recommendations[0]
            take = min(best.payload["quantity_available"], quantity)
            lines.append(self._line(best, take))
            remaining = quantity - take

        filled = sum(line["quantity"] for line in lines)
        cost = sum(line["line_total"] for line in lines)
        savings = sum(line["line_savings"] for line in lines)

        return {
            "product": product,
            "requested": round(float(quantity), 1),
            "filled": round(filled, 1),
            "status": "full" if remaining <= 1e-6 else ("partial" if filled else "unfilled"),
            "lines": lines,
            "cost": round(cost, 2),
            "savings": round(savings, 2),
            "n_sellers": len({line["seller_id"] for line in lines}),
            "candidates": scored,
        }

    @staticmethod
    def _line(recommendation: Recommendation, quantity: float) -> dict[str, Any]:
        payload = recommendation.payload
        unit_price = float(payload["unit_price"])
        mrp = float(payload.get("mrp") or unit_price)
        return {
            "batch_id": recommendation.target_id,
            "seller_id": payload.get("seller_id"),
            "seller_name": payload.get("seller_name"),
            "zone": payload.get("zone"),
            "product": payload.get("product_name"),
            "quantity": round(float(quantity), 1),
            "unit": payload.get("unit", "kg"),
            "unit_price": unit_price,
            "mrp": mrp,
            "discount_pct": float(payload.get("discount_pct", 0)),
            "line_total": round(unit_price * quantity, 2),
            "line_savings": round((mrp - unit_price) * quantity, 2),
            "distance_km": float(payload.get("distance_km", 0)),
            "days_to_expiry": int(payload.get("days_to_expiry", 0)),
            "quality_grade": payload.get("quality_grade"),
            "why": recommendation.rationale,
            "match_score": recommendation.score,
        }


# ══════════════════════════════════════════════════════════════════════════
class RestockingRecommender:
    """Recommends reorder quantities from forecast demand and current cover.

    Reorder quantity = forecast demand over the cover window × safety factor −
    stock on hand. Products whose recent waste rate is high are flagged for a
    reduced order rather than a larger one: reordering into a spoilage problem
    makes it worse.
    """

    def __init__(self) -> None:
        self.config = SETTINGS.recommendation
        self.safety_factor = float(self.config.get("restock_safety_factor", 1.15))

    def recommend(
        self,
        inventory: pd.DataFrame,
        forecasts: pd.DataFrame,
        *,
        cover_days: int = 3,
        top_n: int = 10,
    ) -> list[Recommendation]:
        if inventory.empty or forecasts.empty:
            return []

        on_hand = (
            inventory[inventory.get("Status", "active") == "active"]
            .groupby("Product_Name")
            .agg(
                quantity_on_hand=("Quantity_Available", "sum"),
                avg_days_to_expiry=("Days_To_Expiry", "mean"),
                waste_units=("Waste_Quantity", "sum"),
                unit=("Unit", "first"),
                avg_cost=("Cost_Price", "mean"),
            )
            .reset_index()
        )

        merged = on_hand.merge(
            forecasts.rename(columns={"product_name": "Product_Name"}),
            on="Product_Name", how="left",
        )
        merged["forecast_units"] = merged["forecast_units"].fillna(
            merged["quantity_on_hand"] * 0.3
        )

        # forecast_units is a 2-day figure; scale to the requested cover window.
        merged["cover_demand"] = merged["forecast_units"] / 2 * cover_days
        merged["target_stock"] = merged["cover_demand"] * self.safety_factor
        merged["reorder_qty"] = (
            merged["target_stock"] - merged["quantity_on_hand"]
        ).clip(lower=0).round(1)
        merged["days_of_cover"] = (
            merged["quantity_on_hand"] / (merged["forecast_units"] / 2).clip(lower=0.1)
        ).round(1)
        merged["waste_rate"] = (
            merged["waste_units"] / merged["quantity_on_hand"].clip(lower=0.1)
        ).round(3)

        merged = merged[merged["reorder_qty"] > 0].copy()
        if merged.empty:
            return []

        merged["urgency"] = (
            0.6 * (1 - (merged["days_of_cover"] / cover_days).clip(0, 1))
            + 0.4 * (merged["reorder_qty"] / merged["reorder_qty"].max()).clip(0, 1)
        ).round(4)
        merged = merged.sort_values("urgency", ascending=False).head(top_n)

        recommendations: list[Recommendation] = []
        for rank, (_, row) in enumerate(merged.iterrows(), start=1):
            high_waste = float(row["waste_rate"]) > 0.15
            reorder = float(row["reorder_qty"])
            if high_waste:
                reorder = round(reorder * 0.7, 1)
                note = (f"Waste rate is {row['waste_rate']:.0%} — order reduced 30% "
                        f"to avoid reordering into a spoilage problem.")
            else:
                note = (f"Only {row['days_of_cover']:.1f} day(s) of cover against a "
                        f"{cover_days}-day window.")

            recommendations.append(Recommendation(
                rec_type="restocking",
                entity_type="product",
                entity_id=str(row["Product_Name"]),
                score=float(row["urgency"]),
                rank=rank,
                rationale=note,
                algorithm=f"forecast x {self.safety_factor} safety - on-hand",
                payload={
                    "product_name": row["Product_Name"],
                    "unit": row.get("unit", "kg"),
                    "quantity_on_hand": round(float(row["quantity_on_hand"]), 1),
                    "forecast_demand": round(float(row["cover_demand"]), 1),
                    "reorder_qty": reorder,
                    "days_of_cover": float(row["days_of_cover"]),
                    "waste_rate": float(row["waste_rate"]),
                    "estimated_cost": round(reorder * float(row.get("avg_cost") or 0), 2),
                    "cover_days": cover_days,
                },
            ))
        return recommendations


# ══════════════════════════════════════════════════════════════════════════
class CollaborativeRecommender:
    """Item-item cosine similarity over the buyer × product matrix.

    Implicit feedback (purchase quantity), not ratings. Includes a popularity
    baseline so the lift from personalisation can be measured rather than
    assumed.
    """

    def __init__(self, orders: pd.DataFrame) -> None:
        self.orders = orders
        self.matrix = self._build_matrix()
        self.similarity = self._build_similarity()

    def _build_matrix(self) -> pd.DataFrame:
        if self.orders.empty or "Buyer_ID" not in self.orders.columns:
            return pd.DataFrame()
        return self.orders.pivot_table(
            index="Buyer_ID", columns="Product_Name",
            values="Quantity_Ordered", aggfunc="sum", fill_value=0,
        )

    def _build_similarity(self) -> pd.DataFrame:
        if self.matrix.empty:
            return pd.DataFrame()
        values = self.matrix.to_numpy(dtype=float)
        norms = np.linalg.norm(values, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        normalised = values / norms
        return pd.DataFrame(
            normalised.T @ normalised,
            index=self.matrix.columns, columns=self.matrix.columns,
        )

    def similar_products(self, product: str, top_n: int = 5) -> pd.DataFrame:
        """Products frequently bought by the same buyers."""
        if self.similarity.empty or product not in self.similarity.columns:
            return pd.DataFrame(columns=["product_name", "similarity"])
        series = (
            self.similarity[product].drop(labels=[product], errors="ignore")
            .sort_values(ascending=False).head(top_n)
        )
        return series.reset_index().set_axis(["product_name", "similarity"], axis=1)

    def for_buyer(self, buyer_id: str, top_n: int = 5) -> pd.DataFrame:
        """Products this buyer has not purchased, ranked by neighbourhood score."""
        if self.matrix.empty or buyer_id not in self.matrix.index:
            return pd.DataFrame(columns=["product_name", "score"])
        purchased = self.matrix.loc[buyer_id]
        scores = (
            self.similarity.mul(purchased, axis=0).sum()
            / self.similarity.sum().clip(lower=1e-9)
        )
        scores = scores[purchased == 0].sort_values(ascending=False).head(top_n)
        return scores.reset_index().set_axis(["product_name", "score"], axis=1)

    def popular_baseline(self, top_n: int = 5) -> pd.DataFrame:
        """The baseline every personalised recommender must beat."""
        if self.orders.empty:
            return pd.DataFrame(columns=["product_name", "total_quantity"])
        return (
            self.orders.groupby("Product_Name")["Quantity_Ordered"].sum()
            .sort_values(ascending=False).head(top_n)
            .reset_index().set_axis(["product_name", "total_quantity"], axis=1)
        )

    def popular_for_segment(self, buyer_type: str, top_n: int = 5) -> pd.DataFrame:
        """What comparable businesses order most."""
        if self.orders.empty or "Buyer_Type" not in self.orders.columns:
            return pd.DataFrame(columns=["product_name", "total_quantity"])
        subset = self.orders[self.orders["Buyer_Type"] == buyer_type]
        if subset.empty:
            return self.popular_baseline(top_n)
        return (
            subset.groupby("Product_Name")["Quantity_Ordered"].sum()
            .sort_values(ascending=False).head(top_n)
            .reset_index().set_axis(["product_name", "total_quantity"], axis=1)
        )


# ══════════════════════════════════════════════════════════════════════════
class RecommendationEngine:
    """Facade over every recommender. The agents and the UI use this only."""

    def __init__(
        self,
        *,
        orders: pd.DataFrame | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        orders = orders if orders is not None else pd.DataFrame()
        self.pricing = PricingRecommender()
        self.inventory = InventoryRecommender(self.pricing)
        self.buyer = BuyerRecommender(orders)
        self.seller = SellerRecommender(weights)
        self.restocking = RestockingRecommender()
        self.collaborative = CollaborativeRecommender(orders)
        self.weights = self.seller.scorer.weights

    def set_weights(self, weights: dict[str, float]) -> None:
        """Re-weight the matching scorer at runtime (UI sliders)."""
        self.seller.scorer = MatchingScorer(weights)
        self.weights = self.seller.scorer.weights

    def summary(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "collaborative_products": int(self.collaborative.matrix.shape[1])
            if not self.collaborative.matrix.empty else 0,
            "collaborative_buyers": int(self.collaborative.matrix.shape[0])
            if not self.collaborative.matrix.empty else 0,
        }


__all__ = [
    "Recommendation", "MatchingScorer", "PricingRecommender",
    "InventoryRecommender", "BuyerRecommender", "SellerRecommender",
    "RestockingRecommender", "CollaborativeRecommender", "RecommendationEngine",
    "GRADE_RANK",
]