"""
FreshSense AI

Main Streamlit application.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from freshsense.config import SETTINGS
from freshsense.db.database import Database
from freshsense.db.repository import (
    InventoryRepository,
    SalesRepository,
    OrdersRepository,
)
from freshsense.models.model_registry import get_registry
from freshsense.logging_config import get_logger

LOG = get_logger(__name__)

st.set_page_config(
    page_title=SETTINGS.application.get("name", "FreshSense AI"),
    page_icon=SETTINGS.ui.get("page_icon", "🥬"),
    layout=SETTINGS.ui.get("layout", "wide"),
)

db = Database()
registry = get_registry()

inventory_repo = InventoryRepository(db)
sales_repo = SalesRepository(db)
orders_repo = OrdersRepository(db)


st.sidebar.title("🥬 FreshSense AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Inventory",
        "Forecasting",
        "Spoilage",
        "Recommendations",
        "Knowledge Assistant",
        "Monitoring",
        "Settings",
    ],
)


if page == "Dashboard":

    st.title("🥬 FreshSense AI")

    st.subheader("System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Inventory Records",
        inventory_repo.count(),
    )

    col2.metric(
        "Sales Records",
        sales_repo.count(),
    )

    col3.metric(
        "Orders",
        orders_repo.count(),
    )

    st.success("FreshSense AI loaded successfully.")


elif page == "Inventory":

    st.title("📦 Inventory Management")

    inventory = inventory_repo.load_dataframe()

    st.subheader("Current Inventory")

    st.dataframe(
        inventory,
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Low Stock Items")

        low_stock = inventory_repo.low_stock()

        st.dataframe(
            low_stock,
            use_container_width=True,
        )

    with col2:
        st.subheader("Products Near Expiry")

        expiring = inventory_repo.expiring_products()

        st.dataframe(
            expiring,
            use_container_width=True,
        )

    st.success(
        f"Loaded {len(inventory)} inventory records."
    )


elif page == "Forecasting":

    st.title("📈 Demand Forecasting")

    st.info("Forecast demand for perishable inventory.")

    if st.button("Run Forecast"):

        forecaster = DemandForecaster()

        forecast = forecaster.predict()

        st.dataframe(
            forecast,
            use_container_width=True,
        )





elif page == "Spoilage":

    st.title("🍎 Spoilage Prediction")

    st.info("Predict products at risk of spoilage.")

    if st.button("Predict Spoilage"):

        predictor = SpoilagePredictor()

        predictions = predictor.predict()

        st.dataframe(
            predictions,
            use_container_width=True,
        )




elif page == "Recommendations":

    st.title("🤝 Smart Recommendations")

    st.info("Generate AI-powered recommendations for buyers and sellers.")

    if st.button("Generate Recommendations"):

        engine = RecommendationEngine()

        recommendations = engine.generate()

        st.dataframe(
            recommendations,
            use_container_width=True,
        )





elif page == "Knowledge Assistant":

    st.title("🤖 FreshSense Knowledge Assistant")

    question = st.text_area(
        "Ask a food safety or inventory question"
    )

    if st.button("Ask"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            retriever = Retriever()

            documents = retriever.retrieve(question)

            answer = ask_llm(
                question,
                documents,
            )

            st.write(answer)



elif page == "Monitoring":

    st.title("📊 System Monitoring")

    st.info("Monitor AI models, database and system performance.")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Models Loaded",
        len(registry.list_models()),
    )

    col2.metric(
        "Database",
        "Connected",
    )

    col3.metric(
        "Application",
        "Healthy",
    )

    st.success(
        "System monitoring dashboard loaded."
    )



elif page == "Settings":

    st.title("⚙️ Settings")

    st.subheader("Application Information")

    st.json(SETTINGS.application)

    st.subheader("Database")

    st.write(db.summary())

    st.subheader("Installed Models")

    st.dataframe(
        registry.list_models(),
        use_container_width=True,
    )







