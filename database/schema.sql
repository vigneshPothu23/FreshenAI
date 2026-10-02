-- ══════════════════════════════════════════════════════════════════════════
-- FreshSense AI — SQLite schema
--
-- Design notes
--   * `batches` is the central fact: it has identity, history and an outcome.
--   * `predictions` + `prediction_outcomes` are split deliberately. Without the
--     outcome table there is logging but no monitoring.
--   * `daily_item_stats` is denormalised on purpose so dashboard queries are
--     O(days) rather than O(events).
--   * Chunk text and provenance live here; embeddings live in the vector store,
--     joined on `chunks.vector_id`. The index can be rebuilt without losing
--     lineage.
-- ══════════════════════════════════════════════════════════════════════════

PRAGMA foreign_keys = ON;

-- ── Reference catalogue ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS items (
    item_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT    NOT NULL UNIQUE,
    category             TEXT    NOT NULL,
    unit                 TEXT    NOT NULL DEFAULT 'kg',
    shelf_life_days      INTEGER NOT NULL DEFAULT 5,
    perishability_level  TEXT    NOT NULL DEFAULT 'Medium',
    storage_type         TEXT    NOT NULL DEFAULT 'Ambient',
    created_at           TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sellers (
    seller_id         TEXT    PRIMARY KEY,
    seller_name       TEXT    NOT NULL,
    store_type        TEXT,
    zone              TEXT,
    latitude          REAL    NOT NULL DEFAULT 13.0827,
    longitude         REAL    NOT NULL DEFAULT 80.2707,
    seller_rating     REAL    NOT NULL DEFAULT 0.0,
    total_orders      INTEGER NOT NULL DEFAULT 0,
    dispute_rate_pct  REAL    NOT NULL DEFAULT 0.0,
    fssai_verified    INTEGER NOT NULL DEFAULT 0,
    onboarded_date    TEXT,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS buyers (
    buyer_id             TEXT    PRIMARY KEY,
    buyer_name           TEXT    NOT NULL,
    buyer_type           TEXT,
    zone                 TEXT,
    latitude             REAL    NOT NULL DEFAULT 13.0827,
    longitude            REAL    NOT NULL DEFAULT 80.2707,
    avg_order_value      REAL    NOT NULL DEFAULT 0.0,
    price_sensitivity    REAL    NOT NULL DEFAULT 0.5,
    max_distance_km      REAL    NOT NULL DEFAULT 10.0,
    min_acceptable_grade TEXT    NOT NULL DEFAULT 'C',
    accepts_split_order  INTEGER NOT NULL DEFAULT 1,
    buyer_rating         REAL    NOT NULL DEFAULT 0.0,
    total_orders         INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── Central fact: a stock batch tracked over time ─────────────────────────
CREATE TABLE IF NOT EXISTS batches (
    batch_id                 TEXT    PRIMARY KEY,
    item_id                  INTEGER REFERENCES items(item_id),
    seller_id                TEXT    REFERENCES sellers(seller_id),
    product_name             TEXT    NOT NULL,
    category                 TEXT,
    brand                    TEXT,
    unit                     TEXT    NOT NULL DEFAULT 'kg',
    zone                     TEXT,
    latitude                 REAL,
    longitude                REAL,

    quantity_initial         REAL    NOT NULL DEFAULT 0,
    quantity_available       REAL    NOT NULL DEFAULT 0,

    manufacturing_date       TEXT,
    arrival_date             TEXT,
    expiry_date              TEXT    NOT NULL,
    shelf_life_days          INTEGER NOT NULL DEFAULT 5,
    stock_age_days           INTEGER NOT NULL DEFAULT 0,
    days_to_expiry           INTEGER NOT NULL DEFAULT 0,
    age_ratio                REAL    NOT NULL DEFAULT 0,

    storage_type             TEXT,
    storage_temperature_c    REAL,
    ambient_temp_c           REAL,
    humidity_pct             REAL,
    temp_breach_hours        REAL    NOT NULL DEFAULT 0,
    perishability_level      TEXT,
    perishability_score      INTEGER NOT NULL DEFAULT 2,
    storage_risk_score       INTEGER NOT NULL DEFAULT 2,

    cost_price               REAL,
    mrp                      REAL,
    discount_pct             REAL    NOT NULL DEFAULT 0,
    effective_price          REAL,
    daily_avg_sales          REAL    NOT NULL DEFAULT 1,

    quality_grade            TEXT    NOT NULL DEFAULT 'A',
    status                   TEXT    NOT NULL DEFAULT 'active',
    is_spoiled               INTEGER NOT NULL DEFAULT 0,
    waste_quantity           REAL    NOT NULL DEFAULT 0,

    environment_stress_index REAL    NOT NULL DEFAULT 0,
    action_priority          REAL    NOT NULL DEFAULT 0,
    inventory_value          REAL    NOT NULL DEFAULT 0,
    capital_at_risk          REAL    NOT NULL DEFAULT 0,
    projected_surplus        REAL    NOT NULL DEFAULT 0,

    has_humidity_sensor      INTEGER NOT NULL DEFAULT 1,
    has_temp_logger          INTEGER NOT NULL DEFAULT 1,
    is_new_seller            INTEGER NOT NULL DEFAULT 0,

    created_at               TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at               TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_batches_expiry   ON batches(expiry_date);
CREATE INDEX IF NOT EXISTS ix_batches_seller   ON batches(seller_id, status);
CREATE INDEX IF NOT EXISTS ix_batches_product  ON batches(product_name);
CREATE INDEX IF NOT EXISTS ix_batches_priority ON batches(action_priority DESC);

-- ── Daily aggregated demand (forecasting source) ──────────────────────────
CREATE TABLE IF NOT EXISTS daily_item_stats (
    stat_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    stat_date         TEXT    NOT NULL,
    product_name      TEXT    NOT NULL,
    category          TEXT,
    unit              TEXT,
    units_sold        REAL    NOT NULL DEFAULT 0,
    avg_selling_price REAL,
    revenue           REAL,
    ambient_temp_c    REAL,
    humidity_pct      REAL,
    is_weekend        INTEGER NOT NULL DEFAULT 0,
    is_festival       INTEGER NOT NULL DEFAULT 0,
    day_of_week       TEXT,
    UNIQUE (stat_date, product_name)
);

CREATE INDEX IF NOT EXISTS ix_stats_product_date
    ON daily_item_stats(product_name, stat_date);

-- ── Transactions ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    order_id           TEXT    PRIMARY KEY,
    order_date         TEXT    NOT NULL,
    buyer_id           TEXT    REFERENCES buyers(buyer_id),
    seller_id          TEXT    REFERENCES sellers(seller_id),
    batch_id           TEXT    REFERENCES batches(batch_id),
    product_name       TEXT    NOT NULL,
    category           TEXT,
    quantity           REAL    NOT NULL,
    unit               TEXT,
    unit_price         REAL    NOT NULL,
    mrp                REAL,
    order_value        REAL    NOT NULL,
    buyer_savings      REAL    NOT NULL DEFAULT 0,
    distance_km        REAL,
    quality_grade      TEXT,
    match_source       TEXT,
    fulfilment         TEXT,
    payment_status     TEXT,
    dispute_raised     INTEGER NOT NULL DEFAULT 0,
    dispute_reason     TEXT,
    buyer_rating_given REAL,
    awaiting_rating    INTEGER NOT NULL DEFAULT 0,
    waste_prevented_kg REAL    NOT NULL DEFAULT 0,
    created_at         TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_orders_date   ON orders(order_date);
CREATE INDEX IF NOT EXISTS ix_orders_buyer  ON orders(buyer_id);
CREATE INDEX IF NOT EXISTS ix_orders_seller ON orders(seller_id);

-- ── Model output ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type      TEXT    NOT NULL,      -- batch | product
    entity_id        TEXT    NOT NULL,
    prediction_type  TEXT    NOT NULL,      -- spoilage_risk | demand | price | shelf_life
    value            REAL    NOT NULL,
    label            TEXT,
    confidence       REAL    NOT NULL DEFAULT 0,
    explanation      TEXT,                  -- JSON: feature contributions
    model_name       TEXT,
    model_version    TEXT    NOT NULL DEFAULT 'v1.0',
    horizon_date     TEXT,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_pred_entity
    ON predictions(entity_type, entity_id, prediction_type);
CREATE INDEX IF NOT EXISTS ix_pred_created ON predictions(created_at);

-- The table that turns logging into monitoring.
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    outcome_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(prediction_id)
                          ON DELETE CASCADE,
    actual_value  REAL    NOT NULL,
    error         REAL,
    abs_error     REAL,
    observed_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_outcome_pred ON prediction_outcomes(prediction_id);

-- ── Recommendations ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    rec_type          TEXT    NOT NULL,     -- inventory | buyer | seller | pricing | restocking
    entity_type       TEXT    NOT NULL,
    entity_id         TEXT    NOT NULL,
    target_type       TEXT,
    target_id         TEXT,
    score             REAL    NOT NULL DEFAULT 0,
    rank              INTEGER NOT NULL DEFAULT 0,
    rationale         TEXT,
    payload           TEXT,                 -- JSON detail
    algorithm         TEXT,
    shown_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    accepted_at       TEXT
);

CREATE INDEX IF NOT EXISTS ix_rec_type ON recommendations(rec_type, entity_id);

-- ── RAG provenance ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    document_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,
    title        TEXT NOT NULL,
    uri          TEXT,
    doc_type     TEXT,
    license      TEXT,
    checksum     TEXT,
    ingested_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(document_id)
                         ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    section      TEXT,
    text         TEXT    NOT NULL,
    token_count  INTEGER NOT NULL DEFAULT 0,
    vector_id    TEXT,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(document_id);

CREATE TABLE IF NOT EXISTS chat_history (
    message_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL,           -- user | assistant
    content     TEXT    NOT NULL,
    sources     TEXT,                       -- JSON: retrieved chunk ids
    latency_ms  INTEGER NOT NULL DEFAULT 0,
    provider    TEXT,
    refused     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_chat_session ON chat_history(session_id, created_at);

-- ── Agent execution trace ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id           TEXT    PRIMARY KEY,
    session_id       TEXT,
    user_query       TEXT    NOT NULL,
    final_output     TEXT,
    status           TEXT    NOT NULL DEFAULT 'running',
    total_latency_ms INTEGER NOT NULL DEFAULT 0,
    replan_count     INTEGER NOT NULL DEFAULT 0,
    started_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    completed_at     TEXT
);

CREATE TABLE IF NOT EXISTS agent_steps (
    step_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT    NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    step_index  INTEGER NOT NULL,
    agent_name  TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    tool_called TEXT,
    detail      TEXT,
    status      TEXT    NOT NULL DEFAULT 'ok',
    latency_ms  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_steps_run ON agent_steps(run_id, step_index);

-- ── Observability ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS system_logs (
    log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    component  TEXT    NOT NULL,
    action     TEXT    NOT NULL,
    status     TEXT    NOT NULL DEFAULT 'ok',
    latency_ms INTEGER NOT NULL DEFAULT 0,
    detail     TEXT,
    metadata   TEXT,                        -- JSON
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_logs_component ON system_logs(component, created_at);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    provider          TEXT    NOT NULL,
    model             TEXT,
    prompt_chars      INTEGER NOT NULL DEFAULT 0,
    completion_chars  INTEGER NOT NULL DEFAULT 0,
    latency_ms        INTEGER NOT NULL DEFAULT 0,
    status            TEXT    NOT NULL DEFAULT 'ok',
    error             TEXT,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name   TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value REAL NOT NULL,
    dataset      TEXT,
    recorded_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS business_metrics (
    metric_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_date  TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value REAL NOT NULL,
    dimension    TEXT,
    UNIQUE (metric_date, metric_name, dimension)
);

-- ── Application users (demo scope) ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    username     TEXT    NOT NULL UNIQUE,
    display_name TEXT    NOT NULL,
    role         TEXT    NOT NULL DEFAULT 'operator',
    linked_id    TEXT,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── Convenience views for the dashboard ───────────────────────────────────
CREATE VIEW IF NOT EXISTS v_active_batches AS
SELECT *
FROM batches
WHERE status = 'active' AND quantity_available > 0;

CREATE VIEW IF NOT EXISTS v_at_risk_batches AS
SELECT batch_id, product_name, category, seller_id, zone, quantity_available,
       unit, days_to_expiry, quality_grade, effective_price, discount_pct,
       environment_stress_index, action_priority, capital_at_risk
FROM batches
WHERE status = 'active'
  AND quantity_available > 0
  AND days_to_expiry BETWEEN 0 AND 2
ORDER BY action_priority DESC;

CREATE VIEW IF NOT EXISTS v_daily_revenue AS
SELECT substr(order_date, 1, 10) AS day,
       COUNT(*)                  AS orders,
       SUM(order_value)          AS revenue,
       SUM(buyer_savings)        AS savings,
       SUM(waste_prevented_kg)   AS waste_prevented_kg
FROM orders
GROUP BY day
ORDER BY day;