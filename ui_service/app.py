"""
Equipment Utilization Dashboard — Streamlit UI

Polls TimescaleDB every few seconds and renders:
  • Live status table for each machine
  • Utilization % bar chart
  • Time-series activity line chart

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg2
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equipment Utilization Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_DSN = (
    f"host={os.getenv('DB_HOST', 'localhost')} "
    f"port={os.getenv('DB_PORT', '5432')} "
    f"dbname={os.getenv('DB_NAME', 'equipment_db')} "
    f"user={os.getenv('DB_USER', 'postgres')} "
    f"password={os.getenv('DB_PASS', 'postgres')}"
)

REFRESH_SEC = int(os.getenv("REFRESH_SEC", "3"))


# ── DB helpers ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_connection():
    """Persistent connection (cached by Streamlit)."""
    for _ in range(10):
        try:
            conn = psycopg2.connect(DB_DSN)
            return conn
        except psycopg2.OperationalError:
            time.sleep(2)
    st.error("Cannot connect to TimescaleDB. Is the DB running?")
    st.stop()


def query(sql: str, params=None) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        conn.close()
        # Force reconnect on next call
        get_connection.clear()
        return pd.DataFrame()


def latest_state() -> pd.DataFrame:
    """Most recent row per equipment_id (current live state)."""
    return query("""
        SELECT DISTINCT ON (equipment_id)
            equipment_id,
            equipment_class,
            current_state,
            current_activity,
            motion_source,
            total_tracked_seconds,
            total_active_seconds,
            total_idle_seconds,
            utilization_percent,
            recorded_at
        FROM equipment_events
        ORDER BY equipment_id, recorded_at DESC;
    """)


def utilization_history(minutes: int = 5) -> pd.DataFrame:
    """Per-minute utilization % for each machine over the last N minutes."""
    return query("""
        SELECT
            time_bucket('10 seconds', recorded_at) AS bucket,
            equipment_id,
            AVG(utilization_percent) AS util_pct,
            MODE() WITHIN GROUP (ORDER BY current_activity) AS activity
        FROM equipment_events
        WHERE recorded_at > NOW() - INTERVAL '%s minutes'
        GROUP BY bucket, equipment_id
        ORDER BY bucket;
    """, params=(minutes,))


def activity_breakdown(equipment_id: str) -> pd.DataFrame:
    return query("""
        SELECT
            current_activity,
            COUNT(*) AS frames
        FROM equipment_events
        WHERE equipment_id = %s
        GROUP BY current_activity
        ORDER BY frames DESC;
    """, params=(equipment_id,))


# ── UI ───────────────────────────────────────────────────────────────────────

def state_badge(state: str) -> str:
    if state == "ACTIVE":
        return "🟢 ACTIVE"
    return "🔴 INACTIVE"


def fmt_seconds(s: float) -> str:
    return str(timedelta(seconds=int(s)))


def render():
    st.title("🏗️ Equipment Utilization Dashboard")
    st.caption(f"Auto-refreshes every {REFRESH_SEC}s  |  DB: {os.getenv('DB_HOST', 'localhost')}")

    # ── Sidebar controls ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")
        history_min = st.slider("History window (minutes)", 1, 30, 5)
        st.markdown("---")
        st.markdown("**Legend**")
        st.markdown("🟢 ACTIVE  |  🔴 INACTIVE")

    df_state = latest_state()

    if df_state.empty:
        st.info("⏳ Waiting for data from the CV service…")
        time.sleep(2)
        st.rerun()

    # ── Live status cards ─────────────────────────────────────────────────
    st.subheader("Live Machine Status")
    cols = st.columns(min(len(df_state), 4))

    for i, row in df_state.iterrows():
        col = cols[i % len(cols)]
        with col:
            util  = row["utilization_percent"]
            color = "#27ae60" if row["current_state"] == "ACTIVE" else "#e74c3c"
            st.markdown(f"""
            <div style="border:2px solid {color}; border-radius:10px; padding:12px; margin-bottom:8px;">
                <h4 style="margin:0; color:{color};">{row['equipment_id']}</h4>
                <small style="color:#888;">{row['equipment_class']}</small><br>
                <b>{state_badge(row['current_state'])}</b><br>
                <span style="font-size:0.85em;">Activity: <b>{row['current_activity']}</b></span><br>
                <span style="font-size:0.85em;">Motion: {row['motion_source']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Utilization summary table ─────────────────────────────────────────
    st.subheader("Utilization Summary")

    display_df = df_state[[
        "equipment_id", "equipment_class",
        "total_tracked_seconds", "total_active_seconds",
        "total_idle_seconds", "utilization_percent",
    ]].copy()

    display_df["total_tracked_seconds"] = display_df["total_tracked_seconds"].apply(fmt_seconds)
    display_df["total_active_seconds"]  = display_df["total_active_seconds"].apply(fmt_seconds)
    display_df["total_idle_seconds"]    = display_df["total_idle_seconds"].apply(fmt_seconds)
    display_df.columns = [
        "Equipment ID", "Class",
        "Total Tracked", "Active Time", "Idle Time", "Utilization %",
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Utilization bar chart ─────────────────────────────────────────────
    st.subheader("Utilization %")
    bar_data = df_state.set_index("equipment_id")[["utilization_percent"]]
    st.bar_chart(bar_data, color="#27ae60")

    # ── Time-series chart ─────────────────────────────────────────────────
    st.subheader(f"Utilization Over Time (last {history_min} min)")
    df_hist = utilization_history(history_min)
    if not df_hist.empty:
        pivot = df_hist.pivot(index="bucket", columns="equipment_id", values="util_pct")
        st.line_chart(pivot)
    else:
        st.info("Not enough history yet.")

    # ── Per-machine activity breakdown ────────────────────────────────────
    st.subheader("Activity Breakdown")
    machine_ids = df_state["equipment_id"].tolist()
    selected = st.selectbox("Select machine", machine_ids)

    if selected:
        df_act = activity_breakdown(selected)
        if not df_act.empty:
            st.bar_chart(df_act.set_index("current_activity")["frames"])
        else:
            st.info("No activity data yet for this machine.")

    # ── Auto-refresh ──────────────────────────────────────────────────────
    time.sleep(REFRESH_SEC)
    st.rerun()


if __name__ == "__main__":
    render()
