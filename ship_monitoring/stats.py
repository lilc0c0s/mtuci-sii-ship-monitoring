from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px


def records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).copy()
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp_dt"] = pd.NaT

    def _effective_ship(row):
        if row.get("input_type") == "image":
            return row.get("ship_count")
        return row.get("ship_count_max")

    df["ship_count_effective"] = df.apply(_effective_ship, axis=1)
    df = df.sort_values("timestamp_dt")
    return df


def compute_kpis(records: list[dict[str, Any]]) -> dict[str, Any]:
    df = records_to_df(records)
    if df.empty:
        return {
            "total_requests": 0,
            "images": 0,
            "videos": 0,
            "avg_ships": 0.0,
            "max_ships": 0,
            "last_timestamp": None,
        }

    total = int(len(df))
    images = int((df["input_type"] == "image").sum()) if "input_type" in df.columns else 0
    videos = int((df["input_type"] == "video").sum()) if "input_type" in df.columns else 0

    ships = pd.to_numeric(df["ship_count_effective"], errors="coerce").fillna(0)
    return {
        "total_requests": total,
        "images": images,
        "videos": videos,
        "avg_ships": float(ships.mean()) if len(ships) else 0.0,
        "max_ships": int(ships.max()) if len(ships) else 0,
        "last_timestamp": (df["timestamp"].iloc[-1] if "timestamp" in df.columns and len(df) else None),
    }


def make_figures(records: list[dict[str, Any]]):
    df = records_to_df(records)
    if df.empty:
        return None, None

    fig_time = px.line(
        df,
        x="timestamp_dt",
        y="ship_count_effective",
        title="Детекции судов (boat) по времени",
        markers=True,
    )

    fig_hist = px.histogram(
        df,
        x="ship_count_effective",
        nbins=15,
        title="Распределение количества судов на запрос",
    )

    return fig_time, fig_hist
