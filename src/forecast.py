"""Недельный прогноз выдачи наличных на модели из notebooks/03_optimization.ipynb.

Модель прогнозирует один день вперёд: ей нужна выдача вчера, неделю назад и т.д.
Чтобы получить прогноз на 7 дней, делаем его пошагово (рекурсивно): прогноз на день D
делаем по истории до D−1, а для дня D+1 вместо неизвестной «вчерашней» выдачи
подставляем прогноз на D, и так далее. Будущие факты модель не видит.

Календарь, погода и курс считаются известными заранее (в жизни это календарь
и прогноз погоды; погода почти не влияет на модель).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 7
WINDOW = 28  # самый длинный лаг / окно среднего
WEATHER = ["temp_avg", "precip_mm", "snowfall_mm", "wind_max"]
BOOL = ["is_holiday", "is_pre_holiday", "is_payday", "is_pension_day", "is_ramadan", "is_weekend"]


def load_model(models_dir=ROOT / "models"):
    model = joblib.load(Path(models_dir) / "lgbm_cash_forecast.joblib")
    meta = json.loads((Path(models_dir) / "features.json").read_text())
    return model, meta


def prepare(daily: pd.DataFrame) -> pd.DataFrame:
    """Признаки, которые не зависят от истории выдачи (календарь, погода вчера, банкомат)."""
    df = daily.sort_values(["atm_id", "date"]).reset_index(drop=True)
    g = df.groupby("atm_id", sort=False)
    df["clean"] = df["cash_out"].where(~df["is_disrupted"])  # дни сбоев — не настоящий спрос
    for c in WEATHER:
        df[f"{c}_lag1"] = g[c].shift(1)
    df["fx_shock_last3"] = g["fx_shock"].transform(
        lambda s: s.astype(int).shift(1).rolling(3, min_periods=1).max()).fillna(0).astype(int)
    df["is_street"] = (df["placement"] == "street").astype(int)
    df["dom"] = df["date"].dt.day
    for c in BOOL:
        df[c] = df[c].astype(int)
    df["first_row"] = np.arange(len(df)) - g.cumcount().to_numpy()  # первая строка своего банкомата
    return df


def _model_frame(x: pd.DataFrame, meta: dict) -> pd.DataFrame:
    x = x[meta["features"]].copy()
    for c, cats in meta["categorical"].items():
        x[c] = pd.Categorical(x[c].astype(str), categories=cats)
    return x


def _mean(win: np.ndarray, min_count: int) -> np.ndarray:
    cnt = (~np.isnan(win)).sum(axis=1)
    s = np.nansum(win, axis=1)
    return np.where(cnt >= min_count, s / np.maximum(cnt, 1), np.nan)


def forecast_week(df: pd.DataFrame, starts: np.ndarray, model, meta) -> dict:
    """Прогноз на HORIZON дней для каждого старта (номер строки df = первый день недели).

    Возвращает массивы формы (len(starts), HORIZON): model, baseline, actual, disrupted, rows.
    Baseline — среднее по тому же дню недели за 4 прошлые недели (все они в прошлом).
    """
    starts = np.asarray(starts)
    n, k = len(df), len(starts)
    clean = df["clean"].to_numpy(float)
    first = df["first_row"].to_numpy()[starts]
    pred = np.full((k, HORIZON), np.nan)
    base = np.full((k, HORIZON), np.nan)
    offs = np.arange(-WINDOW, 0)

    for h in range(HORIZON):
        t = starts + h
        idx = t[:, None] + offs[None, :]                       # дни t−28 … t−1
        known = (idx >= first[:, None]) & (idx < starts[:, None])
        hist = np.where(known, clean[np.clip(idx, 0, n - 1)], np.nan)
        future = idx >= starts[:, None]                        # уже спрогнозированные дни недели
        fut = pred[np.arange(k)[:, None], np.clip(idx - starts[:, None], 0, HORIZON - 1)]
        win = np.where(future, fut, hist)

        x = df.iloc[t].copy()
        for lag in (1, 7, 14, 28):
            x[f"lag_{lag}"] = win[:, WINDOW - lag]
        x["roll_mean_7"] = _mean(win[:, -7:], 3)
        x["roll_mean_28"] = _mean(win, 10)
        pred[:, h] = model.predict(_model_frame(x, meta))
        base[:, h] = _mean(win[:, [WINDOW - 7, WINDOW - 14, WINDOW - 21, WINDOW - 28]], 1)

    rows = starts[:, None] + np.arange(HORIZON)[None, :]
    return {
        "model": pred,
        "baseline": base,
        "actual": df["cash_out"].to_numpy(float)[rows],
        "disrupted": df["is_disrupted"].to_numpy(bool)[rows],
        "rows": rows,
    }


def valid_starts(df: pd.DataFrame, first_day, last_day, min_history: int = WINDOW) -> np.ndarray:
    """Строки, с которых можно начать неделю: дата в периоде, 7 дней помещаются, есть 28 дней истории."""
    pos = np.arange(len(df))
    last_row = df.groupby("atm_id", sort=False).cumcount(ascending=False).to_numpy()
    ok = (df["date"].between(first_day, last_day).to_numpy()
          & (pos - df["first_row"].to_numpy() >= min_history)
          & (last_row >= HORIZON - 1))
    return pos[ok]
