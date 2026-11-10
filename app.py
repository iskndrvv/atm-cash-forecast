"""Демо: прогноз выдачи наличных в банкоматах Бишкека на неделю вперёд.

Запуск:  streamlit run app.py
Нужно:   data/processed/daily.parquet (ноутбук 01) и models/ (ноутбук 03).
"""
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.forecast import HORIZON, ROOT, forecast_week, load_model, prepare, valid_starts

TEST_START, TEST_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31")
LAST_START = TEST_END - pd.Timedelta(days=HORIZON - 1)   # неделя должна целиком помещаться в test
BUFFER = 0.15

ATM_TYPE_RU = {
    "office_business": "офисный район",
    "pension_social": "у Соцфонда (пенсии и пособия)",
    "transport_market": "рынок / транспортный узел",
    "residential": "жилой район",
    "shopping_mall": "торговый центр",
    "tourist_center": "туристический центр",
    "highway_gas": "АЗС на трассе",
}
PLACEMENT_RU = {"street": "на улице", "indoor_branch": "в отделении банка",
                "mall": "в торговом центре", "gas_station": "на АЗС"}
DOW_RU = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
MODEL_C, BASE_C, FACT_C, BAD_C = "#eb6834", "#8a8984", "#2a78d6", "#a8c8f0"

st.set_page_config(page_title="Прогноз наличных в банкоматах", page_icon="🏧", layout="wide")


def som(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") + " сом"


def short_som(v: float) -> str:
    return f"{v / 1e6:,.2f} млн".replace(",", " ").replace(".", ",") if abs(v) >= 1e6 else f"{v / 1e3:,.0f} тыс.".replace(",", " ")


def pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}%".replace(".", ",")


def round_up(v: float, step: int = 10_000) -> float:
    return math.ceil(v / step) * step


def day_note(r) -> str:
    if r.is_holiday:
        return f"праздник: {r.holiday_name}"
    if r.is_pre_holiday:
        return "предпраздничный день"
    if r.is_pension_day:
        return "пенсии и зарплата"
    if r.is_payday:
        return "зарплата"
    return "выходной" if r.is_weekend else ""


# ------------------------------------------------------------------ данные и прогнозы
@st.cache_resource(show_spinner="Загружаю данные и считаю прогнозы…")
def load_all():
    data_path = ROOT / "data" / "processed" / "daily.parquet"
    if not data_path.exists():
        return None
    model, meta = load_model()
    df = prepare(pd.read_parquet(data_path))
    starts = valid_starts(df, TEST_START, LAST_START)
    res = forecast_week(df, starts, model, meta)

    weeks = pd.DataFrame({
        "row": starts,
        "atm_id": df["atm_id"].to_numpy()[starts],
        "date": df["date"].to_numpy()[starts],
        "fact": res["actual"].sum(1),
        "model": res["model"].sum(1),
        "baseline": res["baseline"].sum(1),
        "clean_week": ~res["disrupted"].any(1),   # в неделе не было сбоев — факт = настоящий спрос
    })
    for m in ("model", "baseline"):
        weeks[f"rec_{m}"] = [round_up(v) for v in weeks[m] * (1 + BUFFER)]
    return df, meta, res, weeks, starts


def accuracy(res, weeks, mask_starts) -> dict:
    """Точность модели и baseline на выбранных неделях."""
    ok = ~res["disrupted"][mask_starts]
    act = res["actual"][mask_starts]
    w = weeks[mask_starts & weeks.clean_week.to_numpy()]
    out = {}
    for m in ("model", "baseline"):
        p = res[m][mask_starts]
        out[m] = {
            "wape_day": np.abs(p - act)[ok].sum() / act[ok].sum(),
            "wape_week": np.abs(w[m] - w.fact).sum() / w.fact.sum(),
            "enough": (w[f"rec_{m}"] >= w.fact).mean(),
            "idle": (w[f"rec_{m}"] - w.fact).clip(lower=0).mean(),
            "short": (w.fact - w[f"rec_{m}"]).clip(lower=0).mean(),
        }
    out["weeks"] = len(w)
    return out


loaded = load_all()
if loaded is None:
    st.error("Не найден файл `data/processed/daily.parquet`. Сначала скачайте данные и запустите "
             "`notebooks/01_eda.ipynb` (см. README).")
    st.stop()
df, meta, res, weeks, starts = loaded

# ------------------------------------------------------------------ выбор банкомата и даты
atms = (df.groupby("atm_id").agg(district=("district_name", "first"), atm_type=("atm_type", "first"),
                                 placement=("placement", "first"), capacity=("cash_capacity", "first"),
                                 install=("install_date", "first")))
atms = atms[atms.index.isin(weeks.atm_id.unique())]

st.title("🏧 Сколько наличных загрузить в банкомат на неделю")
st.caption("Демо на данных MBank (синтетические, Бишкек). Выберите банкомат и день, с которого начинается неделя. "
           "Прогноз делается вечером накануне: модель знает только то, что было до этого дня, "
           "а потом сравниваем с тем, что произошло на самом деле.")

# Ссылка на конкретную неделю: ?atm=ATM-0053&start=2025-08-08 (удобно для показа)
qp_atm = st.query_params.get("atm")
qp_start = pd.to_datetime(st.query_params.get("start"), errors="coerce")

with st.sidebar:
    st.header("Параметры")
    atm_id = st.selectbox(
        "Банкомат", atms.index, index=atms.index.get_loc(qp_atm) if qp_atm in atms.index else 0,
        format_func=lambda a: f"{a} · {atms.loc[a, 'district']} · {ATM_TYPE_RU.get(atms.loc[a, 'atm_type'], atms.loc[a, 'atm_type'])}")
    a = atms.loc[atm_id]
    st.markdown(f"**Район:** {a.district}  \n**Тип места:** {ATM_TYPE_RU.get(a.atm_type, a.atm_type)}  \n"
                f"**Где стоит:** {PLACEMENT_RU.get(a.placement, a.placement)}  \n"
                f"**Вмещает:** {som(a.capacity)}")

    atm_weeks = weeks[weeks.atm_id == atm_id]
    min_d, max_d = atm_weeks.date.min().date(), atm_weeks.date.max().date()
    default = qp_start.date() if pd.notna(qp_start) and qp_atm == atm_id else pd.Timestamp("2025-10-06").date()
    start_date = st.date_input("Начало недели", value=min(max(min_d, default), max_d),
                               min_value=min_d, max_value=max_d, format="DD.MM.YYYY")
    st.caption(f"Доступно: {min_d:%d.%m.%Y} – {max_d:%d.%m.%Y} (проверочный период, который модель не видела при обучении).")

i = int(np.flatnonzero((weeks.atm_id == atm_id).to_numpy() & (weeks.date == pd.Timestamp(start_date)).to_numpy())[0])
w = weeks.iloc[i]
days = df.iloc[res["rows"][i]].copy()
days["Прогноз модели"] = res["model"][i]
days["Простой прогноз"] = res["baseline"][i]
days["Факт"] = res["actual"][i]
days["сбой"] = res["disrupted"][i]
days["label"] = [f"{DOW_RU[d.day_of_week]} {d:%d.%m}" for d in days.date]
end_date = days.date.iloc[-1]

# ------------------------------------------------------------------ прогноз на неделю
st.subheader(f"Прогноз на неделю {start_date:%d.%m} – {end_date:%d.%m.%Y}")

long = days.melt(id_vars=["label", "сбой"], value_vars=["Прогноз модели", "Простой прогноз", "Факт"],
                 var_name="ряд", value_name="сом")
order = list(days.label)
x = alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=0))
y_axis = alt.Axis(title="тыс. сом в день", labelExpr="replace(format(datum.value / 1000, ',.0f'), ',', ' ')")
bars = alt.Chart(long[long["ряд"] == "Факт"]).mark_bar(size=34, opacity=0.85).encode(
    x=x, y=alt.Y("сом:Q", axis=y_axis),
    color=alt.condition("datum['сбой']", alt.value(BAD_C), alt.value(FACT_C)),
    tooltip=[alt.Tooltip("label:N", title="день"), alt.Tooltip("сом:Q", format=",.0f", title="факт, сом")])
lines = alt.Chart(long[long["ряд"] != "Факт"]).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=3).encode(
    x=x, y="сом:Q",
    color=alt.Color("ряд:N", scale=alt.Scale(domain=["Прогноз модели", "Простой прогноз"], range=[MODEL_C, BASE_C]),
                    legend=alt.Legend(title=None, orient="top")),
    strokeDash=alt.condition("datum['ряд'] == 'Простой прогноз'", alt.value([6, 4]), alt.value([1, 0])),
    tooltip=[alt.Tooltip("label:N", title="день"), alt.Tooltip("ряд:N", title=""),
             alt.Tooltip("сом:Q", format=",.0f", title="сом")])
st.altair_chart((bars + lines).properties(height=340), width="stretch")
legend = "Синие столбики — сколько банкомат выдал на самом деле. Оранжевая линия — прогноз модели, серая пунктирная — простой прогноз (среднее за тот же день недели за 4 прошлые недели)."
if days["сбой"].any():
    legend += " **Светлые столбики — дни, когда банкомат был пуст или не работал:** настоящий спрос в эти дни был выше."
st.caption(legend)

table = pd.DataFrame({
    "День": days.label,
    "Особый день": [day_note(r) for r in days.itertuples()],
    "Прогноз модели": days["Прогноз модели"].map(som),
    "Простой прогноз": days["Простой прогноз"].map(som),
    "Факт": [som(v) + (" ⚠️ сбой" if bad else "") for v, bad in zip(days["Факт"], days["сбой"])],
})
with st.expander("По дням"):
    st.dataframe(table, hide_index=True, width="stretch")

# ------------------------------------------------------------------ рекомендация
st.subheader("Сколько загрузить на неделю")
st.caption(f"Рекомендация = сумма прогноза на 7 дней + запас {pct(BUFFER, 0)}, округлено вверх до 10 тыс. сом.")

c1, c2, c3 = st.columns(3)
c1.metric("По модели", som(w.rec_model), help=f"Прогноз на неделю {som(w.model)} + {pct(BUFFER, 0)}")
c2.metric("По простому прогнозу", som(w.rec_baseline), delta=som(w.rec_baseline - w.rec_model),
          delta_color="off", help=f"Прогноз на неделю {som(w.baseline)} + {pct(BUFFER, 0)}. Разница — относительно модели.")
c3.metric("Выдано на самом деле", som(w.fact))


def verdict(name: str, rec: float) -> str:
    diff = rec - w.fact
    if diff >= 0:
        return f"✅ **{name}:** денег хватило бы. Без дела осталось бы {som(diff)} ({pct(diff / w.fact, 0)} от выдачи за неделю)."
    return f"❌ **{name}:** денег **не хватило бы** — не хватает {som(-diff)}. Банкомат опустел бы раньше конца недели."


st.markdown(verdict("Модель", w.rec_model) + "  \n" + verdict("Простой прогноз", w.rec_baseline))
if not w.clean_week:
    st.info("В эту неделю банкомат был пуст или не работал хотя бы один день — фактическая выдача занижена, "
            "реальный спрос был больше.")
if w.rec_model > a.capacity:
    st.warning(f"Рекомендация модели больше, чем вмещает банкомат ({som(a.capacity)}). "
               "Понадобится дополнительная загрузка в середине недели.")

# ------------------------------------------------------------------ точность
st.subheader("Насколько модели можно доверять")
st.caption("Проверка на всём проверочном периоде (июль–декабрь 2025): для каждого дня делаем недельный прогноз "
           "так же, как выше, и сравниваем с фактом. Учитываются только недели без сбоев.")

all_acc = accuracy(res, weeks, np.ones(len(weeks), bool))
atm_acc = accuracy(res, weeks, (weeks.atm_id == atm_id).to_numpy())

rows = [
    ("Ошибка по дням", "wape_day", "Сумма всех ошибок по дням / сумма выдачи. «20%» — в среднем ошибаемся на пятую часть дневной выдачи. Меньше — лучше.", pct),
    ("Ошибка за неделю", "wape_week", "Насколько сумма прогноза за 7 дней отличается от факта. Ошибки разных дней частично гасят друг друга, поэтому она меньше.", pct),
    ("Загрузки хватает", "enough", f"В какой доле недель рекомендации (прогноз + {pct(BUFFER, 0)}) хватило бы на всю неделю.", lambda v: pct(v, 0)),
    ("Лишние деньги в банкомате", "idle", "Сколько в среднем осталось бы невыданным к концу недели. Меньше — лучше: деньги не лежат без дела.", short_som),
]

for title, acc, n_weeks in [(f"Банкомат {atm_id}", atm_acc, atm_acc["weeks"]), ("Вся сеть (120 банкоматов)", all_acc, all_acc["weeks"])]:
    st.markdown(f"**{title}** · недель в проверке: {n_weeks:,}".replace(",", " "))
    cols = st.columns(len(rows))
    for col, (label, key, help_, f) in zip(cols, rows):
        mv, bv = acc["model"][key], acc["baseline"][key]
        better = (mv >= bv) if key == "enough" else (mv <= bv)
        col.metric(label, f(mv), delta=f"простой прогноз: {f(bv)}", delta_color="normal" if better else "inverse",
                   delta_arrow="off", help=help_)

d = all_acc["model"]["wape_day"] / all_acc["baseline"]["wape_day"] - 1
st.success(f"По всей сети модель ошибается на **{pct(-d, 0)} меньше**, чем простой прогноз, "
           f"а её недельной загрузки хватает в {pct(all_acc['model']['enough'], 0)} недель "
           f"против {pct(all_acc['baseline']['enough'], 0)} у простого прогноза.")

with st.expander("Как это работает"):
    st.markdown(f"""
- **Модель** — LightGBM (градиентный бустинг), обучена на данных до 30.06.2025. Использует историю выдачи банкомата (вчера, 7, 14, 28 дней назад, средние за 7 и 28 дней), календарь (день недели и месяца, праздники, дни пенсий и зарплат), тип места, район и погоду.
- **Прогноз на неделю** строится по шагам: на первый день — по известной истории, на следующие — подставляя прогнозы предыдущих дней вместо ещё неизвестной выдачи.
- **Простой прогноз** — среднее за тот же день недели за 4 прошлые недели. Так часто планируют загрузку без модели.
- **Дни сбоев** (банкомат пуст или не работает) модель не использует как историю: в такие дни видна не потребность клиентов, а то, что банкомат успел выдать.
- Ошибка модели на прогнозе «на завтра» по всей сети: **{pct(meta['metrics_test']['WAPE'])}** (подробно — `notebooks/03_optimization.ipynb`).
""")
