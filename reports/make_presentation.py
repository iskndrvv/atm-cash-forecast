"""Собирает reports/presentation.pptx (16:9, 8 слайдов + 2 запасных в приложении).

Запуск из корня проекта:  python reports/make_presentation.py
Нужны data/processed/daily.parquet и models/ (ноутбуки 01 и 03).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.forecast import forecast_week, load_model, prepare  # noqa: E402

FIG = ROOT / "reports" / "figures"
PRES_FIG = FIG / "presentation"
PRES_FIG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "reports" / "presentation.pptx"

INK, INK2, MUTED, LINE = "1A1A1A", "52514E", "8A8984", "E4E3DE"
ORANGE, BLUE, AQUA, SOFT = "EB6834", "2A78D6", "1BAF7A", "F6F5F2"
FONT = "Arial"

# ------------------------------------------------------------------ графики для слайдов
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#d6d5d0",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ebeae6",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#52514e", "legend.frameon": False,
})
TYPE_RU = {"transport_market": "Рынок / транспорт", "shopping_mall": "Торговый центр", "office_business": "Офисы",
           "tourist_center": "Туристический центр", "residential": "Жилой район", "pension_social": "Соцфонд",
           "highway_gas": "АЗС на трассе"}

daily = pd.read_parquet(ROOT / "data" / "processed" / "daily.parquet")

# 1) Средняя выдача по типам точек
clean = daily[~daily.is_disrupted]
by_type = clean.groupby(["atm_type", "atm_id"]).cash_out.mean().groupby("atm_type").median().sort_values() / 1e3
fig, ax = plt.subplots(figsize=(6.4, 4.6))
colors = ["#" + (ORANGE if t in ("transport_market", "shopping_mall") else BLUE) for t in by_type.index]
ax.barh([TYPE_RU[t] for t in by_type.index], by_type.values, color=colors, height=0.62)
for y, v in enumerate(by_type.values):
    ax.text(v + 12, y, f"{v:.0f}", va="center", fontsize=14, color="#52514e")
ax.set_xlabel("тыс. сом в день на банкомат")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, by_type.max() * 1.15)
fig.tight_layout(); fig.savefig(PRES_FIG / "types.png", dpi=200); plt.close(fig)

# 1б) Особые дни — значения из notebooks/01_eda.ipynb (раздел «Особые дни»)
special = pd.Series({"Предпраздник": 61.3, "Пенсии": 46.5, "Зарплата": 37.9, "Выходной": 6.1, "Праздник": -13.6})[::-1]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.barh(special.index, special.values, color=["#" + (ORANGE if v > 0 else BLUE) for v in special.values], height=0.62)
for y, v in enumerate(special.values):
    ax.text(v + (2 if v > 0 else -2), y, f"{v:+.0f}%", va="center", ha="left" if v > 0 else "right", fontsize=14, color="#52514e")
ax.axvline(0, color="#52514e", lw=1)
ax.set_xlim(-30, 75)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
ax.set_xlabel("выдача относительно обычного будня")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(PRES_FIG / "special_days.png", dpi=200); plt.close(fig)

# 1в) Рост спроса: средняя выдача на банкомат в день по месяцам (все дни, как в ноутбуке 01)
monthly = daily.groupby(daily.date.dt.to_period("M")).cash_out.mean() / 1e3
fig, ax = plt.subplots(figsize=(6.4, 4.6))
x = monthly.index.to_timestamp()
ax.plot(x, monthly.values, color="#" + BLUE, lw=3, marker="o", ms=5)
for k, (yr, c) in enumerate([(2024, MUTED), (2025, ORANGE)]):
    v = daily[daily.date.dt.year == yr].cash_out.mean() / 1e3
    ax.hlines(v, pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr}-12-01"), color="#" + c, lw=2.5, ls="--")
    ax.text(0.03, 0.95 - k * 0.09, f"среднее {yr}: {v:.0f}", transform=ax.transAxes, color="#" + c,
            fontsize=14, fontweight="bold", va="top")
ax.set_ylabel("тыс. сом в день на банкомат")
ax.set_xticks(pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"]),
              ["янв 24", "июл 24", "янв 25", "июл 25"])
ax.set_ylim(monthly.min() * 0.9, monthly.max() * 1.18)
fig.tight_layout(); fig.savefig(PRES_FIG / "growth.png", dpi=200); plt.close(fig)

# 1г) День месяца (как в ноутбуке 03: выдача относительно среднего за 28 прошлых дней, train, без сбоев)
dd = daily.sort_values(["atm_id", "date"]).copy()
cc = dd.cash_out.where(~dd.is_disrupted)
dd["roll28"] = cc.groupby(dd.atm_id).transform(lambda s: s.shift(1).rolling(28, min_periods=10).mean())
tr = dd[(dd.date <= "2025-06-30") & ~dd.is_disrupted & dd.roll28.notna()]
by_dom = (tr.cash_out / tr.roll28 - 1).groupby(tr.date.dt.day).mean() * 100
fig, ax = plt.subplots(figsize=(7.4, 3.9))
ax.bar(by_dom.index, by_dom.values, width=0.72,
       color=["#" + (ORANGE if 10 <= d <= 14 else AQUA if 25 <= d <= 27 else BLUE) for d in by_dom.index])
ax.axhline(0, color="#52514e", lw=1)
ax.set_xticks([1, 5, 10, 14, 20, 25, 27, 31])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
ax.set_xlabel("день месяца")
ax.text(12, by_dom.max() + 3, "пенсии + зарплата", ha="center", color="#" + ORANGE, fontsize=14, fontweight="bold")
ax.text(26, by_dom.loc[25:27].max() + 3, "зарплата", ha="center", color="#127a56", fontsize=14, fontweight="bold")
ax.set_ylim(by_dom.min() - 5, by_dom.max() + 12)
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(PRES_FIG / "day_of_month.png", dpi=200); plt.close(fig)
print("День месяца, %: 10–14 —", by_dom.loc[10:14].round(0).tolist(), "| 25–27 —", by_dom.loc[25:27].round(0).tolist())

# 2) Неделя пенсий у банкомата Соцфонда: модель против простого прогноза
model, meta = load_model()
df = prepare(daily)
DEMO_ATM, DEMO_START = "ATM-0053", "2025-08-08"
s = np.flatnonzero(((df.atm_id == DEMO_ATM) & (df.date == DEMO_START)).to_numpy())
res = forecast_week(df, s, model, meta)
d = df.iloc[res["rows"][0]]
dow = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
labels = [f"{dow[x.day_of_week]}\n{x:%d.%m}" for x in d.date]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
xs = np.arange(7)
ax.bar(xs, res["actual"][0] / 1e3, color="#" + BLUE, width=0.6, alpha=0.9, label="факт")
ax.plot(xs, res["model"][0] / 1e3, color="#" + ORANGE, lw=3.5, marker="o", ms=8, label="модель")
ax.plot(xs, res["baseline"][0] / 1e3, color="#" + MUTED, lw=3, ls="--", marker="o", ms=7, label="простой прогноз")
pens = np.flatnonzero(d.is_pension_day.to_numpy())
ax.axvspan(pens.min() - 0.45, pens.max() + 0.45, color="#" + AQUA, alpha=0.1, lw=0)
ax.text((pens.min() + pens.max()) / 2, ax.get_ylim()[1] * 0.97, "выплата пенсий", ha="center", va="top",
        color="#127a56", fontsize=14, fontweight="bold")
ax.set_xticks(xs, labels)
ax.set_ylabel("тыс. сом в день")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.13), fontsize=13)
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(PRES_FIG / "pension_week.png", dpi=200); plt.close(fig)
week = {k: res[k][0].sum() for k in ("actual", "model", "baseline")}
rec = {k: np.ceil(week[k] * 1.15 / 1e4) * 1e4 for k in ("model", "baseline")}
shortfall_mln = f"{(week['actual'] - rec['baseline']) / 1e6:.2f}".replace(".", ",")

# 3) Скриншот демо: только основная колонка (график и рекомендация), без боковой панели
shot = Image.open(FIG / "18_demo_atm0053.png")
sw, sh = shot.size
shot.crop((int(sw * 0.265), int(sh * 0.26), int(sw * 0.95), int(sh * 0.935))).save(PRES_FIG / "demo_crop.png")

# ------------------------------------------------------------------ helpers для pptx
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = 13.333


def rgb(h):
    return RGBColor.from_string(h)


def text(slide, x, y, w, h, parts, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=1.0):
    """parts: строка или список абзацев; абзац — строка или список (текст, {size,color,bold}) кусков."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    paras = parts if isinstance(parts, list) else [parts]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        runs = para if isinstance(para, list) else [(para, {})]
        for run_text, style in runs:
            r = p.add_run()
            r.text = run_text
            f = r.font
            f.name = FONT
            f.size = Pt(style.get("size", size))
            f.bold = style.get("bold", bold)
            f.color.rgb = rgb(style.get("color", color))
    return tb


def box(slide, x, y, w, h, fill=SOFT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(fill)
    if line:
        sh.line.color.rgb = rgb(line)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = 0.06
    return sh


def picture(slide, path, x, y, max_w, max_h):
    """Картинка, вписанная в прямоугольник с сохранением пропорций, по центру."""
    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    slide.shapes.add_picture(str(path), Inches(x + (max_w - w) / 2), Inches(y + (max_h - h) / 2), Inches(w), Inches(h))


def new_slide(title, kicker=None, n=None):
    s = prs.slides.add_slide(BLANK)
    if kicker:
        text(s, 0.7, 0.45, 10, 0.4, kicker.upper(), size=13, color=ORANGE, bold=True)
    text(s, 0.7, 0.8, W - 1.4, 1.0, title, size=34, bold=True)
    if n:
        text(s, W - 1.2, 7.0, 0.6, 0.3, str(n), size=12, color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def big_number(slide, x, y, w, value, caption, color=ORANGE, size=60, cap_size=16):
    text(slide, x, y, w, 1.1, value, size=size, bold=True, color=color)
    text(slide, x, y + size / 72 * 1.25, w, 1.2, caption, size=cap_size, color=INK2)


def fmt(v):
    return f"{v:,.0f}".replace(",", " ")


# ------------------------------------------------------------------ 1. Проблема
s = new_slide("Банкомат пуст — клиент уходит.\nПереполнен — деньги лежат без дела", "Проблема", 1)
for i, (val, cap, col, icon) in enumerate([
    ("+87%", "выдача в дни, когда банкомат опустел:\nон пустеет как раз в пиковый спрос", ORANGE, "Пусто"),
    ("760 тыс. сом", "в среднем остаются без дела за неделю\nна банкомат при планировании «как обычно»", BLUE, "Переполнен"),
]):
    x = 0.7 + i * 6.1
    box(s, x, 2.6, 5.7, 3.1)
    text(s, x + 0.45, 2.9, 5, 0.4, icon.upper(), size=14, bold=True, color=MUTED)
    big_number(s, x + 0.45, 3.35, 5.0, val, cap, color=col, size=54)
text(s, 0.7, 6.15, 12, 0.6, [[("Задача: ", {"bold": True}),
                             ("предсказать выдачу каждого банкомата и подсказать, сколько наличных загрузить на неделю.", {})]],
     size=20, color=INK)

# ------------------------------------------------------------------ 2. Данные
s = new_slide("Датасет AI Academy: банкоматы Бишкека", "Данные", 2)
stats = [("120", "банкоматов"), ("2 года", "2024–2025, 731 день"), ("12,6 млн", "транзакций"), ("3", "внешних источника:\nпогода · курс · карта")]
for i, (v, c) in enumerate(stats):
    big_number(s, 0.7 + i * 3.1, 2.0, 3.0, v, c, color=BLUE if i != 2 else ORANGE, size=44)
text(s, 0.7, 4.35, 7.3, 0.4, "Почему этот датасет", size=18, bold=True)
text(s, 0.7, 4.85, 7.3, 1.8, [
    "•  реальный календарь КР: праздники, Орозо айт, дни пенсий и зарплат",
    "•  погода ERA5, курс НБКР, координаты OpenStreetMap",
    "•  есть разметка сбоев: банкомат пуст или не работает",
], size=16, color=INK2, spacing=1.15)
box(s, 8.4, 4.3, 4.25, 1.75, fill="FDF0EA")
text(s, 8.7, 4.5, 3.7, 1.4, [[("Честно: ", {"bold": True, "color": ORANGE}),
                              ("данные синтетические, но построены на реальном контексте Бишкека. На реальных данных ошибка, скорее всего, будет выше.", {})]],
     size=15, color=INK)

# ------------------------------------------------------------------ 3. Постановка
s = new_slide("Сколько сом выдаст банкомат завтра?", "Постановка задачи", 3)
text(s, 0.7, 1.75, 12, 0.5, [[("Регрессия. ", {"bold": True, "color": ORANGE}),
                             ("Цель — выдача банкомата за день. Прогноз на день D делаем вечером D−1: никакой информации из будущего.", {})]],
     size=17, color=INK2)
groups = [("История банкомата", "выдача вчера, 7, 14\nи 28 дней назад,\nсредние за 7 и 28 дней"),
          ("Календарь", "день недели и месяца,\nпраздники, пенсии, зарплата"),
          ("Банкомат", "тип точки, район,\nна улице или в помещении"),
          ("Погода и курс", "вчерашние погода\nи скачки курса")]
for i, (t, c) in enumerate(groups):
    x = 0.7 + i * 3.05
    box(s, x, 2.55, 2.85, 1.75)
    text(s, x + 0.25, 2.75, 2.4, 0.4, t, size=17, bold=True, color=BLUE if i else ORANGE)
    text(s, x + 0.25, 3.25, 2.45, 1.0, c, size=14, color=INK2)
text(s, 0.7, 4.75, 6, 0.4, "Разбиение по времени", size=18, bold=True)
segments = [("Обучение", "янв 2024 – апр 2025", 16, BLUE), ("Валидация", "май–июнь", 2, AQUA), ("Test", "июль–декабрь 2025", 6, ORANGE)]
x0, total_w = 0.7, 11.9
for name, period, months, col in segments:
    w = total_w * months / 24
    b = box(s, x0, 5.3, w - 0.06, 0.62, fill=col, shape=MSO_SHAPE.RECTANGLE)
    if months > 2:
        text(s, x0 + 0.15, 5.3, w - 0.3, 0.62, name, size=15, bold=True, color="FFFFFF", anchor=MSO_ANCHOR.MIDDLE)
    else:  # узкий сегмент: подпись над полосой
        text(s, x0 - 0.3, 4.95, w + 0.6, 0.3, name, size=13, bold=True, color=col, align=PP_ALIGN.CENTER)
    text(s, x0, 6.02, w, 0.4, period, size=13, color=INK2)
    x0 += w
text(s, 0.7, 6.55, 12, 0.4, [[("Метрика — WAPE", {"bold": True}),
                             (" = сумма ошибок / сумма выдачи: на сколько процентов ошибаемся в общем объёме денег.", {})]],
     size=14, color=INK2)

# ------------------------------------------------------------------ 4. EDA
s = new_slide("Три главных вывода о спросе на наличные", "Анализ данных", 4)
cols = [("×2,6", "рынки и ТЦ выдают в 2–3 раза\nбольше, чем АЗС и Соцфонд", PRES_FIG / "types.png"),
        ("+61% / +47%", "предпраздник / пенсии, зарплата +38%:\nпики известны заранее по календарю", PRES_FIG / "special_days.png"),
        ("+18,5%", "рост выдачи на банкомат\nв 2025 году к 2024-му", PRES_FIG / "growth.png")]
for i, (v, c, img) in enumerate(cols):
    x = 0.55 + i * 4.15
    text(s, x, 1.85, 4.0, 0.7, v, size=40, bold=True, color=ORANGE)
    text(s, x, 2.6, 4.0, 0.8, c, size=15, color=INK2)
    picture(s, img, x, 3.35, 3.95, 3.05)
text(s, 0.55, 6.6, 12.3, 0.35, [[("Качество данных: ", {"bold": True, "color": INK}),
                               ("дублей и логических ошибок нет; выбросы — реальные пики (пенсии, зарплаты), "
                                "поэтому не удалялись — обрезка ухудшала валидацию", {})]],
     size=12, color=INK2)

# ------------------------------------------------------------------ 5. Модели
s = new_slide("LightGBM на 21% точнее простого прогноза", "Модели", 5)
rows = [("Модель", "MAE, сом", "WAPE"),
        ("Naive: как 7 дней назад", "218 028", "38,2%"),
        ("Baseline: среднее по дню недели (4 нед.)", "168 136", "29,5%"),
        ("Ridge (линейная)", "162 104", "28,4%"),
        ("LightGBM", "133 229", "23,4%")]
tbl = s.shapes.add_table(len(rows), 3, Inches(0.7), Inches(2.0), Inches(7.9), Inches(3.6)).table
tbl.columns[0].width, tbl.columns[1].width, tbl.columns[2].width = Inches(4.7), Inches(1.7), Inches(1.5)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = rgb("FFFFFF" if r not in (0, 4) else (INK if r == 0 else "FDF0EA"))
        cell.margin_left = cell.margin_right = Inches(0.15)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
        run = p.add_run()
        run.text = val
        run.font.name, run.font.size = FONT, Pt(16 if r else 14)
        run.font.bold = r in (0, 4)
        run.font.color.rgb = rgb("FFFFFF" if r == 0 else (ORANGE if r == 4 and c else INK))
text(s, 0.7, 5.8, 7.9, 0.5, "Test: июль–декабрь 2025, дни без сбоев. Средняя выдача — 570 тыс. сом в день.", size=13, color=MUTED)
big_number(s, 9.3, 2.0, 3.6, "23,4%", "WAPE LightGBM —\nошибка в общем объёме денег", color=ORANGE, size=58)
big_number(s, 9.3, 4.25, 3.6, "−21%", "ошибки по сравнению\nс лучшим простым прогнозом", color=BLUE, size=58)

# ------------------------------------------------------------------ 6. Оптимизация
s = new_slide("Не хватало информации, а не настройки", "Оптимизация", 6)
items = [("−0,56 п.п.", "новые признаки:\nдень месяца и выдача 28 дней назад", ORANGE),
         ("−0,09 п.п.", "Optuna: 45 попыток\nподбора параметров", MUTED),
         ("22,7%", "итоговая WAPE на test\n(было 23,4%)", BLUE)]
for i, (v, c, col) in enumerate(items):
    big_number(s, 0.7, 1.95 + i * 1.65, 4.6, v, c, color=col, size=46, cap_size=15)
text(s, 5.7, 1.95, 7.0, 0.5, "Почему день месяца: пенсии и зарплаты привязаны к числу", size=16, bold=True)
picture(s, PRES_FIG / "day_of_month.png", 5.6, 2.45, 7.2, 3.6)
text(s, 5.7, 6.15, 7.0, 0.7, "10–14 число выдача на 35–50% выше обычной, 25–27 — на 11–21%. Модель этого не знала, а подбор параметров это не исправит.",
     size=14, color=INK2)

# ------------------------------------------------------------------ 7. Результат для бизнеса
s = new_slide("Загрузки по модели хватает чаще", "Результат для бизнеса", 7)
text(s, 0.7, 1.95, 4.8, 1.2, [[("91%", {"color": ORANGE}), ("  vs  ", {"size": 28, "color": MUTED}), ("85%", {"color": MUTED})]],
     size=60, bold=True)
text(s, 0.7, 3.0, 4.6, 0.8, "недель, когда загрузки (прогноз + 15%) хватает до конца недели", size=15, color=INK2)
# Значения из notebooks/04_robustness.ipynb, раздел 4 (запас выбран на июле–сентябре, проверен на октябре–декабре)
text(s, 0.7, 4.0, 4.8, 0.7, [[("−40%", {"color": BLUE}), ("  ≈68 млн сом", {"size": 22, "color": MUTED})]],
     size=36, bold=True)
text(s, 0.7, 4.7, 4.6, 0.6, "лишних наличных при надёжности 95% · экономия на сеть из 120 банкоматов", size=15, color=INK2)
text(s, 0.7, 5.45, 4.8, 0.7, [[("9,4%", {"color": BLUE}), (" vs ", {"size": 22, "color": MUTED}), ("14,4%", {"color": MUTED})]],
     size=36, bold=True)
text(s, 0.7, 6.15, 4.6, 0.6, "ошибка суммы за неделю · модель vs простой прогноз", size=15, color=INK2)
text(s, 5.9, 1.95, 7.0, 0.5, f"Соцфонд {DEMO_ATM}, неделя пенсий 08–14.08.2025", size=16, bold=True)
picture(s, PRES_FIG / "pension_week.png", 5.8, 2.4, 7.0, 3.95)
text(s, 5.9, 6.4, 7.0, 0.7, [[("Модель: ", {"bold": True, "color": ORANGE}), (f"загрузка {fmt(rec['model'] / 1e3)} тыс. — хватило.  ", {}),
                             ("Простой прогноз: ", {"bold": True, "color": MUTED}),
                             (f"{fmt(rec['baseline'] / 1e3)} тыс. — не хватило бы {shortfall_mln} млн сом.", {})]],
     size=14, color=INK2)

# ------------------------------------------------------------------ 8. Демо и дальше
s = new_slide("Где модель ошибается и что дальше", "Итог", 8)
# Демо покажу вживую, поэтому скриншот маленький
picture(s, PRES_FIG / "demo_crop.png", 0.7, 1.9, 3.7, 2.5)
text(s, 0.7, 4.6, 3.7, 0.4, "Живое демо", size=16, bold=True)
text(s, 0.7, 5.0, 3.7, 1.2, [[("streamlit run app.py", {"bold": True, "color": INK})],
                             "банкомат и неделя → прогноз, сколько загрузить, точность"],
     size=13, color=INK2, spacing=1.1)

box(s, 4.85, 1.85, 4.45, 4.85)
text(s, 5.15, 2.05, 3.9, 0.4, "Где модель ошибается", size=18, bold=True)
errors = [("Праздники — WAPE 26%", "в train всего 28 праздничных дат", ORANGE),
          ("Малый поток клиентов", "офисы и Соцфонд: пара крупных снятий\nсдвигает итог дня", ORANGE),
          ("С сентября переоценка на 1–2%", "модель помнит летний уровень спроса\n→ переобучать раз в месяц", BLUE)]
for i, (head, sub, col) in enumerate(errors):
    y = 2.65 + i * 1.3
    text(s, 5.15, y, 3.9, 0.4, head, size=16, bold=True, color=col)
    text(s, 5.15, y + 0.42, 3.9, 0.8, sub, size=13, color=INK2)

text(s, 9.7, 2.05, 3.1, 0.4, "Что дальше", size=18, bold=True)
text(s, 9.7, 2.65, 3.1, 4.0, [
    "•  признаки конкретных праздников: одни дают пик, другие — тишину",
    "•  переобучение раз в месяц",
    "•  от прогноза — к расписанию инкассаций с учётом ёмкости банкомата",
], size=14, color=INK2, spacing=1.15)

# ------------------------------------------------------------------ 9–10. Приложение (показывать только при вопросах)
# Графики из notebooks/04_robustness.ipynb
appendix = [
    ("Проверка по кварталам", FIG / "19_backtest_quarters.png",
     "Модель училась только на данных до начала квартала. В каждом квартале 2025 она на 22–25% точнее "
     "простого прогноза — результат не зависит от удачного периода проверки."),
    ("Запас при загрузке", FIG / "22_buffer_tradeoff.png",
     "Запас выбран на июле–сентябре, проверен на октябре–декабре. Чтобы 95% недель банкомат не пустел, модели нужен "
     "запас 21%, простому прогнозу — 35%; без дела остаётся 0,86 млн против 1,43 млн сом на банкомат."),
]
for i, (kicker, img, caption) in enumerate(appendix, 9):
    s = new_slide("Приложение", kicker, i)
    picture(s, img, 0.7, 1.85, W - 1.4, 4.3)
    text(s, 0.7, 6.3, W - 1.4, 0.7, caption, size=15, color=INK2)

# ------------------------------------------------------------------ заметки докладчика
# Всё выступление ≤ 5 минут: 3,5 мин слайды + 1,5 мин демо после слайда 7.
NOTES = {
    1: ("~25 с (0:00–0:25)",
        "Банку нужно решать, сколько наличных положить в банкомат. Положишь мало — банкомат пустеет как раз "
        "в дни наплыва, когда выдача на 87% выше обычной, и клиент уходит. Положишь много — деньги лежат без дела, "
        "около 760 тысяч сом на банкомат за неделю. Моя задача — предсказать выдачу и подсказать, сколько загрузить."),
    2: ("~25 с (0:25–0:50)",
        "Я взял открытый датасет AI Academy: 120 банкоматов Бишкека за два года, 12,6 миллиона транзакций, "
        "плюс погода, курс и карта. Выбрал его, потому что в нём настоящий календарь Кыргызстана — праздники, "
        "пенсии, зарплаты — и размечены сбои банкоматов. Честно скажу: данные синтетические, поэтому на реальных "
        "ошибка будет выше."),
    3: ("~25 с (0:50–1:15)",
        "Это регрессия: сколько сом банкомат выдаст завтра. Прогноз я делаю вечером накануне, так что модель видит "
        "только прошлое. Признаки — история банкомата, календарь, тип и район, погода. Данные делю по времени: "
        "учу до мая 2025, на мае–июне настраиваю, а проверяю на втором полугодии, которое модель не видела. "
        "Метрика — WAPE, ошибка в процентах от всей суммы."),
    4: ("~35 с (1:15–1:50)",
        "Три вывода из анализа. Первое — всё решает место: рынки и торговые центры выдают в 2–3 раза больше, "
        "чем АЗС и Соцфонд. Второе — календарь даёт предсказуемые пики: перед праздником плюс 61%, в пенсии плюс 47%. "
        "Третье — спрос растёт, в 2025 году на 18,5%. Данные я проверил: ошибок нет, а резкие выбросы оказались "
        "настоящими пиками, поэтому я их оставил."),
    5: ("~25 с (1:50–2:15)",
        "Я сравнил модели от простой к сложной. Прогноз «как неделю назад» ошибается на 38%. Среднее по дню недели — "
        "29,5%, это мой baseline. Линейная модель почти не лучше. LightGBM — 23,4%, на 21% точнее baseline: "
        "он видит сочетания вроде «Соцфонд плюс день пенсий»."),
    6: ("~30 с (2:15–2:45)",
        "Потом я улучшал модель двумя способами. Два новых признака — день месяца и выдача 28 дней назад — "
        "дали минус 0,56 пункта. Подбор параметров через Optuna — всего минус 0,09. Итог — 22,7%. "
        "Вывод: модели не хватало информации, а не настройки. Пенсии и зарплаты привязаны к числу месяца, "
        "а это число модель раньше не знала."),
    7: ("~30 с (2:45–3:15), затем ДЕМО ~1,5 мин (3:15–4:45)",
        "Что это даёт бизнесу. С запасом 15% денег хватает в 91% недель против 85%. А при надёжности 95% "
        "лишних наличных на 40% меньше — около 68 миллионов сом на сеть. Пример — Соцфонд в неделю пенсий: "
        "простой прогноз не заметил пик, не хватило бы 1,57 миллиона. Модель пик увидела. Покажу вживую.\n\n"
        "[ДЕМО, ~1,5 мин] Открыть localhost:8501/?atm=ATM-0053&start=2025-08-08\n"
        "1) График: серая линия ставит пик на субботу, оранжевая растёт вместе с пенсиями.\n"
        "2) «Сколько загрузить»: модели хватило, простому прогнозу не хватило 1,57 млн.\n"
        "3) «Насколько можно доверять»: по сети 22,7% против 29,5%.\n"
        "Запасной пример: ?atm=ATM-0061&start=2025-08-29 — ТЦ перед Днём независимости."),
    8: ("~15 с (4:45–5:00)",
        "Хуже всего модель работает в праздники и на банкоматах с малым потоком. С осени она немного завышает — "
        "значит, её нужно переобучать раз в месяц. Спасибо, готов ответить на вопросы."),
    9: ("ЗАПАСНОЙ — показывать при вопросе «а вдруг повезло с периодом?»",
        "Я проверил модель не на одном периоде, а на каждом квартале 2025 года. Каждый раз она училась только на "
        "данных до начала квартала. Везде модель на 22–25% точнее простого прогноза, ошибка от 22,6 до 25%."),
    10: ("ЗАПАСНОЙ — показывать при вопросе «откуда 15%?» или «сколько это экономит?»",
         "15% — запас для демо. Я подобрал запас на июле–сентябре и проверил на октябре–декабре. Чтобы 95% недель "
         "банкомат не пустел, модели нужно 21% сверху, простому прогнозу — 35%. Без дела остаётся 0,86 миллиона "
         "вместо 1,43 на банкомат: на 40% меньше, около 68 миллионов сом на всю сеть."),
}
for n, slide in enumerate(prs.slides, 1):
    timing, speech = NOTES[n]
    slide.notes_slide.notes_text_frame.text = f"⏱ {timing}\n\n{speech}"
    words = len(speech.split("[ДЕМО")[0].split())
    print(f"слайд {n}: {words} слов, {timing}")


def register_notes_master(prs):
    """python-pptx создаёт мастер заметок, но не вписывает его в presentation.xml.

    По стандарту ссылка на него должна быть в <p:notesMasterIdLst> сразу после <p:sldMasterIdLst>.
    PowerPoint без неё обходится, а Keynote считает файл повреждённым («Неверный формат файла»).
    """
    pres = prs.part._element
    if pres.find(qn("p:notesMasterIdLst")) is not None:
        return
    rid = next(r.rId for r in prs.part.rels.values() if r.reltype == RT.NOTES_MASTER)
    lst = OxmlElement("p:notesMasterIdLst")
    item = OxmlElement("p:notesMasterId")
    item.set(qn("r:id"), rid)
    lst.append(item)
    pres.find(qn("p:sldMasterIdLst")).addnext(lst)


register_notes_master(prs)

prs.save(OUT)
print("Сохранено:", OUT.relative_to(ROOT), f"({OUT.stat().st_size / 1e6:.1f} МБ, {len(prs.slides)} слайдов)")
print(f"Пенсионная неделя: факт {fmt(week['actual'])}, загрузка модель {fmt(rec['model'])}, простой {fmt(rec['baseline'])}")
