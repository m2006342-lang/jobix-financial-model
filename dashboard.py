import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="Jobix Financial Model Simulator",
    layout="wide"
)

st.title("🧮 Финансовый симулятор стартапа 'Jobix'")
st.markdown("---")

# --- БОКОВАЯ ПАНЕЛЬ С "ПОЛЗУНКАМИ" (ПРЕДПОСЫЛКИ) ---
st.sidebar.header("🕹️ Панель управления")
st.sidebar.markdown("**Изменяйте предпосылки и смотрите, что будет**")

# --- Раздел 1: Воронка ---
st.sidebar.subheader("Воронка привлечения")
monetization_rate = st.sidebar.slider("Конверсия в покупку (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5) / 100
monthly_growth = st.sidebar.slider("Ежемесячный органический рост (%)", min_value=1.0, max_value=30.0, value=10.0, step=1.0) / 100
peak_season_coeff = st.sidebar.slider("Коэф. сезонности (%)", min_value=50.0, max_value=300.0, value=150.0, step=10.0) / 100

# --- Раздел 2: Маркетинг ---
st.sidebar.subheader("Маркетинг")
campaign_budget = st.sidebar.number_input("Бюджет на 1 кампанию (₽)", min_value=50000, max_value=1000000, value=200000, step=10000)
cpa = st.sidebar.number_input("Стоимость привлечения (CPA) (₽)", min_value=10, max_value=500, value=100, step=5)

# --- Раздел 3: Экономика ---
st.sidebar.subheader("Юнит-экономика и WACC")
aov = st.sidebar.number_input("Средний чек (AOV) (₽)", min_value=300, max_value=1000, value=699, step=10)
api_cost_per_diamond = st.sidebar.slider("Стоимость API за алмаз (₽)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
wacc = st.sidebar.slider("Ставка дисконтирования (WACC) (%/год)", min_value=15.0, max_value=50.0, value=25.0, step=1.0) / 100


# --- ОСНОВНАЯ ФУНКЦИЯ-КАЛЬКУЛЯТОР (НАШ ДВИЖОК) ---
# Она почти не изменилась, просто берет предпосылки из ползунков
@st.cache_data # Кэшируем результаты, чтобы все летало
def calculate_financial_model(assumptions):
    # (Здесь тот же самый код расчета, что и в прошлый раз, но я немного его почистил)
    assumptions["avg_diamonds_purchased"] = (15 * 0.8 + 50 * 0.2) # Упростим, так как они не меняются
    
    months = pd.date_range(start="2026-01-01", periods=36, freq='ME')
    pnl = pd.DataFrame(index=[
        "Всего новые пользователи", "Новые платящие клиенты", "Выручка",
        "Стоимость API", "Комиссия платежей", "Валовая прибыль",
        "Постоянные издержки", "Маркетинг (прямые затраты)",
        "Операционная прибыль (EBITDA)", "Налог", "Чистая прибыль"
    ], columns=months)

    base_organic_users = 1000
    for i, month in enumerate(months):
        if i > 0:
            base_organic_users *= (1 + assumptions["monthly_growth"])
        
        is_peak_season = month.month in [1, 2, 9, 10]
        seasonal_boost = base_organic_users * assumptions["peak_season_coeff"] if is_peak_season else 0
        organic_total = base_organic_users + seasonal_boost
        
        marketing_spend = 0
        if month.month == 12 or month.month == 8:
            marketing_spend = assumptions["campaign_budget"]
        pnl.loc["Маркетинг (прямые затраты)", month] = marketing_spend
        
        paid_users = marketing_spend / assumptions["cpa"] if assumptions["cpa"] > 0 else 0
        total_new_users = organic_total + paid_users
        pnl.loc["Всего новые пользователи", month] = total_new_users
        
        new_paying = total_new_users * assumptions["monetization_rate"]
        pnl.loc["Новые платящие клиенты", month] = new_paying
        pnl.loc["Выручка", month] = new_paying * assumptions["aov"]
        
        total_diamonds_used = (total_new_users * 5 + new_paying * assumptions["avg_diamonds_purchased"])
        pnl.loc["Стоимость API", month] = total_diamonds_used * assumptions["api_cost_per_diamond"]
        pnl.loc["Комиссия платежей", month] = pnl.loc["Выручка", month] * 0.035
        pnl.loc["Валовая прибыль", month] = pnl.loc["Выручка", month] - pnl.loc["Стоимость API", month] - pnl.loc["Комиссия платежей", month]
        
        pnl.loc["Постоянные издержки", month] = 8000 + 1000
        
        pnl.loc["Операционная прибыль (EBITDA)", month] = pnl.loc["Валовая прибыль", month] - pnl.loc["Постоянные издержки", month] - pnl.loc["Маркетинг (прямые затраты)", month]
        
        pnl.loc["Налог", month] = pnl.loc["Выручка", month] * 0.04 if pnl.loc["Выручка", month] > 0 else 0
        pnl.loc["Чистая прибыль", month] = pnl.loc["Операционная прибыль (EBITDA)", month] - pnl.loc["Налог", month]

    cash_flow = pnl.loc["Чистая прибыль"].copy()
    start_date = months[0] - pd.DateOffset(months=1)
    cash_flow[start_date] = -assumptions["campaign_budget"]
    cash_flow = cash_flow.sort_index()
    
    return pnl, cash_flow

# --- СОБИРАЕМ АКТУАЛЬНЫЕ ПРЕДПОСЫЛКИ С ПОЛЗУНКОВ ---
current_assumptions = {
    "monetization_rate": monetization_rate,
    "monthly_growth": monthly_growth,
    "peak_season_coeff": peak_season_coeff,
    "campaign_budget": campaign_budget,
    "cpa": cpa,
    "aov": aov,
    "api_cost_per_diamond": api_cost_per_diamond,
    "wacc": wacc,
    # Добавляем остальные, которые не меняются
    "base_new_users": 1000, "forecast_periods": 36, "free_diamonds_on_signup": 5,
    "infrastructure_cost": 8000, "legal_cost": 1000, "tax_rate": 0.04,
}

# --- ВЫПОЛНЯЕМ РАСЧЕТЫ ---
pnl, cash_flow = calculate_financial_model(current_assumptions)

# --- РАСЧЕТ NPV ---
monthly_wacc = (1 + wacc)**(1/12) - 1
npv = 0
for i, cf in enumerate(cash_flow):
    npv += cf / (1 + monthly_wacc)**i

# --- ВИЗУАЛИЗАЦИЯ (ДАШБОРД) ---
st.header("📈 Ключевые показатели эффективности")

col1, col2, col3 = st.columns(3)
col1.metric("NPV (Чистая приведенная стоимость)", f"{npv:,.0f} ₽")
col2.metric("Итоговая выручка (3 года)", f"{pnl.loc['Выручка'].sum()/1_000_000:.2f} млн ₽")
col3.metric("Итоговая чистая прибыль (3 года)", f"{pnl.loc['Чистая прибыль'].sum()/1_000_000:.2f} млн ₽")

st.markdown("---")

# --- НОВЫЙ БЛОК С ФОРМУЛАМИ ---
with st.expander("🔬 Показать методологию расчета"):
    st.subheader("Формула расчета выручки")
    st.latex(r'''
    \text{Выручка} = \text{Новые платящие} \times \text{Средний чек (AOV)}
    ''')
    st.markdown("где `Новые платящие` = `Всего новые пользователи` * `Конверсия в покупку`")

    st.subheader("Формула расчета Чистой Прибыли")
    st.latex(r'''
    \text{Чистая прибыль} = \text{Валовая прибыль} - \text{Постоянные издержки} - \text{Маркетинг} - \text{Налог}
    ''')
    st.markdown("где `Валовая прибыль` = `Выручка` - `Переменные издержки (API + Комиссии)`")
    
    st.subheader("Формула расчета NPV")
    st.latex(r'''
    NPV = \sum_{i=0}^{n} \frac{CF_i}{(1 + r)^i}
    ''')
    st.markdown("где `CF_i` - чистый денежный поток в месяце `i`, а `r` - месячная ставка дисконтирования.")
# --- КОНЕЦ НОВОГО БЛОКА ---

st.header("📊 Динамика денежного потока")
st.line_chart(cash_flow.cumsum())

st.header("📜 Детальный P&L (Отчет о прибылях и убытках)")
st.dataframe(pnl.transpose().round(0))

st.sidebar.markdown("---")
st.sidebar.info("Этот дашборд — пример того, как Python-скрипт превращается в интерактивный инструмент для принятия решений.")