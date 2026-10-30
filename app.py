import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import time

# --- 1. НАСТРОЙКИ СТРАНИЦЫ И ПРЕМИУМ-ДИЗАЙН (CSS) ---
st.set_page_config(page_title="AI Cyber Risk Dashboard", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    /* Фон приложения и шрифты */
    .reportview-container {background: #f4f6f9;}
    h1, h2, h3 {color: #1e3d59; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}

    /* Стилизация карточек с метриками */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e4e8;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }

    /* Красивая кнопка */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.4);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #ff4b2b 0%, #ff416c 100%);
        box-shadow: 0 6px 20px rgba(255, 75, 43, 0.6);
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)


# --- 2. КЕШИРОВАНИЕ НЕЙРОСЕТИ (чтобы работало молниеносно) ---
@st.cache_resource
def load_assets():
    model = load_model('cyber_risk_model.h5')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features_list.pkl')
    return model, scaler, features


model, scaler, features_list = load_assets()

# --- 3. ШАПКА САЙТА ---
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=80)  # Иконка щита
with col_title:
    st.title("CyberRisk AI Predictor")
    st.markdown("**Предиктивная аналитика киберугроз на базе Deep Learning (Multi-output Regression)**")

st.divider()

# --- 4. БОКОВОЕ МЕНЮ (Сгруппированное и аккуратное) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/8365/8365022.png", width=60)
st.sidebar.title("Параметры симуляции")

st.sidebar.markdown("### 🏢 Профиль бизнеса")
industry = st.sidebar.selectbox("Индустрия", ['Finance', 'Healthcare', 'Retail', 'Technology', 'Manufacturing'],
                                help="Сфера деятельности компании")
company_size = st.sidebar.selectbox("Масштаб бизнеса", ['Small', 'Medium', 'Large'])

st.sidebar.markdown("### ⚠️ Вектор атаки")
records = st.sidebar.slider("Украдено записей (шт)", 1000, 1000000, 50000, step=5000,
                            help="Сколько данных клиентов утекло")
detection = st.sidebar.slider("Время обнаружения (мин)", 10, 1440, 120,
                              help="Как долго хакеры были в системе до обнаружения")
response = st.sidebar.slider("Время реакции ИТ (мин)", 10, 1440, 60, help="Как быстро отреагировали ИТ-специалисты")

st.sidebar.markdown("### 🛡️ Текущая защита (As-Is)")
mfa = st.sidebar.toggle("Двухфакторная защита (MFA)", value=False)
firewall = st.sidebar.toggle("Корпоративный Firewall", value=False)
training = st.sidebar.toggle("Кибер-тренинги персонала", value=False)

# --- 5. ГЛАВНАЙ ЭКРАН И ЛОГИКА ---
if not st.sidebar.button("🚀 ЗАПУСТИТЬ ИИ-АНАЛИЗ"):
    # Пустой экран до нажатия кнопки
    st.info(
        "👈 Настройте параметры инфраструктуры в левом меню и нажмите **«Запустить ИИ-анализ»**, чтобы увидеть магию нейросетей.")

    st.markdown("### Что умеет наша модель?")
    col1, col2, col3 = st.columns(3)
    col1.markdown("💰 **Прогноз убытков**<br>Точный расчет прямых и скрытых финансовых потерь в долларах США.",
                  unsafe_allow_html=True)
    col2.markdown("⏱ **Downtime-анализ**<br>Оценка времени паралича серверов и бизнес-процессов.",
                  unsafe_allow_html=True)
    col3.markdown("💡 **What-If симулятор**<br>Мгновенный расчет окупаемости инвестиций в кибербезопасность.",
                  unsafe_allow_html=True)

else:
    # --- ИМИТАЦИЯ ЗАГРУЗКИ ДЛЯ "ВАУ" ЭФФЕКТА НА ЗАЩИТЕ ---
    with st.spinner('Нейросеть анализирует векторы атаки и пересчитывает веса...'):
        time.sleep(1.5)  # Искусственная пауза для красоты

    # Готовим данные
    input_data = {
        'MFA': [1 if mfa else 0], 'Firewall': [1 if firewall else 0], 'Security_Training': [1 if training else 0],
        'Detection_Time_Min': [detection], 'Response_Time_Min': [response], 'Records_Compromised': [records],
        'Industry': [industry], 'Company_Size': [company_size]
    }
    df_input = pd.DataFrame(input_data)
    df_encoded = pd.get_dummies(df_input, columns=['Industry', 'Company_Size'])

    for col in features_list:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[features_list]

    # 1. Прогноз "КАК ЕСТЬ" (As-Is)
    pred_current = model.predict(scaler.transform(df_encoded), verbose=0)[0]
    loss_current, downtime_current = pred_current[0], pred_current[1]

    # 2. Прогноз "ИДЕАЛЬНАЯ ЗАЩИТА" (To-Be)
    df_ideal = df_encoded.copy()
    df_ideal['MFA'], df_ideal['Firewall'], df_ideal['Security_Training'], df_ideal['Response_Time_Min'] = 1, 1, 1, 10
    pred_ideal = model.predict(scaler.transform(df_ideal), verbose=0)[0]
    loss_ideal, downtime_ideal = pred_ideal[0], pred_ideal[1]

    saved_money = loss_current - loss_ideal
    saved_time = downtime_current - downtime_ideal

    st.success("✅ Анализ успешно завершен!")

    # --- ВКЛАДКИ (TABS) ДЛЯ СТРУКТУРИРОВАНИЯ ИНФОРМАЦИИ ---
    tab1, tab2, tab3 = st.tabs(["📊 Текущие риски (As-Is)", "💡 Симулятор инвестиций (What-If)", "🧠 Архитектура модели"])

    with tab1:
        st.markdown("### Прогнозируемый ущерб при текущих настройках безопасности")
        m_col1, m_col2 = st.columns(2)
        # Красивые метрики
        m_col1.metric("💸 Финансовые потери", f"${loss_current:,.0f}")
        m_col2.metric("⏱ Время простоя бизнеса", f"{downtime_current:.1f} ч.")

        st.caption("Данные рассчитаны на основе архитектуры Deep Learning с функцией Huber Loss.")

    with tab2:
        st.markdown("### Оценка окупаемости ИТ-бюджета (ROI)")
        st.write("Что будет, если прямо сейчас внедрить корпоративный Firewall, MFA и обучить сотрудников?")

        # Метрики с индикатором разницы (Delta)
        d_col1, d_col2 = st.columns(2)
        d_col1.metric("Новый финансовый ущерб", f"${loss_ideal:,.0f}", delta=f"-${saved_money:,.0f} (Экономия)",
                      delta_color="inverse")
        d_col2.metric("Новое время простоя", f"{downtime_ideal:.1f} ч.", delta=f"-{saved_time:.1f} ч. (Сэкономлено)",
                      delta_color="inverse")

        # ИНТЕРАКТИВНЫЙ ГРАФИК PLOTLY
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Финансовые потери (USD)'], y=[loss_current],
            name='До (Без защиты)', marker_color='#ff4757', text=f"${loss_current:,.0f}", textposition='auto'
        ))
        fig.add_trace(go.Bar(
            x=['Финансовые потери (USD)'], y=[loss_ideal],
            name='После (С защитой)', marker_color='#2ed573', text=f"${loss_ideal:,.0f}", textposition='auto'
        ))
        fig.update_layout(
            title='Сравнение сценариев ущерба',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=(dict(showgrid=True, gridcolor='#e0e4e8'))
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### ⚙️ Под капотом")
        st.code("""
# Архитектура Keras (Көп мақсатты регрессия)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(2) # Выходной слой на 2 нейрона (Деньги и Часы)
])
model.compile(optimizer='adam', loss=keras.losses.Huber())
        """, language="python")
        st.info(
            "Модель использует функцию **Huber Loss**, которая устойчива к сильным выбросам (например, атакам с убытками в миллиарды долларов), и механизм **Early Stopping** для предотвращения переобучения.")