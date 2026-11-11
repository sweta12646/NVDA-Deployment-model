

# ============================================================
# NVDA STOCK INTELLIGENCE
# PROFESSIONAL FINANCIAL ANALYTICS DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NVDA | Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. PROFESSIONAL DARK THEME
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0B1120;
    color: #E5E7EB;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #253247;
}

[data-testid="stSidebar"] * {
    color: #E5E7EB;
}

.main-title {
    font-size: 32px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -1px;
}

.subtitle {
    color: #94A3B8;
    font-size: 14px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 19px;
    font-weight: 700;
    color: #FFFFFF;
    margin-top: 15px;
}

.metric-card {
    background: linear-gradient(145deg, #172337, #111C2D);
    border: 1px solid #293B50;
    border-radius: 12px;
    padding: 18px;
    min-height: 110px;
}

.metric-label {
    color: #94A3B8;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
}

.metric-value {
    color: #FFFFFF;
    font-size: 26px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-caption {
    color: #76B900;
    font-size: 12px;
    margin-top: 5px;
}

div[data-testid="stMetric"] {
    background: #172337;
    border: 1px solid #293B50;
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF;
}

.stButton > button {
    background-color: #76B900;
    color: #07110A;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    min-height: 42px;
}

.stButton > button:hover {
    background-color: #8DD000;
    color: #07110A;
}

div[data-testid="stTabs"] button {
    color: #94A3B8;
}

hr {
    border-color: #293B50;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_bundle():

    return joblib.load("nvda_models.pkl")


try:

    bundle = load_bundle()

except FileNotFoundError:

    st.error(
        "Model file not found. Please run train_model.py first."
    )

    st.stop()

except Exception as error:

    st.error(f"Unable to load models: {error}")

    st.stop()


regression_models = bundle["regression_models"]

classification_models = bundle["classification_models"]

regression_results = bundle["regression_results"]

classification_results = bundle["classification_results"]

regression_predictions = bundle["regression_predictions"]

classification_predictions = bundle["classification_predictions"]

latest_features = bundle["latest_features"]

latest_close = float(bundle["latest_close"])

latest_date = pd.to_datetime(bundle["latest_date"])

history = bundle["history"].copy()

history["Date"] = pd.to_datetime(history["Date"])

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def format_currency(value):

    return f"${value:,.2f}"


def metric_card(label, value, caption=""):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def chart_layout(fig, height=400):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111C2D",
        plot_bgcolor="#111C2D",
        font=dict(
            color="#E5E7EB",
            family="Inter"
        ),
        height=height,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis=dict(
            gridcolor="#26364A"
        ),
        yaxis=dict(
            gridcolor="#26364A"
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )

    return fig


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:10px 0 20px 0;">
            <div style="color:#76B900;
                        font-size:25px;
                        font-weight:800;">
                NVDA.
            </div>
            <div style="color:#94A3B8;
                        font-size:11px;
                        letter-spacing:2px;">
                STOCK INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Executive Overview",
            "Market Explorer",
            "AI Price Forecast",
            "Market Direction",
            "Model Laboratory",
            "Data Center"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### MODEL STATUS")

    st.success("All 6 models loaded")

    st.caption(
        f"Dataset: {bundle.get('source_file', 'NVDA CSV')}"
    )

    st.caption(
        f"Last record: {latest_date.strftime('%d %b %Y')}"
    )

    st.divider()

    st.caption(
        "NVDA STOCK INTELLIGENCE | MACHINE LEARNING PROJECT"
    )

# ============================================================
# 6. EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.markdown(
        '<div class="main-title">NVIDIA Corporation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'NASDAQ: NVDA | Executive Market Intelligence Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    previous_close = float(
        history["Close"].iloc[-2]
    )

    daily_change = latest_close - previous_close

    daily_percentage = (
        daily_change / previous_close
    ) * 100

    total_records = len(history)

    start_price = float(history["Close"].iloc[0])

    overall_return = (
        (latest_close - start_price) / start_price
    ) * 100

    # KPI CARDS

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "LATEST CLOSE",
            format_currency(latest_close),
            "Historical closing price"
        )

    with c2:
        metric_card(
            "DAILY CHANGE",
            f"{daily_change:+.2f}",
            f"{daily_percentage:+.2f}%"
        )

    with c3:
        metric_card(
            "HISTORICAL RECORDS",
            f"{total_records:,}",
            "Available trading records"
        )

    with c4:
        metric_card(
            "OVERALL PRICE CHANGE",
            f"{overall_return:+.2f}%",
            "First to latest available close"
        )

    st.markdown(
        '<div class="section-title">Historical Market Performance</div>',
        unsafe_allow_html=True
    )

    # PRICE CHART

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["Date"],
            y=history["Close"],
            name="Closing Price",
            mode="lines",
            line=dict(
                color="#76B900",
                width=2
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=history["Date"],
            y=history["Close"].rolling(20).mean(),
            name="20-Day Moving Average",
            mode="lines",
            line=dict(
                color="#60A5FA",
                width=1.5,
                dash="dash"
            )
        )
    )

    fig.update_layout(
        title="NVDA Historical Closing Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified"
    )

    st.plotly_chart(
        chart_layout(fig, 450),
        use_container_width=True
    )

    # LOWER SECTION

    left, right = st.columns(2)

    with left:

        st.markdown("### Market Statistics")

        statistics = pd.DataFrame({
            "Metric": [
                "Highest Price",
                "Lowest Price",
                "Average Price",
                "Average Volume",
                "Latest Trading Date"
            ],

            "Value": [
                format_currency(history["High"].max()),
                format_currency(history["Low"].min()),
                format_currency(history["Close"].mean()),
                f"{history['Volume'].mean():,.0f}",
                latest_date.strftime("%d %B %Y")
            ]
        })

        st.dataframe(
            statistics,
            hide_index=True,
            use_container_width=True
        )

    with right:

        st.markdown("### Project Information")

        st.info(
            """
            **Prediction Engine**

            - 3 Regression Algorithms
            - 3 Classification Algorithms
            - Chronological Train/Test Split
            - Historical Model Evaluation
            - Interactive Financial Analytics
            """
        )

        st.metric(
            "Training Records",
            f"{bundle['training_rows']:,}"
        )

        st.metric(
            "Testing Records",
            f"{bundle['testing_rows']:,}"
        )

# ============================================================
# 7. MARKET EXPLORER
# ============================================================

elif page == "Market Explorer":

    st.markdown(
        '<div class="main-title">Market Explorer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore NVIDIA historical price movements and trading volume.'
        '</div>',
        unsafe_allow_html=True
    )

    min_date = history["Date"].min().date()

    max_date = history["Date"].max().date()

    selected_dates = st.date_input(
        "Select Historical Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:

        start_date, end_date = selected_dates

        filtered = history[
            (history["Date"].dt.date >= start_date) &
            (history["Date"].dt.date <= end_date)
        ].copy()

    else:

        filtered = history.copy()

    if filtered.empty:

        st.warning("No records available for this date range.")

    else:

        # CANDLESTICK CHART

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=filtered["Date"],
                open=filtered["Open"],
                high=filtered["High"],
                low=filtered["Low"],
                close=filtered["Close"],
                name="NVDA",
                increasing_line_color="#76B900",
                decreasing_line_color="#EF4444"
            )
        )

        fig.update_layout(
            title="NVDA Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Stock Price ($)",
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(
            chart_layout(fig, 500),
            use_container_width=True
        )

        # VOLUME CHART

        st.markdown("### Trading Volume")

        volume_fig = px.bar(
            filtered,
            x="Date",
            y="Volume",
            title="Historical Trading Volume"
        )

        volume_fig.update_traces(
            marker_color="#76B900"
        )

        st.plotly_chart(
            chart_layout(volume_fig, 350),
            use_container_width=True
        )

        # RETURNS

        st.markdown("### Daily Returns")

        returns = filtered.copy()

        returns["Daily Return (%)"] = (
            returns["Close"].pct_change() * 100
        )

        return_fig = px.line(
            returns,
            x="Date",
            y="Daily Return (%)",
            title="Daily Stock Returns"
        )

        return_fig.update_traces(
            line_color="#60A5FA"
        )

        st.plotly_chart(
            chart_layout(return_fig, 350),
            use_container_width=True
        )

# ============================================================
# 8. AI PRICE FORECAST
# ============================================================

elif page == "AI Price Forecast":

    st.markdown(
        '<div class="main-title">AI Price Forecast</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Predict NVIDIA next-trading-day closing price using regression.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"Latest historical record: {latest_date.strftime('%d %B %Y')} "
        f"| Closing price: {format_currency(latest_close)}"
    )

    model_name = st.selectbox(
        "Select Regression Algorithm",
        list(regression_models.keys())
    )

    selected_model = regression_models[model_name]

    if st.button(
        "GENERATE AI PRICE FORECAST",
        type="primary",
        use_container_width=True
    ):

        prediction = float(
            selected_model.predict(latest_features)[0]
        )

        difference = prediction - latest_close

        percentage = (
            difference / latest_close
        ) * 100

        st.divider()

        st.markdown("### Prediction Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Current Close",
            format_currency(latest_close)
        )

        c2.metric(
            "Predicted Next Close",
            format_currency(prediction)
        )

        c3.metric(
            "Expected Price Change",
            f"{percentage:+.2f}%",
            f"{difference:+.2f}"
        )

        if difference > 0:

            st.success(
                "The model predicts a higher closing price "
                "for the next trading session."
            )

        elif difference < 0:

            st.warning(
                "The model predicts a lower closing price "
                "for the next trading session."
            )

        else:

            st.info(
                "The model predicts almost no price change."
            )

        # PRICE COMPARISON

        forecast_chart = go.Figure()

        forecast_chart.add_trace(
            go.Bar(
                x=["Latest Historical Close", "Predicted Next Close"],
                y=[latest_close, prediction],
                marker_color=["#60A5FA", "#76B900"],
                text=[
                    format_currency(latest_close),
                    format_currency(prediction)
                ],
                textposition="outside"
            )
        )

        forecast_chart.update_layout(
            title=f"Price Forecast - {model_name}",
            yaxis_title="Price ($)",
            showlegend=False
        )

        st.plotly_chart(
            chart_layout(forecast_chart, 380),
            use_container_width=True
        )

        st.caption(
            "Forecast is based on the latest historical features. "
            "The next trading date is not guaranteed to be the next "
            "calendar day. This is an experimental estimate, not "
            "financial advice."
        )

    st.divider()

    st.markdown("### Regression Model Leaderboard")

    st.dataframe(
        regression_results.round(4),
        hide_index=True,
        use_container_width=True
    )

# ============================================================
# 9. MARKET DIRECTION CLASSIFICATION
# ============================================================

elif page == "Market Direction":

    st.markdown(
        '<div class="main-title">Market Direction Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-based classification of the next trading day direction.'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "UP = Next closing price is higher than the latest close. "
        "DOWN = Next closing price is equal to or lower than the latest close."
    )

    model_name = st.selectbox(
        "Select Classification Algorithm",
        list(classification_models.keys())
    )

    selected_model = classification_models[model_name]

    if st.button(
        "ANALYZE MARKET DIRECTION",
        type="primary",
        use_container_width=True
    ):

        prediction = int(
            selected_model.predict(latest_features)[0]
        )

        probabilities = selected_model.predict_proba(
            latest_features
        )[0]

        classes = list(selected_model.classes_)

        up_probability = probabilities[
            classes.index(1)
        ] * 100

        down_probability = probabilities[
            classes.index(0)
        ] * 100

        st.divider()

        if prediction == 1:

            st.success("PREDICTION: UPWARD DIRECTION")

        else:

            st.error("PREDICTION: DOWNWARD DIRECTION")

        c1, c2 = st.columns(2)

        c1.metric(
            "UP Probability",
            f"{up_probability:.2f}%"
        )

        c2.metric(
            "DOWN Probability",
            f"{down_probability:.2f}%"
        )

        # PROBABILITY CHART

        probability_fig = go.Figure(
            go.Bar(
                x=["UP", "DOWN"],
                y=[up_probability, down_probability],
                marker_color=["#76B900", "#EF4444"],
                text=[
                    f"{up_probability:.2f}%",
                    f"{down_probability:.2f}%"
                ],
                textposition="outside"
            )
        )

        probability_fig.update_layout(
            title="Predicted Class Probabilities",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 110],
            showlegend=False
        )

        st.plotly_chart(
            chart_layout(probability_fig, 380),
            use_container_width=True
        )

        st.markdown("### Probability Distribution")

        st.progress(
            int(round(up_probability)),
            text=f"UP: {up_probability:.2f}%"
        )

        st.progress(
            int(round(down_probability)),
            text=f"DOWN: {down_probability:.2f}%"
        )

        st.caption(
            "Probabilities are model outputs, not calibrated guarantees "
            "of market outcomes."
        )

    st.divider()

    st.markdown("### Classification Model Leaderboard")

    st.dataframe(
        classification_results.round(4),
        hide_index=True,
        use_container_width=True
    )

# ============================================================
# 10. MODEL LABORATORY
# ============================================================

elif page == "Model Laboratory":

    st.markdown(
        '<div class="main-title">Model Laboratory</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Evaluate and compare all six machine learning algorithms.'
        '</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs([
        "Regression Analysis",
        "Classification Analysis",
        "Actual vs Predicted"
    ])

    # REGRESSION

    with tab1:

        st.markdown("### Regression Performance")

        st.dataframe(
            regression_results.round(4),
            hide_index=True,
            use_container_width=True
        )

        best_regression = regression_results.iloc[0]

        st.success(
            f"Lowest RMSE: {best_regression['Model']}"
        )

        fig = px.bar(
            regression_results,
            x="Model",
            y="RMSE",
            color="Model",
            title="Regression RMSE Comparison"
        )

        st.plotly_chart(
            chart_layout(fig, 400),
            use_container_width=True
        )

        fig2 = px.bar(
            regression_results,
            x="Model",
            y="R2 Score",
            color="Model",
            title="Regression R² Comparison"
        )

        st.plotly_chart(
            chart_layout(fig2, 400),
            use_container_width=True
        )

        st.caption(
            "Lower MAE and RMSE are better. Higher R² is generally better."
        )

    # CLASSIFICATION

    with tab2:

        st.markdown("### Classification Performance")

        st.dataframe(
            classification_results.round(4),
            hide_index=True,
            use_container_width=True
        )

        best_classification = classification_results.iloc[0]

        st.success(
            f"Highest F1 Score: {best_classification['Model']}"
        )

        fig = px.bar(
            classification_results,
            x="Model",
            y=["Accuracy", "Precision", "Recall", "F1 Score"],
            barmode="group",
            title="Classification Metrics Comparison"
        )

        st.plotly_chart(
            chart_layout(fig, 450),
            use_container_width=True
        )

        st.caption(
            "Higher scores are generally better. "
            "Compare these metrics with the class balance and "
            "a simple majority-class baseline."
        )

    # ACTUAL VS PREDICTED

    with tab3:

        st.markdown("### Actual vs Predicted Closing Price")

        selected_regression = st.selectbox(
            "Choose Model",
            list(regression_models.keys()),
            key="actual_prediction_model"
        )

        prediction_data = regression_predictions.copy()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=prediction_data["Date"],
                y=prediction_data["Actual"],
                name="Actual Price",
                line=dict(
                    color="#60A5FA",
                    width=2
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=prediction_data["Date"],
                y=prediction_data[selected_regression],
                name="Predicted Price",
                line=dict(
                    color="#76B900",
                    width=2,
                    dash="dash"
                )
            )
        )

        fig.update_layout(
            title=f"Actual vs Predicted - {selected_regression}",
            xaxis_title="Date",
            yaxis_title="Closing Price ($)",
            hovermode="x unified"
        )

        st.plotly_chart(
            chart_layout(fig, 450),
            use_container_width=True
        )

        st.caption(
            "These predictions come from the held-out chronological "
            "test set, not the latest future forecast."
        )

# ============================================================
# 11. DATA CENTER
# ============================================================

elif page == "Data Center":

    st.markdown(
        '<div class="main-title">Data Center</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore dataset quality, historical records and downloads.'
        '</div>',
        unsafe_allow_html=True
    )

    # DATA SUMMARY

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Records",
        f"{len(history):,}"
    )

    c2.metric(
        "Total Features",
        str(len(bundle["feature_columns"]))
    )

    c3.metric(
        "Missing Values",
        str(int(history.isna().sum().sum()))
    )

    st.divider()

    st.markdown("### Dataset Preview")

    st.dataframe(
        history.tail(100).sort_values(
            "Date", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Dataset Statistics")

    st.dataframe(
        history[
            ["Open", "High", "Low", "Close", "Volume"]
        ].describe().round(3),
        use_container_width=True
    )

    st.markdown("### Feature Information")

    feature_table = pd.DataFrame({
        "Feature": bundle["feature_columns"],
        "Description": [
            "Opening stock price",
            "Highest stock price",
            "Lowest stock price",
            "Closing stock price",
            "Trading volume",
            "Daily percentage return",
            "Daily price range relative to close",
            "Previous day's close",
            "Close from two days earlier",
            "Close from five days earlier",
            "Previous day's trading volume",
            "5-day moving average",
            "10-day moving average",
            "20-day moving average",
            "5-day return volatility",
            "10-day return volatility"
        ]
    })

    st.dataframe(
        feature_table,
        hide_index=True,
        use_container_width=True
    )

    # DOWNLOAD

    csv_data = history.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="DOWNLOAD CLEANED NVDA DATA",
        data=csv_data,
        file_name="NVDA_Cleaned_Data.csv",
        mime="text/csv",
        use_container_width=True
    )

    # MODEL DOWNLOAD

    with open("nvda_models.pkl", "rb") as file:

        st.download_button(
            label="DOWNLOAD TRAINED MODEL",
            data=file,
            file_name="nvda_models.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )

# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748B;
        font-size:12px;
        padding:15px;
    ">
        NVDA STOCK INTELLIGENCE |
        MACHINE LEARNING ANALYTICS |
        EDUCATIONAL PROJECT
        <br><br>
        Market predictions are experimental estimates,
        not financial advice.
    </div>
    """,
    unsafe_allow_html=True
)
