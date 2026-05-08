import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np
from datetime import datetime

# ---------- CONFIG ----------
st.set_page_config(page_title="Mutual Fund Dashboard", layout="wide")

# ---------- LOAD ----------
model = pickle.load(open("model.pkl", "rb"))
df = pd.read_csv("mutual_fund_data.csv")

# ---------- FEATURE ENGINEERING ----------
df['Launch_Date'] = pd.to_datetime(df['Launch_Date'], format='%d-%m-%y', errors='coerce')
df['fund_age'] = datetime.now().year - df['Launch_Date'].dt.year

df = df[(df['Average_AUM_Cr'] > 0) & (df['NAV'] > 0)]
df = df.dropna(subset=['fund_age'])

# ---------- SIDEBAR ----------
st.sidebar.title("📊 Mutual Funds")

watchlist = df.sort_values(by="NAV", ascending=False).head(10)

for _, row in watchlist.iterrows():
    nav = round(row['NAV'], 2)
    change = round(np.random.uniform(-2, 2), 2)

    color = "green" if change >= 0 else "red"
    arrow = "▲" if change >= 0 else "▼"

    st.sidebar.markdown(f"**{row['Scheme_Name'][:35]}...**")
    st.sidebar.write(f"NAV: ₹ {nav}")
    st.sidebar.markdown(
        f"<span style='color:{color}; font-weight:bold;'>{arrow} {abs(change)}%</span>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

# ---------- HEADER ----------
st.title("💼 Mutual Fund Dashboard")

# ---------- SEARCH ----------
search = st.text_input("🔍 Search Mutual Funds")
if search:
    df = df[df['Scheme_Name'].str.contains(search, case=False, na=False)]

# ---------- KPIs ----------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total AUM", f"₹ {round(df['Average_AUM_Cr'].sum(),2)}")
col2.metric("Total Funds", df.shape[0])
col3.metric("Avg NAV", f"₹ {round(df['NAV'].mean(),2)}")
col4.metric("Max NAV", f"₹ {round(df['NAV'].max(),2)}")

st.markdown("---")

# ---------- FUND SELECT ----------
fund_list = df['Scheme_Name'].unique()
selected_fund = st.selectbox("Select Fund", fund_list)

fund_data = df[df['Scheme_Name'] == selected_fund].iloc[0]

aum = float(fund_data['Average_AUM_Cr'])
age = int(fund_data['fund_age'])
actual_nav = float(fund_data['NAV'])

# ---------- FUND DETAILS ----------
st.subheader(selected_fund)

col1, col2, col3 = st.columns(3)
col1.metric("AUM", f"₹ {round(aum,2)} Cr")
col2.metric("Age", f"{age} Years")
col3.metric("NAV", f"₹ {round(actual_nav,2)}")

# ---------- FUTURE INPUT ----------
future_years = st.slider("📅 Predict after how many years?", 1, 10, 1)
future_age = age + future_years

investment = st.number_input("💰 Investment Amount (₹)", min_value=100, value=100)

# ---------- SESSION ----------
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False

if st.button("🚀 Predict NAV"):
    st.session_state.predict_clicked = True

st.markdown("---")

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["Dashboard", "Analytics", "Prediction"])

# ================= DASHBOARD =================
with tab1:
    st.subheader("🏆 Top Performing Funds")

    top = df.sort_values(by="NAV", ascending=False).head(3)
    cols = st.columns(3)

    for i, (_, row) in enumerate(top.iterrows()):
        cols[i].write(row['Scheme_Name'])
        cols[i].metric("NAV", f"₹ {round(row['NAV'],2)}")

# ================= ANALYTICS (FIXED) =================
with tab2:
    st.subheader("📊 Analytics (Cleaned View)")

    # Remove extreme outliers
    df_clean = df[df['NAV'] < df['NAV'].quantile(0.99)]

    col1, col2 = st.columns(2)

    # Scatter with log scale
    fig1 = px.scatter(
        df_clean,
        x="Average_AUM_Cr",
        y="NAV",
        log_y=True,
        title="AUM vs NAV (Log Scale)"
    )

    col1.plotly_chart(fig1, use_container_width=True)

    # Histogram
    fig2 = px.histogram(
        df_clean,
        x="NAV",
        nbins=50,
        title="NAV Distribution"
    )

    col2.plotly_chart(fig2, use_container_width=True)

# ================= PREDICTION =================
with tab3:
    st.subheader("📈 Investment Prediction")

    if st.session_state.predict_clicked:

        predicted_nav = model.predict([[aum, future_age]])[0]

        units = investment / actual_nav
        future_value = units * predicted_nav

        profit = future_value - investment
        return_pct = (profit / investment) * 100

        st.success(f"After {future_years} years, ₹{investment} → ₹{round(future_value,2)}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Invested", f"₹ {investment}")
        c2.metric("Future Value", f"₹ {round(future_value,2)}")
        c3.metric("Profit", f"₹ {round(profit,2)}", f"{round(return_pct,2)}%")

        # Growth line chart
        years = list(range(0, future_years + 1))
        values = []

        for y in years:
            temp_age = age + y
            temp_nav = model.predict([[aum, temp_age]])[0]
            temp_value = (investment / actual_nav) * temp_nav
            values.append(temp_value)

        growth_df = pd.DataFrame({
            "Year": years,
            "Value": values
        })

        fig = px.line(growth_df, x="Year", y="Value", markers=True,
                      title="Investment Growth Over Time")

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Click Predict NAV to see result")

# ---------- DOWNLOAD ----------
st.markdown("---")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Report", csv, "report.csv")

# ---------- FOOTER ----------
st.markdown("<p style='text-align:center;'>Built by Shweta | ML + Streamlit Project</p>", unsafe_allow_html=True)