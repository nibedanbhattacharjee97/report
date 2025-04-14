import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime


st.set_page_config(page_title="Indian Stock AI", layout="wide")
st.title("📊 Indian Stock Market AI Analysis (Last 5 Years)")
st.markdown("Use this tool to analyze and predict trends for all major Indian stocks using historical data and AI.")


@st.cache_data
def load_stock_list():
    return pd.read_excel("nse_stocks.xlsx")  


stocks = load_stock_list()
selected = st.multiselect(
    "Select companies to analyze",
    options=stocks['Company Name'].tolist(),
    default=["Reliance", "TCS"]
)


def fetch_stock_data(ticker):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=5*365)
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data


for company in selected:
    ticker = stocks.loc[stocks['Company Name'] == company, 'Ticker'].values[0]
    st.subheader(f"📈 {company} ({ticker})")

    data = fetch_stock_data(ticker)


    st.markdown("**Close Price Trend**")
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Close', color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{company} - Close Price")
    ax.legend()
    st.pyplot(fig)

    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()

    st.markdown("**Simple Moving Averages**")
    fig2, ax2 = plt.subplots()
    ax2.plot(data['Close'], label='Close', alpha=0.5)
    ax2.plot(data['SMA20'], label='SMA 20', color='orange')
    ax2.plot(data['SMA50'], label='SMA 50', color='green')
    ax2.set_title(f"{company} - SMA")
    ax2.legend()
    st.pyplot(fig2)


    st.markdown("**AI-Based Trend Prediction**")

    df = data.copy()
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)

    if len(df) > 100:  
        X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.write(f"📊 Model Accuracy: **{acc:.2f}**")

        signal = y_pred[-1]
        if signal == 1:
            st.success("📈 AI Prediction: Expecting **UP** trend tomorrow!")
        else:
            st.error("📉 AI Prediction: Expecting **DOWN** trend tomorrow.")
    else:
        st.warning("Not enough data to run prediction.")
    
    st.markdown("---")
