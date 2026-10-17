import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=300, show_spinner=False)
def history(symbol, period="5y", interval="1d"):
    df=yf.download(symbol,period=period,interval=interval,auto_adjust=True,progress=False,threads=False)
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    if df.empty: raise ValueError(f"No market data returned for {symbol}")
    return df.dropna(how="all")

@st.cache_data(ttl=900, show_spinner=False)
def snapshot(symbol):
    t=yf.Ticker(symbol)
    try: info=t.info or {}
    except Exception: info={}
    h=history(symbol,"1y")
    close=float(h.Close.iloc[-1])
    return {"name":info.get("longName",symbol),"price":info.get("currentPrice",close),
      "marketCap":info.get("marketCap"),"volume":info.get("volume",int(h.Volume.iloc[-1])),
      "fiftyTwoWeekHigh":info.get("fiftyTwoWeekHigh",float(h.High.max())),"fiftyTwoWeekLow":info.get("fiftyTwoWeekLow",float(h.Low.min())),
      "trailingPE":info.get("trailingPE"),"priceToBook":info.get("priceToBook"),"dividendYield":info.get("dividendYield"),
      "returnOnEquity":info.get("returnOnEquity"),"debtToEquity":info.get("debtToEquity"),"freeCashflow":info.get("freeCashflow")}
