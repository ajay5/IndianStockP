import os
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from dotenv import load_dotenv
load_dotenv()
from config.stocks import SECTORS,INDICES,HORIZONS
from services.market_data import history,snapshot
from services.indicators import add_indicators
from services.news import fetch_news
from ml.predictor import predict
from ml.price_forecaster import forecast_next_close
st.set_page_config(page_title="Bharat Market AI Lab",page_icon="📈",layout="wide")
st.markdown("""<style>.block-container{padding-top:1.2rem}.metric-card{border:1px solid #263246;border-radius:14px;padding:12px;background:#101827}</style>""",unsafe_allow_html=True)
st.title("📈 Bharat Market AI Lab")
st.warning("Educational research only. Model outputs are probabilistic, can be wrong, and are not investment advice. Do not trade solely from this app.")
with st.sidebar:
    st.header("Research filters")
    exchange=st.selectbox("Exchange",["NSE","BSE"])
    sector=st.selectbox("Sector",list(SECTORS))
    company=st.selectbox("Company",list(SECTORS[sector]))
    raw=SECTORS[sector][company]; suffix=".NS" if exchange=="NSE" else ".BO"; symbol=raw+suffix
    timeframe=st.selectbox("Prediction horizon",list(HORIZONS))
    period=st.selectbox("Chart history",["6mo","1y","2y","5y"],index=2)
    run=st.button("Run analysis",type="primary",use_container_width=True)

tabs=st.tabs(["Market Overview","Stock Prediction","Technical Analysis","News Intelligence","Sector Analysis","Portfolio","Alerts","Architecture"])
with tabs[0]:
    st.subheader("Indian market snapshot")
    cols=st.columns(5); moves=[]
    for col,(name,ticker) in zip(cols,INDICES.items()):
        try:
            d=history(ticker,"5d"); price=float(d.Close.iloc[-1]); change=(price/d.Close.iloc[-2]-1)*100; moves.append(change); col.metric(name,f"{price:,.2f}",f"{change:+.2f}%")
        except Exception: col.metric(name,"Unavailable")
    market_move_pct=float(np.mean(moves)) if moves else 0.0
    mood="🟢 Bullish" if moves and market_move_pct>.3 else "🔴 Bearish" if moves and market_move_pct<-.3 else "🟡 Neutral"
    st.info(f"Market mood: **{mood}** based on average latest index movement, not a forecast.")

try:
    df=add_indicators(history(symbol,period)); info=snapshot(symbol); news=fetch_news(company); news_score=float(np.mean([n['score'] for n in news])) if news else 0
except Exception as e:
    st.error(f"Data provider error: {e}"); st.stop()

with tabs[1]:
    st.subheader(f"{company} · {symbol}")
    m=st.columns(6)
    vals=[("Current price",info.get('price')),('Market cap',info.get('marketCap')),('Volume',info.get('volume')),('52W high',info.get('fiftyTwoWeekHigh')),('52W low',info.get('fiftyTwoWeekLow')),('P/E',info.get('trailingPE'))]
    for c,(k,v) in zip(m,vals): c.metric(k,"N/A" if v is None else f"{v:,.2f}")
    fig=go.Figure(data=[go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name="Price")]); fig.add_trace(go.Scatter(x=df.index,y=df.EMA20,name="EMA 20")); fig.add_trace(go.Scatter(x=df.index,y=df.EMA50,name="EMA 50")); fig.update_layout(height=480,xaxis_rangeslider_visible=False); st.plotly_chart(fig,use_container_width=True)
    if run:
        try:
            result=predict(df,HORIZONS[timeframe],news_score,50,50); a=st.columns(6)
            for c,(k,v) in zip(a,[('Research signal',result['label']),('Confidence',f"{result['confidence']}%"),('Bullish probability',f"{result['bullish']}%"),('Bearish probability',f"{result['bearish']}%"),('Expected return*',f"{result['expected_return']:+.2f}%"),('Risk',result['risk'])]): c.metric(k,v)
            st.caption(f"Walk-forward holdout balanced accuracy: {result['backtest_balanced_accuracy']}%. *Naive historical-drift estimate, not a price target.")
            st.markdown("**Why the model produced this signal**")
            for r in result['reasons']: st.write("• "+r)
            st.divider()
            st.subheader("Next trading-day price forecast")
            price_fc=forecast_next_close(df, news_score, market_move_pct)
            pcols=st.columns(5)
            pcols[0].metric("Latest completed close", f"₹{price_fc['current_close']:,.2f}")
            pcols[1].metric("Predicted next close", f"₹{price_fc['predicted_close']:,.2f}", f"{price_fc['predicted_change_pct']:+.2f}%")
            pcols[2].metric("80% model interval", f"₹{price_fc['lower']:,.2f} - ₹{price_fc['upper']:,.2f}")
            pcols[3].metric("Walk-forward MAE", f"{price_fc['cv_mae_pct']:.2f}%")
            pcols[4].metric("Training rows", f"{price_fc['trained_rows']:,}")
            st.caption(f"Automatically refitted from latest completed history at {price_fc['trained_at']} UTC. Components: technical {price_fc['technical_return_pct']:+.2f}%, news {price_fc['news_adjustment_pct']:+.2f}%, market {price_fc['market_adjustment_pct']:+.2f}%. This is an uncertain estimate, not a guaranteed target.")
        except Exception as e: st.warning(str(e))
    else: st.info("Choose filters and click **Run analysis**.")

with tabs[2]:
    last=df.iloc[-1]; cols=st.columns(8)
    for c,k in zip(cols,["RSI","MACD","EMA20","EMA50","EMA200","ATR","ADX","STOCH_RSI"]): c.metric(k,"N/A" if pd.isna(last[k]) else f"{last[k]:.2f}")
    ind=df[["Close","EMA20","EMA50","EMA200","BB_UPPER","BB_LOWER"]].dropna(); st.line_chart(ind)
with tabs[3]:
    st.metric("Aggregate news sentiment",f"{(news_score+1)*50:.1f}/100")
    if not news: st.info("No recent headlines were returned.")
    for n in news:
        color="🟢" if n['label']=='Positive' else "🔴" if n['label']=='Negative' else "🟡"
        st.markdown(f"{color} [{n['title']}]({n['url']})  \n{n['source']} · {n['published']} · score {n['score']:+.2f}")
    st.caption("Headline sentiment uses VADER as a lightweight baseline. Validate against FinBERT before production use. Social APIs require separate licensed credentials and are not scraped by this project.")
with tabs[4]:
    rows=[]
    for cname,t in list(SECTORS[sector].items())[:8]:
        try:
            d=history(t+suffix,"1mo"); ret=(d.Close.iloc[-1]/d.Close.iloc[0]-1)*100; rows.append({"Company":cname,"1M return %":round(ret,2)})
        except Exception: pass
    if rows:
        sd=pd.DataFrame(rows).sort_values("1M return %",ascending=False); st.dataframe(sd,use_container_width=True,hide_index=True); st.bar_chart(sd.set_index("Company"))
with tabs[5]:
    st.info("MVP portfolio editor. Persistence is defined in database/schema.sql.")
    st.data_editor(pd.DataFrame([{"Symbol":symbol,"Quantity":0.0,"Average Price":0.0}]),num_rows="dynamic",use_container_width=True)
with tabs[6]:
    st.info("Alert rules are stored in the alerts table. Connect a scheduled worker plus Telegram Bot API/email provider before enabling notifications.")
    st.selectbox("Rule",["Price above target","Price below target","RSI below 30","RSI above 70","MACD crossover","News sentiment changed"]); st.text_input("Threshold / target"); st.button("Save alert",disabled=True)
with tabs[7]:
    st.code("Browser → Streamlit UI → cached services → Yahoo market data / licensed news API → feature engineering → time-aware ensemble → PostgreSQL; Redis cache + scheduled workers for alerts",language="text")
    st.markdown("See `README.md`, `database/schema.sql`, and `docs/architecture.md` in the downloaded project.")
