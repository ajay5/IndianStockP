import numpy as np
import pandas as pd

def add_indicators(df):
    x=df.copy(); c=x.Close; h=x.High; l=x.Low; v=x.Volume
    delta=c.diff(); up=delta.clip(lower=0); dn=-delta.clip(upper=0)
    rs=up.ewm(alpha=1/14,adjust=False).mean()/dn.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan)
    x["RSI"]=100-(100/(1+rs)); x["EMA20"]=c.ewm(span=20).mean(); x["EMA50"]=c.ewm(span=50).mean(); x["EMA200"]=c.ewm(span=200).mean()
    x["SMA20"]=c.rolling(20).mean(); sd=c.rolling(20).std(); x["BB_UPPER"]=x.SMA20+2*sd; x["BB_LOWER"]=x.SMA20-2*sd
    ema12=c.ewm(span=12).mean(); ema26=c.ewm(span=26).mean(); x["MACD"]=ema12-ema26; x["MACD_SIGNAL"]=x.MACD.ewm(span=9).mean()
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); x["ATR"]=tr.rolling(14).mean()
    x["VWAP"]=(c*v).cumsum()/v.replace(0,np.nan).cumsum(); low14=l.rolling(14).min(); high14=h.rolling(14).max(); x["STOCH_RSI"]=(c-low14)/(high14-low14)*100
    plus=h.diff(); minus=-l.diff(); plus_dm=plus.where((plus>minus)&(plus>0),0); minus_dm=minus.where((minus>plus)&(minus>0),0)
    atr=x.ATR.replace(0,np.nan); pdi=100*plus_dm.rolling(14).mean()/atr; mdi=100*minus_dm.rolling(14).mean()/atr; x["ADX"]=(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)).rolling(14).mean()
    x["RET1"]=c.pct_change(); x["VOL20"]=x.RET1.rolling(20).std()*np.sqrt(252); x["VOL_Z"]=(v-v.rolling(20).mean())/v.rolling(20).std()
    return x
