import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
FEATURES=["RSI","MACD","MACD_SIGNAL","EMA20","EMA50","EMA200","ATR","ADX","STOCH_RSI","VOL20","VOL_Z","RET1"]

def predict(df,horizon,news_score=0,sector_score=0,market_score=0):
    x=df.copy(); x["target"]=(x.Close.shift(-horizon)/x.Close-1)
    threshold=max(.007, float(x.RET1.std() or .01)*np.sqrt(horizon)*.35)
    x["y"]=np.select([x.target>threshold,x.target<-threshold],[2,0],default=1)
    data=x[FEATURES+["y"]].dropna(); data=data.iloc[:-horizon] if len(data)>horizon else data
    if len(data)<140 or data.y.nunique()<2: raise ValueError("Insufficient history for a reliable model.")
    cut=int(len(data)*.8); tr,te=data.iloc[:cut],data.iloc[cut:]
    rf=RandomForestClassifier(n_estimators=250,max_depth=7,min_samples_leaf=5,class_weight="balanced",random_state=42,n_jobs=-1)
    hg=make_pipeline(StandardScaler(),HistGradientBoostingClassifier(max_iter=180,max_depth=5,learning_rate=.05,random_state=42))
    models=[rf,hg]; probs=[]; scores=[]
    for m in models:
        m.fit(tr[FEATURES],tr.y); p=m.predict_proba(te[FEATURES]); scores.append(balanced_accuracy_score(te.y,m.predict(te[FEATURES])))
        latest=x[FEATURES].dropna().iloc[[-1]]; q=m.predict_proba(latest)[0]; full=np.zeros(3)
        for cls,val in zip(m.classes_,q): full[int(cls)]=val
        probs.append(full)
    base=np.average(probs,axis=0,weights=np.maximum(scores,.01)); technical=(base[2]-base[0]+1)/2*100
    final=.35*technical+.20*((news_score+1)*50)+.15*sector_score+.10*market_score+.20*50
    label="Strong Buy" if final>=75 else "Buy" if final>=60 else "Hold" if final>=40 else "Sell" if final>=25 else "Strong Sell"
    risk=min(100,max(0,float(x.VOL20.dropna().iloc[-1])*100)); exp=float(x.RET1.mean()*horizon*100)
    reasons=[]; last=x.iloc[-1]
    reasons += ["RSI indicates oversold conditions"] if last.RSI<35 else ["RSI indicates overbought conditions"] if last.RSI>70 else ["RSI is neutral"]
    reasons += ["MACD is above its signal line"] if last.MACD>last.MACD_SIGNAL else ["MACD is below its signal line"]
    reasons += ["Latest news tone is positive"] if news_score>.1 else ["Latest news tone is negative"] if news_score<-.1 else ["Latest news tone is neutral"]
    return {"label":label,"confidence":round(float(base.max()*100),1),"bullish":round(float(base[2]*100),1),"bearish":round(float(base[0]*100),1),"expected_return":round(exp,2),"risk":"High" if risk>35 else "Medium" if risk>20 else "Low","backtest_balanced_accuracy":round(float(np.mean(scores)*100),1),"reasons":reasons,"score":round(final,1)}
