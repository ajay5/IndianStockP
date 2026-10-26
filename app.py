import os,json,urllib.parse
from datetime import datetime,timedelta,timezone
import numpy as np,pandas as pd,streamlit as st,plotly.graph_objects as go,yfinance as yf,feedparser
import requests
from dotenv import load_dotenv
from database import connect_database,database_backend,initialize_database
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from market_data import fetch_alphavantage_daily,fetch_bluesky_posts,fetch_finnhub_general_news,fetch_finnhub_news,fetch_gdelt_news,fetch_marketaux_news,fetch_nifty50_constituents,fetch_nse_announcements_rss,fetch_nse_commodity_etf_catalog,fetch_nse_company_announcements,fetch_nse_equity_catalog,fetch_reddit_posts,normalize_yahoo_news,point_in_time_news_feature
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,HistGradientBoostingRegressor
from validation import build_walk_forward_outcomes,evaluate_walk_forward,expanding_time_splits,recent_holdout_split

load_dotenv()
st.set_page_config(page_title='Bharat Market AI Production',page_icon='📈',layout='wide')
def config_value(name,default=''):
 value=os.getenv(name)
 if value not in (None,''):return value
 try:
  value=st.secrets.get(name,default)
 except Exception:
  return default
 return value if value not in (None,'') else default
DB=config_value('APP_DB_PATH','stock_ai.db'); DATABASE_URL=config_value('DATABASE_URL',''); AN=SentimentIntensityAnalyzer()
MARKET_NEWS_SYMBOL='__INDIA_MARKET__'
H={'1 Day':1,'3 Days':3,'7 Days':7,'15 Days':15,'30 Days':30,'3 Months':63,'1 Year':252}

def conn():
 return connect_database(DATABASE_URL,DB)
def init():
 initialize_database(DATABASE_URL,DB)
init()
@st.cache_data(ttl=6*3600,show_spinner=False)
def nse_equity_sector_catalog_cached():
 return fetch_nse_equity_catalog()
@st.cache_data(ttl=6*3600,show_spinner=False)
def nse_commodity_etf_catalog_cached():
 return fetch_nse_commodity_etf_catalog()
@st.cache_data(ttl=300,show_spinner=False)
def hist(ticker,period='5y'):
 d=yf.download(ticker,period=period,auto_adjust=True,progress=False,threads=False)
 if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
 if d.empty:raise ValueError('No data for '+ticker)
 return d.dropna()
def optional_hist(ticker,period='5y'):
 try:return hist(ticker,period)
 except Exception:return None
def sector_proxy(industry):
 value=str(industry or '').casefold()
 proxies=[(('bank','financial'),'^NSEBANK'),(('information technology','it - software','software'),'^CNXIT'),(('pharma','pharmaceutical'),'^CNXPHARMA'),(('automobile','auto'),'^CNXAUTO'),(('fmcg','consumer staples'),'^CNXFMCG'),(('metal','metals'),'^CNXMETAL'),(('real estate','realty'),'^CNXREALTY'),(('oil, gas','energy'),'^CNXENERGY')]
 return next((ticker for keys,ticker in proxies if any(key in value for key in keys)),'^NSEI')
def weighted_sentiment(items):
 return float(np.average([item['score'] for item in items],weights=[item.get('relevance',1) for item in items])) if items else 0.0
@st.cache_data(ttl=6*3600,show_spinner=False)
def alphavantage_daily_cached(api_key,symbol):
 return fetch_alphavantage_daily(api_key,symbol)
def indicators(d):
 x=d.copy();c=x.Close;v=x.Volume;delta=c.diff();up=delta.clip(lower=0);dn=-delta.clip(upper=0);rs=up.ewm(alpha=1/14,adjust=False).mean()/dn.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan)
 x['RSI']=100-100/(1+rs);x['EMA20']=c.ewm(span=20).mean();x['EMA50']=c.ewm(span=50).mean();x['EMA200']=c.ewm(span=200).mean();x['MACD']=c.ewm(span=12).mean()-c.ewm(span=26).mean();x['MACDS']=x.MACD.ewm(span=9).mean();x['RET']=c.pct_change();x['GAP']=x.Open/c.shift()-1;x['INTRADAY']=c/x.Open-1;x['RANGE']=(x.High-x.Low)/c;x['VOL']=x.RET.rolling(20).std()*np.sqrt(252);x['VOLZ']=(v-v.rolling(20).mean())/v.rolling(20).std();x['ATR']=pd.concat([x.High-x.Low,(x.High-c.shift()).abs(),(x.Low-c.shift()).abs()],axis=1).max(axis=1).rolling(14).mean();return x
@st.cache_data(ttl=3300,show_spinner=False)
def reddit_access_token(client_id,client_secret,user_agent):
 if not client_id or not client_secret:return ''
 try:
  response=requests.post('https://www.reddit.com/api/v1/access_token',auth=(client_id,client_secret),data={'grant_type':'client_credentials'},headers={'User-Agent':user_agent},timeout=(3,8));response.raise_for_status();return str(response.json().get('access_token') or '')
 except (requests.RequestException,ValueError):return ''
@st.cache_data(ttl=300,show_spinner=False)
def news(company,symbol,alpha_key,marketaux_token,finnhub_key,reddit_id,reddit_secret,reddit_agent,is_commodity=False):
 out=[];provider_status=[]
 def collect_api(name,key,environment_name,fetch,market_wide=False):
  checked_at=datetime.now(timezone.utc).isoformat()
  if not key:
   provider_status.append({'Provider':name,'Configured':'No','Called':'No','Articles':0,'Status':'Not configured','Details':f'Set {environment_name} in .env','Checked UTC':checked_at})
   return
  try:
   items,detail=fetch();items=items or []
   if market_wide:
    for item in items:item['market_wide']=True
   out.extend(items)
   provider_status.append({'Provider':name,'Configured':'Yes','Called':'Yes','Articles':len(items),'Status':'OK' if items else ('No results' if detail.startswith('Request succeeded') else 'Error'),'Details':detail,'Checked UTC':checked_at})
  except Exception as exc:
   provider_status.append({'Provider':name,'Configured':'Yes','Called':'Yes','Articles':0,'Status':'Error','Details':f'Unexpected {type(exc).__name__}','Checked UTC':checked_at})
 try:
  out.extend(normalize_yahoo_news(yf.Ticker(symbol).get_news(count=30)))
 except Exception:
  pass
 provider_status.append({'Provider':'Alpha Vantage NEWS_SENTIMENT','Configured':'Yes' if alpha_key else 'No','Called':'No','Articles':0,'Status':'Not available on free plan' if alpha_key else 'Not configured','Details':'Alpha Vantage marks NEWS_SENTIMENT as premium; skipped to avoid a guaranteed premium-plan error.' if alpha_key else 'Set ALPHAVANTAGE_API_KEY in .env if you later enable a plan with NEWS_SENTIMENT access.','Checked UTC':datetime.now(timezone.utc).isoformat()})
 if not is_commodity:
  nse_end=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5,minutes=30))).date()
  nse_start=nse_end-timedelta(days=365)
  collect_api('NSE India official announcements RSS','public','NSE official RSS',lambda:fetch_nse_announcements_rss(symbol,company,30))
  collect_api('NSE India company announcement archive','public','NSE official announcements',lambda:fetch_nse_company_announcements(symbol,nse_start,nse_end,limit=500))
 collect_api('Marketaux',marketaux_token,'MARKETAUX_API_TOKEN',lambda:fetch_marketaux_news(marketaux_token,symbol,20))
 collect_api('Finnhub company news (free tier)',finnhub_key,'FINNHUB_API_KEY',lambda:fetch_finnhub_news(finnhub_key,symbol,20))
 collect_api('Finnhub general market news (free)',finnhub_key,'FINNHUB_API_KEY',lambda:fetch_finnhub_general_news(finnhub_key,20),market_wide=True)
 query=f'{company} commodity India MCX price' if is_commodity else f'{company} stock India'
 url='https://news.google.com/rss/search?q='+urllib.parse.quote(query)+'&hl=en-IN&gl=IN&ceid=IN:en'
 try:
  response=requests.get(url,timeout=(3,8));response.raise_for_status();feed=feedparser.parse(response.content)
  for e in feed.entries[:20]:
   published=e.get('published','')
   if e.get('published_parsed'):
    published=datetime(*e.published_parsed[:6],tzinfo=timezone.utc).isoformat()
   out.append({'title':e.get('title',''),'url':e.get('link',''),'source':e.get('source',{}).get('title','Google News'),'published':published})
 except Exception:
  pass
 if not is_commodity:
  market_query='India stock market Nifty Sensex RBI FII DII banking'
  market_url='https://news.google.com/rss/search?q='+urllib.parse.quote(market_query)+'&hl=en-IN&gl=IN&ceid=IN:en'
  try:
   response=requests.get(market_url,timeout=(3,8));response.raise_for_status();feed=feedparser.parse(response.content)
   for e in feed.entries[:20]:
    published=e.get('published','')
    if e.get('published_parsed'):published=datetime(*e.published_parsed[:6],tzinfo=timezone.utc).isoformat()
    out.append({'title':e.get('title',''),'url':e.get('link',''),'source':'Google News India Market','published':published,'relevance':0.25,'market_wide':True})
  except Exception:
   pass
 bare_symbol=symbol.rsplit('.',1)[0]
 try:out.extend(fetch_gdelt_news(f'("{company}" OR "{bare_symbol}") AND (India OR MCX OR commodity)',limit=20) if is_commodity else fetch_gdelt_news(f'("{company}" OR "{bare_symbol}")',limit=20))
 except Exception:pass
 if not is_commodity:
  try:
   market_items=fetch_gdelt_news('(India OR NSE OR Nifty OR Sensex) AND (RBI OR markets OR stocks OR banking)',limit=20)
   for item in market_items:item['market_wide']=True;item['relevance']=min(float(item.get('relevance',0.25)),0.25)
   out.extend(market_items)
  except Exception:pass
 try:out.extend(fetch_bluesky_posts(f'{company} commodity India MCX' if is_commodity else f'{company} stock',limit=20))
 except Exception:pass
 if reddit_id and reddit_secret:
  token=reddit_access_token(reddit_id,reddit_secret,reddit_agent)
  try:out.extend(fetch_reddit_posts(f'"{company}" commodity OR "{bare_symbol}"',token,reddit_agent,limit=20) if is_commodity else fetch_reddit_posts(f'"{company}" OR "{bare_symbol}"',token,reddit_agent,limit=20))
  except Exception:pass
 articles=[];seen=set();now=datetime.now(timezone.utc).isoformat();c=conn()
 for item in out:
  title=item.get('title','').strip();key=title.casefold()
  if not title or key in seen:continue
  seen.add(key)
  try:score=float(item['sentiment']) if item.get('sentiment') is not None else AN.polarity_scores(item.get('sentiment_text',title))['compound']
  except (TypeError,ValueError):score=AN.polarity_scores(item.get('sentiment_text',title))['compound']
  item['score']=float(np.clip(score,-1,1));articles.append(item)
  history_symbol=MARKET_NEWS_SYMBOL if item.get('market_wide') else symbol
  c.execute('INSERT OR IGNORE INTO news_history(symbol,captured_at,published_at,source,headline,url,sentiment,relevance) VALUES(?,?,?,?,?,?,?,?)',(history_symbol,now,item.get('published',''),item.get('source','Financial News'),title,item.get('url',''),item['score'],float(item.get('relevance',1))))
 c.commit();c.close();return articles,provider_status
def reconcile(symbol,d):
 closes={i.date().isoformat():float(v) for i,v in d.Close.items()};dates=sorted(closes);c=conn();rows=c.execute("SELECT * FROM predictions WHERE symbol=? AND actual_close IS NULL AND model_version NOT LIKE ?",(symbol,'%-walkforward-v2')).fetchall()
 for r in rows:
  future=[z for z in dates if z>r['prediction_date']]
  if len(future)<r['horizon']:continue
  td=future[r['horizon']-1];actual=closes[td];ar=actual/r['latest_close']-1;pdirection=np.sign(r['predicted_return']);adirection=np.sign(ar)
  c.execute('UPDATE predictions SET target_date=?,actual_close=?,actual_return=?,percentage_error=?,direction_correct=?,evaluated_at=? WHERE id=?',(td,actual,ar,abs(actual-r['predicted_close'])/actual*100,int(pdirection==adirection),datetime.now(timezone.utc).isoformat(),r['id']))
 c.commit();c.close()
def save_walk_forward_predictions(symbol,horizon,rows,run_tag='walkforward-v2'):
 if not rows:return 0
 version=f'{symbol}-h{horizon}-{run_tag}'
 captured=datetime.now(timezone.utc).isoformat();c=conn()
 values=[(symbol,row['prediction_date'],row['target_date'],horizon,row['latest_close'],row['predicted_close'],row['predicted_return'],row['signal'],row['confidence'],row['bullish'],row['bearish'],row['actual_close'],row['actual_return'],row['percentage_error'],row['direction_correct'],captured,version) for row in rows]
 c.executemany('INSERT OR REPLACE INTO predictions(symbol,prediction_date,target_date,horizon,latest_close,predicted_close,predicted_return,signal,confidence,bullish,bearish,actual_close,actual_return,percentage_error,direction_correct,evaluated_at,model_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',values)
 c.commit();c.close();return len(values)
def model(stock,market,sector,symbol,horizon,cost,return_only=False,market_context=None,include_market_sentiment=True,recent_holdout=0):
 x=stock.copy();m=market.Close.reindex(x.index).ffill();s=sector.Close.reindex(x.index).ffill();x['MKT']=m.pct_change();x['SEC']=s.pct_change();x['RELM']=x.RET-x.MKT;x['RELS']=x.RET-x.SEC
 x['RET5']=x.Close.pct_change(5);x['RET20']=x.Close.pct_change(20);x['MKT5']=m.pct_change(5);x['MKT20']=m.pct_change(20);x['MKT_VOL20']=x.MKT.rolling(20).std()*np.sqrt(252);x['SEC5']=s.pct_change(5);x['SEC20']=s.pct_change(20);x['SEC_VOL20']=x.SEC.rolling(20).std()*np.sqrt(252)
 market_context=market_context or {};bank_data=market_context.get('bank');vix_data=market_context.get('vix')
 bank_close=(bank_data.Close.reindex(x.index).ffill() if bank_data is not None else m).ffill();x['BANK']=bank_close.pct_change();x['BANK5']=bank_close.pct_change(5);x['BANK20']=bank_close.pct_change(20);x['BANK_VOL20']=x.BANK.rolling(20).std()*np.sqrt(252)
 if vix_data is not None:
  vix_close=vix_data.Close.reindex(x.index).ffill();x['VIX_LEVEL']=vix_close/100;x['VIX_CHANGE5']=vix_close.pct_change(5)
 else:
  x['VIX_LEVEL']=x.MKT_VOL20;x['VIX_CHANGE5']=x.MKT_VOL20.pct_change(5)
 c=conn();rows=[dict(r) for r in c.execute('SELECT published_at,sentiment,relevance,source FROM news_history WHERE symbol=? AND published_at IS NOT NULL',(symbol,)).fetchall()];market_rows=[dict(r) for r in c.execute('SELECT published_at,sentiment,relevance,source FROM news_history WHERE symbol=? AND published_at IS NOT NULL',(MARKET_NEWS_SYMBOL,)).fetchall()] if include_market_sentiment else [];c.close()
 announcement_rows=[];social_rows=[];news_rows=[]
 for row in rows:
  source=str(row.get('source','')).casefold()
  if source.startswith('nse india'):
   announcement_rows.append(row)
  elif source.startswith(('reddit','bluesky')):
   social_rows.append(row)
  else:
   news_rows.append(row)
 x['ANNOUNCEMENTS_365D']=point_in_time_news_feature(x.index,announcement_rows,lookback_days=365).to_numpy()
 x['NEWS_SENTIMENT_90D']=point_in_time_news_feature(x.index,news_rows,lookback_days=90).to_numpy()
 x['SOCIAL_SENTIMENT_30D']=point_in_time_news_feature(x.index,social_rows,lookback_days=30).to_numpy()
 x['MARKET_SENTIMENT_30D']=point_in_time_news_feature(x.index,market_rows,lookback_days=30).to_numpy()
 x['Y']=x.Close.shift(-horizon)/x.Close-1;band=np.maximum(.004,x.RET.rolling(60).std()*np.sqrt(horizon)*.35);x['C']=np.select([x.Y>band,x.Y<-band],[2,0],default=1)
 F=['RSI','EMA20','EMA50','EMA200','MACD','MACDS','RET','GAP','INTRADAY','RANGE','RET5','RET20','VOL','VOLZ','ATR','MKT','MKT5','MKT20','MKT_VOL20','SEC','SEC5','SEC20','SEC_VOL20','RELM','RELS','BANK','BANK5','BANK20','BANK_VOL20','RELBANK','VIX_LEVEL','VIX_CHANGE5','ANNOUNCEMENTS_365D','NEWS_SENTIMENT_90D','SOCIAL_SENTIMENT_30D','MARKET_SENTIMENT_30D'];x['RELBANK']=x.RET-x.BANK;z=x[F+['Y','C']].replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)<260:raise ValueError('Choose 5y or 10y history; insufficient valid rows.')
 X,yr,yc=z[F],z.Y,z.C.astype(int)
 if recent_holdout:
  train_positions,test_positions=recent_holdout_split(len(z),horizon,recent_holdout)
  train_x,train_y,train_c=X.iloc[train_positions],yr.iloc[train_positions],yc.iloc[train_positions]
  test_x,test_y,test_c=X.iloc[test_positions],yr.iloc[test_positions],yc.iloc[test_positions]
  r1=RandomForestRegressor(n_estimators=250,max_depth=8,min_samples_leaf=5,n_jobs=-1,random_state=42).fit(train_x,train_y)
  r2=HistGradientBoostingRegressor(max_iter=180,max_depth=5,random_state=42).fit(train_x,train_y)
  cl=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=5,class_weight='balanced',n_jobs=-1,random_state=42).fit(train_x,train_c)
  predicted=(r1.predict(test_x)+r2.predict(test_x))/2
  class_probs=np.zeros((len(test_x),3));raw_probs=cl.predict_proba(test_x)
  for column,label in enumerate(cl.classes_):class_probs[:,int(label)]=raw_probs[:,column]
  predicted_classes=class_probs.argmax(axis=1);actual=test_y.to_numpy();actual_classes=test_c.to_numpy()
  positions=stock.index.get_indexer(test_x.index)
  metrics,_,_=evaluate_walk_forward(actual,predicted,actual_classes,class_probs,positions,horizon,cost)
  outcomes=build_walk_forward_outcomes(stock.index,stock.Close.to_numpy(),test_x.index,actual,predicted,predicted_classes,class_probs,horizon)
  return {'metrics':metrics,'walk_forward':outcomes,'version':f'{symbol}-h{horizon}-recent-{recent_holdout}-session-walkforward-v2'}
 if return_only:
  r1=RandomForestRegressor(n_estimators=400,max_depth=8,min_samples_leaf=5,n_jobs=-1,random_state=42).fit(X,yr)
  r2=HistGradientBoostingRegressor(max_iter=220,max_depth=5,random_state=42).fit(X,yr)
  latest=x[F].dropna().iloc[[-1]].copy();pr=float((r1.predict(latest)[0]+r2.predict(latest)[0])/2);close=float(stock.Close.iloc[-1])
  return {'return':pr,'price':close*(1+pr),'close':close}
 folds=expanding_time_splits(len(X),horizon);op=np.full(len(z),np.nan);oq=np.full((len(z),3),np.nan)
 for tr,te in folds:
  r1=RandomForestRegressor(n_estimators=250,max_depth=8,min_samples_leaf=5,n_jobs=-1,random_state=42).fit(X.iloc[tr],yr.iloc[tr]);r2=HistGradientBoostingRegressor(max_iter=180,max_depth=5,random_state=42).fit(X.iloc[tr],yr.iloc[tr]);cl=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=5,class_weight='balanced',n_jobs=-1,random_state=42).fit(X.iloc[tr],yc.iloc[tr]);op[te]=(r1.predict(X.iloc[te])+r2.predict(X.iloc[te]))/2;q=cl.predict_proba(X.iloc[te]);
  for j,k in enumerate(cl.classes_):oq[te,int(k)]=q[:,j]
 valid=~np.isnan(op);valid_positions=np.flatnonzero(valid);act=yr.to_numpy()[valid];pred=op[valid];classes=yc.to_numpy()[valid];probs=oq[valid]
 metrics,pc,trade_offsets=evaluate_walk_forward(act,pred,classes,probs,valid_positions,horizon,cost)
 wf_rows=build_walk_forward_outcomes(stock.index,stock.Close.to_numpy(),z.index[valid],act,pred,pc,probs,horizon)
 r1=RandomForestRegressor(n_estimators=400,max_depth=8,min_samples_leaf=5,n_jobs=-1,random_state=42).fit(X,yr);r2=HistGradientBoostingRegressor(max_iter=220,max_depth=5,random_state=42).fit(X,yr);cl=RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=5,class_weight='balanced',n_jobs=-1,random_state=42).fit(X,yc);latest=x[F].dropna().iloc[[-1]].copy();pr=float((r1.predict(latest)[0]+r2.predict(latest)[0])/2);raw=np.zeros((1,3));q=cl.predict_proba(latest)[0]
 for k,v in zip(cl.classes_,q):raw[0,int(k)]=v
 pp=raw[0];res=act-pred;lo,hi=np.quantile(res,[.1,.9]);close=float(stock.Close.iloc[-1]);version=f'{symbol}-h{horizon}-{stock.index[-1].date()}'
 return {'return':pr,'price':close*(1+pr),'low':close*(1+pr+lo),'high':close*(1+pr+hi),'signal':['SELL','HOLD','BUY'][pp.argmax()],'confidence':pp.max()*100,'bull':pp[2]*100,'bear':pp[0]*100,'metrics':metrics,'version':version,'features':latest.iloc[0].to_dict(),'walk_forward':wf_rows}

@st.cache_data(ttl=900,show_spinner=False)
def commodity_price_forecast_cached(symbol,benchmark,period,horizon,db_path):
 stock=indicators(hist(symbol,period));market=hist(benchmark,period);currency=hist('USDINR=X',period)
 return model(stock,market,currency,symbol,horizon,0,return_only=True,include_market_sentiment=False)

st.title('📈 Bharat Market AI Production Research Platform');st.error('No model can guarantee returns. Use this only for research and paper trading, not as investment advice.')
equity_catalog,equity_catalog_status=nse_equity_sector_catalog_cached()
commodity_catalog,commodity_catalog_status=nse_commodity_etf_catalog_cached()
equity_options={f"{item['nse_symbol']} — {item['name']}":item for item in equity_catalog}
commodity_options={f"{item['nse_symbol']} — {item['asset']}":item for item in commodity_catalog}
industry_options=sorted({item['industry'] for item in equity_catalog})
commodity_categories=sorted({item['category'] for item in commodity_catalog})
with st.sidebar:
 if st.button('Refresh online NSE lists'):
  nse_equity_sector_catalog_cached.clear();nse_commodity_etf_catalog_cached.clear();st.rerun()
 st.caption(f"Online NSE list: {len(equity_options):,} EQ stocks · {len(commodity_options):,} commodity ETFs")
 market_types=['NSE stocks']+(['NSE commodity ETFs'] if commodity_options else [])
 market_type=st.selectbox('Instrument type',market_types)
 is_commodity=market_type=='NSE commodity ETFs'
 if is_commodity:
  commodity_category=st.selectbox('Commodity',commodity_categories)
  category_funds=[item for item in commodity_catalog if item['category']==commodity_category]
  category_options={f"{item['nse_symbol']} — {item['asset']}":item for item in category_funds}
  commodity_label=st.selectbox('Commodity ETF',list(category_options))
  instrument=category_options[commodity_label];company=instrument['asset'];symbol=instrument['symbol']
 else:
  if equity_options:
   sector_options=['All sectors']+industry_options
   selected_sector=st.selectbox('Sector',sector_options)
   sector_equities=equity_catalog if selected_sector=='All sectors' else [item for item in equity_catalog if item['industry']==selected_sector]
   stock_options={f"{item['nse_symbol']} — {item['name']}":item for item in sector_equities}
   stock_labels=list(stock_options)
   stock_label=st.selectbox('NSE-listed stock',stock_labels,index=0)
   instrument=stock_options[stock_label];company=instrument['name'];symbol=instrument['symbol']
 hn=st.selectbox('Forecast horizon',list(H),index=0);horizon=H[hn];period=st.selectbox('Price history for model',['5y','10y'],index=0);cost=st.number_input('Round-trip cost (bps)',0,200,20,5);run=st.button('Train, evaluate & predict',type='primary',use_container_width=True)
if not commodity_options and not equity_options:
 st.error(f"Could not load NSE instruments online. Stocks: {equity_catalog_status}. Commodity ETFs: {commodity_catalog_status}. Use Refresh online NSE lists to retry.")
 st.stop()
if not is_commodity and not equity_options:
 st.error(f"NSE stock list is unavailable: {equity_catalog_status}. Choose commodity ETFs or retry the online list refresh.")
 st.stop()
try:
 raw=hist(symbol,period);stock=indicators(raw)
 if is_commodity:
  market=hist(instrument['benchmark'],period);sector=hist('USDINR=X',period);market_context={}
 else:
  market=hist('^NSEI',period);sector=optional_hist(sector_proxy(instrument.get('industry')),period);sector=sector if sector is not None else market.copy();market_context={'bank':optional_hist('^NSEBANK',period),'vix':optional_hist('^INDIAVIX',period)}
 reddit_id=config_value('REDDIT_CLIENT_ID');reddit_secret=config_value('REDDIT_CLIENT_SECRET');reddit_agent=config_value('REDDIT_USER_AGENT','indian-stock-ai/1.0');alpha_key=config_value('ALPHAVANTAGE_API_KEY');marketaux_token=config_value('MARKETAUX_API_TOKEN');finnhub_key=config_value('FINNHUB_API_KEY');articles,provider_status=news(company,symbol,alpha_key,marketaux_token,finnhub_key,reddit_id,reddit_secret,reddit_agent,is_commodity);reconcile(symbol,raw);nscore=weighted_sentiment([article for article in articles if not article.get('market_wide')]);mret=market.Close.pct_change().iloc[-1];sret=sector.Close.pct_change().iloc[-1];condition='Bullish' if mret>.003 else 'Bearish' if mret<-.003 else 'Neutral'
except Exception as e:st.error(str(e));st.stop()
tabs=st.tabs(['Prediction','Accuracy & Learning','News','Technical','Data Center','Operations'])
with tabs[0]:
 if is_commodity:
  st.info(f"Commodity proxy: {company} is an NSE-listed ETF. The target is its next-day INR per-unit closing price, using {instrument['benchmark']} and USD/INR as context; this is not an MCX futures quote.")
 else:
  broad_news=[article for article in articles if article.get('market_wide')]
  broad_score=np.average([article['score'] for article in broad_news],weights=[article.get('relevance',1) for article in broad_news]) if broad_news else 0
  bank_return=market_context['bank'].Close.pct_change().iloc[-1] if market_context.get('bank') is not None else np.nan
  vix_level=market_context['vix'].Close.iloc[-1] if market_context.get('vix') is not None else np.nan
  st.caption(f"Market context · NIFTY 50 daily: {mret*100:+.2f}% · selected sector proxy daily: {sret*100:+.2f}% · NIFTY Bank daily: {bank_return*100:+.2f}% · India VIX: {vix_level:.2f} · broad India-market news sentiment: {broad_score:+.2f} ({len(broad_news)} headlines). Missing optional feeds are reported as n/a and fall back to market-derived features.")
 fig=go.Figure([go.Candlestick(x=stock.index,open=stock.Open,high=stock.High,low=stock.Low,close=stock.Close),go.Scatter(x=stock.index,y=stock.EMA20,name='EMA20'),go.Scatter(x=stock.index,y=stock.EMA50,name='EMA50')]);fig.update_layout(height=430,xaxis_rangeslider_visible=False);st.plotly_chart(fig,use_container_width=True)
 r=None
 if run:
  try:
   r=model(stock,market,sector,symbol,horizon,cost,market_context=market_context)
   close=float(stock.Close.iloc[-1])
   wf_saved=save_walk_forward_predictions(symbol,horizon,r['walk_forward'])
   cols=st.columns(7)
   vals=[('Signal',r['signal']),('Raw model score',f"{r['confidence']:.1f}%"),(f"Latest close · {stock.index[-1].date()}",f'₹{close:,.2f}'),('Forecast',f"₹{r['price']:,.2f}"),('Return',f"{r['return']*100:+.2f}%"),('Bullish score',f"{r['bull']:.1f}%"),('Bearish score',f"{r['bear']:.1f}%")]
   for c,(k,v) in zip(cols,vals):c.metric(k,v)
   st.write(f"Historical out-of-fold residual band (not independently calibrated): **₹{r['low']:,.2f} to ₹{r['high']:,.2f}**")
   c=conn()
   c.execute('INSERT OR REPLACE INTO predictions(symbol,prediction_date,horizon,latest_close,predicted_close,predicted_return,signal,confidence,bullish,bearish,low,high,market_condition,market_return,sector_return,news_score,news_json,feature_json,model_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(symbol,stock.index[-1].date().isoformat(),horizon,close,r['price'],r['return'],r['signal'],r['confidence'],r['bull'],r['bear'],r['low'],r['high'],condition,mret,sret,nscore,json.dumps(articles),json.dumps(r['features']),r['version']))
   c.execute('INSERT INTO model_registry(symbol,horizon,version,trained_at,rows,metrics_json,active) VALUES(?,?,?,?,?,?,1)',(symbol,horizon,r['version'],datetime.now(timezone.utc).isoformat(),len(stock),json.dumps(r['metrics'])))
   c.commit();c.close()
   st.success('Forecast saved. Historical walk-forward outcome rows were refreshed from online price history.')
   st.subheader('Historical walk-forward evaluation — not a live success rate')
   M=r['metrics']
   st.caption(f"{wf_saved:,} simulated historical forecasts · {M['non_overlapping_windows']:,} non-overlapping horizon windows · {M['executed_trades']:,} BUY/SELL trades · raw Random Forest probabilities (uncalibrated). Historical news may be incomplete or revised; these are not verified forecasts issued in real time.")
   mc=st.columns(8)
   ev=[('Balanced accuracy',M['balanced_accuracy']),('All-HOLD baseline',M['baseline_accuracy']),('Return MAE (pp)',M['mae']),('Zero-return MAE (pp)',M['naive_mae']),('BUY precision',M['buy_precision']),('SELL precision',M['sell_precision']),('Brier (raw)',M['brier']),('Non-overlap net return',M['non_overlapping_cost_return'])]
   for c,(k,v) in zip(mc,ev):c.metric(k,f'{v:.2f}%')
   if M['balanced_accuracy']<=M['baseline_accuracy'] or M['mae']>=M['naive_mae'] or M['non_overlapping_cost_return']<=0:st.warning('Research gate failed on this historical walk-forward test. Do not treat the forecast as a verified edge.')
  except Exception as e:st.error(str(e))
 if is_commodity:
  st.subheader(f'{commodity_category} ETF comparison — {hn} forecast')
  st.caption('Sorted by model-predicted return over the selected horizon, highest first. Prices are INR per ETF unit; forecasts are not guarantees of returns.')
  comparison_rows=[];unavailable=[]
  with st.spinner(f'Preparing cached next-day forecasts for {len(category_funds)} {commodity_category} ETFs...'):
   for fund in category_funds:
    try:
     if run and fund['symbol']==symbol and r is not None:
      forecast={'close':close,'price':r['price'],'return':r['return']}
     else:
      forecast=commodity_price_forecast_cached(fund['symbol'],fund['benchmark'],period,horizon,DB)
     comparison_rows.append({'Symbol':fund['nse_symbol'],'ETF / Asset':fund['name'],'Last price (₹)':forecast['close'],'Predicted price (₹)':forecast['price'],'Predicted return (%)':forecast['return']*100})
    except Exception as exc:
     unavailable.append(f"{fund['nse_symbol']}: {type(exc).__name__}")
  if comparison_rows:
   comparison=pd.DataFrame(comparison_rows).sort_values('Predicted return (%)',ascending=False,kind='stable')
   st.dataframe(comparison,use_container_width=True,hide_index=True)
  else:
   st.warning(f'Could not generate a forecast for the {commodity_category} ETFs from available price history.')
  if unavailable:
   with st.expander(f'{len(unavailable)} ETF forecasts unavailable'):
    st.write('\n'.join(unavailable))
with tabs[1]:
 st.subheader('Recent 20-session NIFTY 50 + GPTINFRA backtest')
 st.caption('Generates retrospective, out-of-sample one-session forecasts for the 20 latest completed prediction sessions with available next-session prices. The current official NIFTY 50 list is used and GPT Infraprojects (GPTINFRA.NS) is added separately. These are simulations, not forecasts issued live on those dates. A single model is trained only on earlier rows before each 20-session holdout; past news coverage may be incomplete.')
 if st.button('Generate 20-session NIFTY 50 + GPTINFRA records',type='primary'):
  constituents,basket_status=fetch_nifty50_constituents()
  if not constituents:
   st.error(f'Could not load current NIFTY 50 constituents: {basket_status}')
  else:
   catalog_by_symbol={item['nse_symbol']:item for item in equity_catalog}
   universe=[]
   for item in constituents:
    catalog_item=catalog_by_symbol.get(item['symbol'],{})
    universe.append({'symbol':item['ticker'],'nse_symbol':item['symbol'],'name':catalog_item.get('name') or item['name'],'industry':catalog_item.get('industry') or item['industry']})
   gpt=catalog_by_symbol.get('GPTINFRA',{})
   if not any(item['nse_symbol']=='GPTINFRA' for item in universe):
    universe.append({'symbol':gpt.get('symbol','GPTINFRA.NS'),'nse_symbol':'GPTINFRA','name':gpt.get('name','GPT Infraprojects Limited'),'industry':gpt.get('industry','Construction')})
   successes=[];failures=[];progress=st.progress(0,text=f'Loading shared market data · 0/{len(universe)} instruments')
   try:
    batch_market=hist('^NSEI','5y');batch_context={'bank':optional_hist('^NSEBANK','5y'),'vix':optional_hist('^INDIAVIX','5y')}
    for number,item in enumerate(universe,1):
     try:
      batch_raw=hist(item['symbol'],'5y');batch_stock=indicators(batch_raw)
      sector_data=optional_hist(sector_proxy(item['industry']),'5y')
      if sector_data is None:sector_data=batch_market
      result=model(batch_stock,batch_market,sector_data,item['symbol'],1,cost,market_context=batch_context,recent_holdout=20)
      saved=save_walk_forward_predictions(item['symbol'],1,result['walk_forward'],run_tag='nifty50-20session-walkforward-v2')
      c=conn();c.execute('INSERT INTO model_registry(symbol,horizon,version,trained_at,rows,metrics_json,active) VALUES(?,?,?,?,?,?,1)',(item['symbol'],1,result['version'],datetime.now(timezone.utc).isoformat(),len(batch_stock),json.dumps(result['metrics'])));c.commit();c.close()
      successes.append({'Symbol':item['nse_symbol'],'Company':item['name'],'Records saved':saved,'Balanced accuracy (%)':result['metrics']['balanced_accuracy'],'Return MAE (pp)':result['metrics']['mae'],'Zero-return MAE (pp)':result['metrics']['naive_mae'],'Status':'OK'})
     except Exception as exc:
      failures.append(f"{item['nse_symbol']}: {type(exc).__name__} — {str(exc)[:180]}")
     progress.progress(number/len(universe),text=f"Processed {number}/{len(universe)} instruments · {len(successes)} completed")
    st.session_state['nifty_batch_result']={'successes':successes,'failures':failures,'basket_status':basket_status}
   except Exception as exc:
    st.error(f'Could not load shared market context: {type(exc).__name__}: {exc}')
   finally:
    progress.empty()
 batch_result=st.session_state.get('nifty_batch_result')
 if batch_result:
  st.caption(f"Constituent source: {batch_result['basket_status']} · {len(batch_result['successes'])} instruments completed · {len(batch_result['failures'])} failed or skipped.")
  if batch_result['successes']:
   st.dataframe(pd.DataFrame(batch_result['successes']).sort_values('Symbol'),use_container_width=True,hide_index=True)
  if batch_result['failures']:
   with st.expander(f"{len(batch_result['failures'])} unavailable instruments"):
    st.write('\n'.join(batch_result['failures']))
 c=conn();batch_rows=[dict(row) for row in c.execute("SELECT symbol,prediction_date,target_date,latest_close,predicted_close,predicted_return,actual_close,actual_return,percentage_error,signal,direction_correct,model_version FROM predictions WHERE model_version LIKE ? ORDER BY prediction_date DESC,symbol",('%-nifty50-20session-walkforward-v2',)).fetchall()];c.close()
 if batch_rows:
  batch_frame=pd.DataFrame(batch_rows);batch_frame['ticker']=batch_frame.symbol.str.replace('.NS','',regex=False)
  summary=batch_frame.groupby('ticker').agg(Records=('ticker','size'),First_session=('prediction_date','min'),Last_session=('prediction_date','max'),Direction_accuracy_pct=('direction_correct',lambda values:values.mean()*100),Mean_absolute_price_error_pct=('percentage_error','mean')).reset_index().sort_values('ticker')
  st.subheader('Saved retrospective results by instrument')
  st.dataframe(summary,use_container_width=True,hide_index=True)
  st.subheader('20-session forecast and actual-close records')
  st.dataframe(batch_frame[['ticker','prediction_date','target_date','latest_close','predicted_close','actual_close','predicted_return','actual_return','percentage_error','signal','direction_correct']].rename(columns={'ticker':'Symbol','prediction_date':'Forecast session','target_date':'Actual target session','latest_close':'Close at forecast (₹)','predicted_close':'Predicted close (₹)','actual_close':'Actual close (₹)','predicted_return':'Predicted return','actual_return':'Actual return','percentage_error':'Absolute price error (%)','signal':'Signal','direction_correct':'Direction correct'}),use_container_width=True,hide_index=True)
 else:
  st.info('No batch records yet. Generate the recent NIFTY 50 + GPTINFRA holdout above to create them.')
 c=conn();rows=[dict(x) for x in c.execute('SELECT * FROM predictions WHERE symbol=? ORDER BY prediction_date DESC',(symbol,)).fetchall()];c.close()
 if rows:
  d=pd.DataFrame(rows);walkforward_mask=d.model_version.fillna('').str.endswith('-walkforward-v2');walkforward=d[walkforward_mask].dropna(subset=['actual_close']);live=d[~walkforward_mask];live_matured=live.dropna(subset=['actual_close']);live_pending=live[live.actual_close.isna()]
  st.subheader('Historical walk-forward simulation from online price history')
  st.caption('These are retrospective out-of-fold predictions recomputed from downloaded price history, not forecasts actually issued in those past sessions and not a proven future success rate. Historical news may be incomplete or revised. Results are separated by horizon.')
  if len(walkforward):
   wf_summary=walkforward.groupby('horizon').agg(Simulated=('symbol','size'),Direction_accuracy_pct=('direction_correct',lambda values:values.mean()*100),Mean_price_error_pct=('percentage_error','mean'),First_session=('prediction_date','min'),Last_session=('prediction_date','max')).reset_index()
   st.dataframe(wf_summary,use_container_width=True,hide_index=True)
  else:
   st.info('No online-price walk-forward runs saved yet. Click “Train, evaluate & predict” to download history, run the walk-forward test, and populate this simulation table.')
  st.subheader('Actual outcomes for forecasts issued by this app')
  st.caption('This section excludes retrospective simulations. Target prices are reconciled from downloaded daily prices when the app runs after their target horizon matures.')
  if len(live_matured):
   live_summary=live_matured.groupby('horizon').agg(Matured=('symbol','size'),Direction_accuracy_pct=('direction_correct',lambda values:values.mean()*100),Mean_price_error_pct=('percentage_error','mean'),First_prediction=('prediction_date','min'),Last_prediction=('prediction_date','max')).reset_index()
   st.dataframe(live_summary,use_container_width=True,hide_index=True)
   st.dataframe(live_matured[['prediction_date','target_date','horizon','predicted_close','actual_close','percentage_error','signal','direction_correct','model_version']].sort_values('prediction_date',ascending=False),use_container_width=True,hide_index=True)
  else:
   st.info(f'{len(live_pending)} app-issued predictions are still waiting for target dates. No matured live success rate is available yet.')
  st.caption('Community Cloud local SQLite files may reset. For live outcomes that survive redeployments and are shared across users, configure durable external storage; local historical simulations are regenerated when training runs.')
 else:
  st.info('No saved outcomes yet. Train the selected instrument to run an online-price walk-forward simulation and save an app-issued forecast for future reconciliation.')
with tabs[2]:
 if st.button('Refresh news providers now'):
  alpha_key=config_value('ALPHAVANTAGE_API_KEY');marketaux_token=config_value('MARKETAUX_API_TOKEN');finnhub_key=config_value('FINNHUB_API_KEY');reddit_id=config_value('REDDIT_CLIENT_ID');reddit_secret=config_value('REDDIT_CLIENT_SECRET');reddit_agent=config_value('REDDIT_USER_AGENT','indian-stock-ai/1.0');news.clear();articles,provider_status=news(company,symbol,alpha_key,marketaux_token,finnhub_key,reddit_id,reddit_secret,reddit_agent,is_commodity);nscore=weighted_sentiment([article for article in articles if not article.get('market_wide')])
 with st.expander('Alpha Vantage free daily price-data check'):
  alpha_key=config_value('ALPHAVANTAGE_API_KEY')
  suggested_av_symbol=symbol.rsplit('.',1)[0]+'.BSE'
  av_symbol=st.text_input('Alpha Vantage symbol (editable)',value=suggested_av_symbol,help='Alpha Vantage documents BSE symbols such as RELIANCE.BSE. Verify the provider symbol for the selected company.')
  st.caption('This makes a TIME_SERIES_DAILY request with outputsize=compact (up to 100 OHLCV rows). The daily price endpoint is free-key compatible; NEWS_SENTIMENT remains premium. Data is raw/as-traded and is displayed separately, not merged into the auto-adjusted Yahoo model history.')
  if st.button('Fetch Alpha Vantage daily prices',disabled=not bool(alpha_key)):
   rows,detail=alphavantage_daily_cached(alpha_key,av_symbol.strip())
   st.session_state['av_daily_result']={'symbol':av_symbol.strip(),'rows':rows,'detail':detail,'called':True}
  daily_result=st.session_state.get('av_daily_result',{})
  if daily_result.get('symbol')==av_symbol.strip():
   st.caption(f"Endpoint status: {daily_result.get('detail','')} · Symbol: {av_symbol.strip()}")
   if daily_result.get('rows'):
    st.dataframe(pd.DataFrame(daily_result['rows']).set_index('Date'),use_container_width=True)
 sentiment_cols=st.columns(2);market_articles=[article for article in articles if article.get('market_wide')];market_news_score=weighted_sentiment(market_articles)
 sentiment_cols[0].metric('Company / instrument news sentiment',f'{(nscore+1)*50:.1f}/100')
 sentiment_cols[1].metric('Broad India-market news sentiment',f'{(market_news_score+1)*50:.1f}/100')
 st.caption('Provider checks show whether your API keys were found, whether app.py called each endpoint, how many articles were returned, and provider response details. Alpha Vantage NEWS_SENTIMENT is skipped because it is premium-only; the free TIME_SERIES_DAILY compact endpoint is available in the price-data check above. Finnhub general market news uses its free /news?category=general endpoint; broad headlines receive lower model relevance than company-specific news. News status is cached for up to 5 minutes; click Refresh news providers now to force fresh requests. Alpha daily results are cached for 6 hours to protect the daily request quota.')
 daily_result=st.session_state.get('av_daily_result',{})
 daily_called=daily_result.get('symbol')==av_symbol.strip() and daily_result.get('called',False)
 daily_rows=daily_result.get('rows',[]) if daily_called else []
 daily_detail=daily_result.get('detail','Not called; use the manual check above.') if daily_called else ('Not configured' if not alpha_key else 'Not called; manual check above')
 provider_status.append({'Provider':'Alpha Vantage TIME_SERIES_DAILY (free compact)','Configured':'Yes' if alpha_key else 'No','Called':'Yes' if daily_called else 'No','Articles':len(daily_rows),'Status':'OK' if daily_called and daily_rows else ('Error' if daily_called else ('Not configured' if not alpha_key else 'Not called')),'Details':daily_detail,'Checked UTC':datetime.now(timezone.utc).isoformat()})
 st.dataframe(pd.DataFrame(provider_status),use_container_width=True,hide_index=True)
 for a in articles:st.markdown(f"[{'🟢' if a['score']>.05 else '🔴' if a['score']<-.05 else '🟡'} {a['title']}]({a['url']}) · {a['source']} · {a['score']:+.2f}")
with tabs[3]:
 cols=st.columns(8)
 for c,k in zip(cols,['RSI','MACD','MACDS','EMA20','EMA50','EMA200','ATR','VOL']):c.metric(k,f"{stock[k].iloc[-1]:.2f}")
with tabs[4]:
 st.subheader('Backfill historical NSE company announcements')
 if is_commodity:
  st.info('NSE issuer-announcement backfill is for listed companies. Commodity instruments use commodity-focused news and sentiment instead.')
 today_ist=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5,minutes=30))).date()
 nse_range=st.date_input('NSE announcement date range',value=(today_ist-timedelta(days=365),today_ist),key='nse_announcement_range_12m')
 if len(nse_range)==2:
  nse_start,nse_end=nse_range
  if st.button('Fetch historical NSE announcements',disabled=is_commodity):
   nse_rows,nse_detail=fetch_nse_company_announcements(symbol,nse_start,nse_end,limit=500)
   st.session_state['nse_history_result']={'symbol':symbol,'start':nse_start.isoformat(),'end':nse_end.isoformat(),'rows':nse_rows,'detail':nse_detail}
  nse_result=st.session_state.get('nse_history_result',{})
  nse_rows=[]
  if (nse_result.get('symbol'),nse_result.get('start'),nse_result.get('end'))==(symbol,nse_start.isoformat(),nse_end.isoformat()):
   st.caption(nse_result.get('detail',''))
   nse_rows=nse_result.get('rows',[])
  if nse_rows:
   st.dataframe(pd.DataFrame(nse_rows)[['published','title','source','url']],use_container_width=True,hide_index=True)
   if st.button('Save fetched NSE announcements to news history'):
    c=conn();saved=0;captured=datetime.now(timezone.utc).isoformat()
    for item in nse_rows:
     published=item.get('published','');title=item.get('title','').strip()
     if not published or not title:continue
     score=AN.polarity_scores(item.get('sentiment_text',title))['compound']
     saved+=c.execute('INSERT OR IGNORE INTO news_history(symbol,captured_at,published_at,source,headline,url,sentiment,relevance) VALUES(?,?,?,?,?,?,?,?)',(symbol,captured,published,item.get('source','NSE India'),title,item.get('url',''),score,float(item.get('relevance',1)))).rowcount
    c.commit();c.close();st.success(f'Saved {saved} new NSE announcements; duplicates were skipped.')
  elif nse_result:
   st.info('No matching NSE announcements were returned for this stock and date range.')
 else:
  st.info('Choose a start and end date to backfill historical NSE announcements.')
 st.caption('The model uses the selected 5-year price history by default, rolling 12-month NSE announcement sentiment, rolling 90-day non-social news sentiment, and rolling 30-day Reddit/Bluesky sentiment. NSE RSS only contains recent items; use the 12-month archive backfill and save the results to make that history available to training. A longer history or more sentiment inputs can help evaluation but do not guarantee better forecasts.')
 st.subheader('Upload point-in-time historical news');f=st.file_uploader('News CSV',type='csv')
 if f:
  d=pd.read_csv(f);req={'published_at','symbol','headline','source','sentiment'}
  if not req<=set(d.columns):st.error('Missing columns: '+str(req-set(d.columns)))
  else:
   d.published_at=pd.to_datetime(d.published_at,utc=True,errors='coerce');d.sentiment=pd.to_numeric(d.sentiment,errors='coerce').clip(-1,1);d['relevance']=pd.to_numeric(d.get('relevance',1),errors='coerce').fillna(1).clip(0,1);d['url']=d.get('url','');st.dataframe(d.head())
   if st.button('Import validated news'):
    c=conn();count=0
    for _,z in d.dropna(subset=['published_at','symbol','headline','sentiment']).iterrows():count+=c.execute('INSERT OR IGNORE INTO news_history(symbol,captured_at,published_at,source,headline,url,sentiment,relevance) VALUES(?,?,?,?,?,?,?,?)',(z.symbol,z.published_at.isoformat(),z.published_at.isoformat(),z.source,z.headline,z.url,float(z.sentiment),float(z.relevance))).rowcount
    c.commit();c.close();st.success(f'Imported {count} articles')
 st.download_button('News template',open('sample_news.csv','rb').read(),'sample_news.csv')
with tabs[5]:
 st.info(f"Persistent storage backend: {database_backend(DATABASE_URL)}")
 st.markdown('''**Daily learning process**
1. Run after market close to capture finalized prices and timestamped news.
2. Preserve `stock_ai.db`; it contains the feedback loop.
3. Re-run after the target date to reconcile forecasts with actual closes.
4. Add licensed historical news through Data Center.
5. Promote a challenger only when it beats direction and price baselines, stays calibrated, and remains profitable after costs.

For real multi-user production, migrate SQLite to managed PostgreSQL, run ingestion/training as scheduled workers, keep immutable raw data in object storage, manage models in a registry, and protect the app with SSO, WAF, backups and monitoring.''')
