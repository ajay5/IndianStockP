import os, urllib.parse, requests, feedparser
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_an=SentimentIntensityAnalyzer()

def sentiment(text):
    s=_an.polarity_scores(text)["compound"]
    return ("Positive" if s>0.05 else "Negative" if s<-0.05 else "Neutral"),s

def fetch_news(query,limit=15):
    out=[]; key=os.getenv("NEWS_API_KEY")
    if key:
        try:
            r=requests.get("https://newsapi.org/v2/everything",params={"q":query,"language":"en","sortBy":"publishedAt","pageSize":limit,"apiKey":key},timeout=10); r.raise_for_status()
            for a in r.json().get("articles",[]): out.append({"title":a.get("title",""),"url":a.get("url",""),"source":a.get("source",{}).get("name","NewsAPI"),"published":a.get("publishedAt","")})
        except Exception: pass
    if not out:
        url="https://news.google.com/rss/search?q="+urllib.parse.quote(query+" stock India")+"&hl=en-IN&gl=IN&ceid=IN:en"
        feed=feedparser.parse(url)
        for e in feed.entries[:limit]: out.append({"title":e.get("title",""),"url":e.get("link",""),"source":e.get("source",{}).get("title","Google News"),"published":e.get("published","")})
    for a in out: a["label"],a["score"]=sentiment(a["title"])
    return out
