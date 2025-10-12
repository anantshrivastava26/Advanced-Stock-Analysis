import os
import requests
from datetime import datetime, timedelta

NEWSAPI_URL = 'https://newsapi.org/v2/everything'

def fetch_headlines(query, from_dt, to_dt, page=1, page_size=100, api_key=None):
    key = api_key or os.getenv('NEWSAPI_KEY')
    if not key:
        raise ValueError('NEWSAPI_KEY not set')
    params = {
        'q': query,
        'from': from_dt,
        'to': to_dt,
        'language': 'en',
        'page': page,
        'pageSize': page_size,
        'apiKey': key,
        'sortBy': 'relevancy'
    }
    r = requests.get(NEWSAPI_URL, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def headlines_by_date(query, start, end, api_key=None):
    st = datetime.fromisoformat(start)
    ed = datetime.fromisoformat(end)
    delta = ed - st
    out = {}
    for i in range(delta.days + 1):
        cur = st + timedelta(days=i)
        day_from = cur.strftime('%Y-%m-%d')
        day_to = (cur + timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            data = fetch_headlines(query, day_from, day_to, api_key=api_key)
            titles = [a.get('title') or a.get('description') or '' for a in data.get('articles', [])]
            out[day_from] = titles
        except Exception:
            out[day_from] = []
    return out
