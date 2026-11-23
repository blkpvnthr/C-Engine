"""Read-only Alpaca intraday ranking overlay. Run daily scanner first."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


class Alpaca:
    def __init__(self, feed="iex"):
        if feed != "iex":
            raise ValueError("Only Alpaca IEX data is enabled")
        self.feed=feed
        self.s=requests.Session()
        key=os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY')
        secret=os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
        if not key or not secret:
            raise RuntimeError('Set Alpaca API credentials in environment')
        self.s.headers.update({'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret})
    def get(self,path,params=None, trading=False):
        base=os.getenv('APCA_API_BASE_URL','https://paper-api.alpaca.markets').rstrip(
            '/') if trading else 'https://data.alpaca.markets'
        for attempt in range(4):
            r=self.s.get(base+path,params=params,timeout=30)
            if r.status_code==429 or r.status_code>=500:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError('Alpaca retry limit exceeded')
    def pages(self,kind,symbols,start,end,**extra):
        result={s:[] for s in symbols}
        for i in range(0,len(symbols),20):
            params=dict(symbols=','.join( symbols[i:i+20]),
    start=start.isoformat(),
    end=end.isoformat(),
    feed=self.feed,
    limit=10000,
    **extra)
            while True:
                d=self.get('/v2/stocks/'+kind,params)
                for s,rows in (d.get(kind) or {}).items():
                    result[s].extend(rows)
                token=d.get('next_page_token')
                if not token:
                    break
                params['page_token']=token
        return result

def flow(trades):
    # Tick-rule estimate; unchanged prices inherit last nonzero direction.
    if not trades:
        return None
    d=pd.DataFrame(trades).sort_values('t',kind='stable').drop_duplicates(
        subset=['t','i','x'] if {'i','x'}<=set(trades[0]) else ['t','p','s'])
    d['t']=pd.to_datetime(d.t,utc=True,format='ISO8601')
    d=d.set_index('t')
    direction=np.sign(d.p.diff()).replace(0,np.nan).ffill().fillna(0)
    signed=direction*d.s
    end=d.index.max()
    recent=d.index>end-pd.Timedelta(seconds=60)
    prior=(d.index<=end-pd.Timedelta(seconds=60))&(d.index>end-pd.Timedelta(seconds=120))
    def imbalance(mask):
        vol=d.s[mask].sum()
        return float(signed[mask].sum()/vol) if vol else np.nan
    a,b=imbalance(recent),imbalance(prior)
    return dict(
        cvd_5m=float(signed.sum()), delta_60=a, prior_delta_60=b,
        trade_count=len(d), last_trade=end,
        exhaustion=bool(b<-.10 and a>.10 and d.p[recent].iloc[-1]>=d.p[recent].iloc[0]))

def atomic(path,content):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(content)
    tmp.replace(path)

def scan(api,base,out,top,spread_limit):
    atomic(out/'live_universe.txt','')
    clock=api.get('/v2/clock',trading=True)
    if not clock['is_open']:
        print('Regular session closed')
        return
    now=pd.Timestamp(clock['timestamp']).tz_convert('UTC')
    day=now.tz_convert('America/New_York')
    session=day.normalize()+pd.Timedelta(hours=9,minutes=30)
    session=session.tz_convert('UTC')
    end=now.floor('min')
    start=max(session,end-pd.Timedelta(minutes=5))
    if end-session<pd.Timedelta(minutes=5):
        return
    symbols=base.symbol.tolist()
    refs=['SPY','QQQ','DIA','IWM','SMH','XLK','XOP','NVDA','TSLA',
        'XLP','XLV','XLU','XLE','XLF','XLI','XLB','XLRE','XLC','XLY']
    all_symbols=list(dict.fromkeys(symbols+refs))
    bars=api.pages('bars',all_symbols,session,end,timeframe='1Min',adjustment='raw')
    trades=api.pages('trades',symbols,start,end)
    quotes={}
    for i in range(0,len(all_symbols),50):
        quotes.update(api.get('/v2/stocks/quotes/latest',
                      dict(symbols=','.join(all_symbols[i:i+50]),feed=api.feed)).get('quotes',{}))
    def context(s):
        d=pd.DataFrame(bars.get(s,[]))
        q=quotes.get(s,{})
        if len(d)<5 or not q:
            return None
        bid,ask=q.get('bp',0),q.get('ap',0)
        if not 0<bid<=ask or q.get('bs',0)<=0 or q.get('as',0)<=0:
            return None
        stamp=pd.Timestamp(q['t'])
        current=pd.Timestamp.now(tz='UTC')
        if not 0<=(current-stamp).total_seconds()<=30:
            return None
        if (end-pd.Timestamp(d.t.iloc[-1])).total_seconds()>120:
            return None
        volume=d.v.sum()
        vwap=float((d.v*d.vw).sum()/volume) if volume else np.nan
        mid=(bid+ask)/2
        return dict(mid=mid,
    vwap=vwap,
    spread=(ask-bid)/mid,
    r_session=mid/d.o.iloc[0]-1,
    r5=mid/d.c.iloc[-5]-1,
    volume=int(volume),
    quote_time=str(stamp))
    ctx={s:context(s) for s in all_symbols}
    if any(ctx[s] is None for s in refs[:4]):
        raise RuntimeError('Missing fresh benchmark quotes/bars')
    bearish=sum(ctx[s]['mid']<ctx[s]['vwap'] and ctx[s]['r5']<0 for s in refs[:4])>=3
    bullish=sum(ctx[s]['mid']>ctx[s]['vwap'] and ctx[s]['r5']>0 for s in refs[:4])>=3
    from market_analyzer import BULL, INVERSE, SECTOR_MAP
    rows=[]
    for r in base.itertuples():
        s=r.symbol
        x=ctx[s]
        f=flow(trades.get(s,[]))
        if not x or not f or f['trade_count']<20 or (end-f['last_trade']).total_seconds()>60:
            continue
        if x['spread']>spread_limit or not x['mid']>x['vwap'] or x['r5']<=0 or not f['delta_60']>0:
            continue
        underlying=ctx.get(INVERSE.get(s) or BULL.get(s,''))
        if s in INVERSE and (
    not underlying or underlying['mid']>=underlying['vwap'] or underlying['r5']>=0):
            continue
        if s in BULL and (
    not underlying or underlying['mid']<=underlying['vwap'] or underlying['r5']<=0):
            continue
        sector=ctx.get(SECTOR_MAP.get(getattr(r,'sector',''),'') or '')
        rs=x['r5']-ctx['SPY']['r5']
        sector_rs=x['r5']-sector['r5'] if sector else 0
        # Linear pace estimate, not a time-of-day historical RVOL.
        pace=x['volume']/max(float(r.adv)*max((end-session).total_seconds()/23400,.01),1)
        rows.append(dict(symbol=s,daily_score=r.score,**x,**f,rs=rs,sector_rs=sector_rs,
                    volume_pace=pace,sector=getattr(r,'sector','Unknown')))
    df=pd.DataFrame(rows)
    selected=[]
    if not df.empty:
        def rank(k):
            return df[k].rank(pct=True)
        df['live_score']=.25*rank('daily_score')+.20*rank('rs')+.15*rank('r5')+.15*rank(
            'delta_60')+.10*rank('volume_pace')+.10*(1-rank('spread'))+.05*df.exhaustion.astype(float)
        df=df.sort_values('live_score',ascending=False)
        groups=set()
        counts={}
        for r in df.itertuples():
            g=INVERSE.get(r.symbol) or BULL.get(r.symbol)
            if g and g in groups:
                continue
            if r.sector!='Unknown' and counts.get(r.sector,0)>=max(1,int(top*.3)):
                continue
            selected.append(r.symbol)
            counts[r.sector]=counts.get(r.sector,0)+1
            if g:
                groups.add(g)
            if len(selected)>=top:
                break
    atomic(out/'live_ranked.csv',df.to_csv(index=False) if not df.empty else 'symbol,live_score\n')
    atomic(out/'live_context.json',
    json.dumps(dict(as_of=str(now),
    feed=api.feed,
    regime='bearish' if bearish else 'bullish' if bullish else 'mixed',
    benchmarks={s:ctx[s] for s in refs[:4]},
    selected=selected),
    indent=2))
    atomic(out/'live_universe.txt',', '.join(map(repr,selected))+'\n')
    print('Selected',len(selected),selected)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='market_scan')
    p.add_argument('--feed',choices=['iex'],default='iex')
    p.add_argument('--top',type=int,default=30)
    p.add_argument('--pool',type=int,default=80)
    p.add_argument('--refresh',type=int,default=60)
    p.add_argument('--max-spread',type=float,default=.001)
    p.add_argument('--once',action='store_true')
    a=p.parse_args()
    if min(a.top,a.pool,a.refresh)<=0 or a.pool<a.top:
        p.error('Invalid pool/top/refresh')
    out=Path(a.output)
    api=Alpaca(a.feed)
    while True:
        try:
            atomic(out/'live_universe.txt','')
            context=json.loads((out/'market_context.json').read_text())
            age=(pd.Timestamp.now(tz='America/New_York').date()- \
                 pd.Timestamp(context['as_of']).date()).days
            if not 0<=age<=4:
                raise RuntimeError('Daily scan stale')
            base=pd.read_csv(out/'ranked.csv').sort_values('score',ascending=False).head(a.pool)
            if not base.empty:
                scan(api,base,out,a.top,a.max_spread)
        except Exception as e:
            print('Scan failed:',e,flush=True)
            if a.once:
                raise
        if a.once:
            return
        time.sleep(a.refresh)
if __name__=='__main__':
    main()
