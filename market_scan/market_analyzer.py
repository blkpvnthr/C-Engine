"""End-of-day market scanner. Run: python market_analyzer.py --top 30"""
import argparse, io, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf

REFERENCES = ['^IXIC','^GSPC','^DJI','^RUT','^VIX','^TNX','SPY','QQQ','DIA','IWM','HYG','TLT','UUP','GLD']
SECTORS = ['XLK','XLC','XLY','XLP','XLV','XLU','XLE','XLF','XLI','XLB','XLRE','SMH']
INVERSE = {'PSQ':'QQQ','QID':'QQQ','SQQQ':'QQQ','SH':'SPY','SDS':'SPY','SPXU':'SPY','DOG':'DIA','SDOW':'DIA','RWM':'IWM','TZA':'IWM','SOXS':'SMH','TECS':'XLK','DRIP':'XOP'}
BULL = {'TQQQ':'QQQ','UPRO':'SPY','UDOW':'DIA','TNA':'IWM','SOXL':'SMH','NVDL':'NVDA','TSLL':'TSLA'}
SECTOR_MAP = {'Technology':'XLK','Healthcare':'XLV','Financial Services':'XLF','Consumer Defensive':'XLP','Consumer Cyclical':'XLY','Industrials':'XLI','Energy':'XLE','Utilities':'XLU','Basic Materials':'XLB','Real Estate':'XLRE','Communication Services':'XLC'}

def discover():
    frames=[]
    for file,key in [('nasdaqlisted.txt','Symbol'),('otherlisted.txt','ACT Symbol')]:
        r=requests.get('https://www.nasdaqtrader.com/dynamic/SymDir/'+file,timeout=30); r.raise_for_status()
        d=pd.read_csv(io.StringIO(r.text),sep='|',dtype=str)
        d=d[d['Test Issue'].eq('N') & d['ETF'].isin(['Y','N'])].copy()
        d=d[~d['Security Name'].str.contains(r'\b(?:Warrant|Right|Unit|Preferred|Depositary|Notes)\b',case=False,na=False)]
        d=d.rename(columns={key:'symbol','ETF':'etf','Security Name':'name'})
        frames.append(d[['symbol','etf','name']])
    d=pd.concat(frames).drop_duplicates('symbol'); d=d[d.symbol.str.fullmatch(r'[A-Z]{1,5}(?:[.][A-Z])?')].copy()
    d['symbol']=d.symbol.str.replace('.','-',regex=False)
    return d.set_index('symbol')

def download(symbols,period='6mo',batch=100):
    result={}
    for i in range(0,len(symbols),batch):
        group=symbols[i:i+batch]
        for attempt in range(3):
            try:
                raw=yf.download(group,period=period,interval='1d',auto_adjust=True,group_by='ticker',threads=8,progress=False,timeout=30)
                for s in group:
                    try:
                        d=raw[s] if isinstance(raw.columns,pd.MultiIndex) else raw
                        d=d[['Open','High','Low','Close','Volume']].dropna(subset=['Close']).copy()
                        if len(d): result[s]=d
                    except (KeyError,ValueError): pass
                break
            except Exception as e:
                print(f'Batch {i}: {e}'); time.sleep(2**attempt)
        time.sleep(.3)
    return result

def metrics(d):
    if len(d)<65:return None
    c,h,l,v=[d[k].astype(float) for k in ['Close','High','Low','Volume']]
    if (c.tail(65)<=0).any():return None
    ret=lambda n: float(c.iloc[-1]/c.iloc[-1-n]-1)
    smooth=lambda x:x.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    delta=c.diff(); gain=smooth(delta.clip(lower=0)); loss=smooth(-delta.clip(upper=0))
    rsi=100*gain/(gain+loss).replace(0,np.nan)
    up=h.diff(); down=-l.diff(); tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=smooth(tr); plus=100*smooth(up.where((up>down)&(up>0),0))/atr; minus=100*smooth(down.where((down>up)&(down>0),0))/atr
    adx=smooth(100*(plus-minus).abs()/(plus+minus).replace(0,np.nan))
    av=v.iloc[-21:-1].mean(); mid=c.rolling(20).mean(); std=c.rolling(20).std()
    return dict(date=str(d.index[-1].date()),price=float(c.iloc[-1]),r1=ret(1),r5=ret(5),r20=ret(20),r60=ret(60),adv=float(av),dollar_volume=float((c*v).iloc[-21:-1].mean()),rvol=float(v.iloc[-1]/av) if av>0 else 0.,rsi=float(rsi.iloc[-1]),adx=float(adx.iloc[-1]),di_plus=float(plus.iloc[-1]),di_minus=float(minus.iloc[-1]),above_sma20=bool(c.iloc[-1]>mid.iloc[-1]),above_sma50=bool(c.iloc[-1]>c.rolling(50).mean().iloc[-1]),bb_mid=float(mid.iloc[-1]),bb_upper=float((mid+2*std).iloc[-1]),bb_lower=float((mid-2*std).iloc[-1]),atr_pct=float(atr.iloc[-1]/c.iloc[-1]))

def regime(m):
    core=[m[s] for s in ['SPY','QQQ','DIA','IWM']]; vix=m['^VIX']['price']; breadth=np.mean([x['above_sma20'] for x in core]); trend=np.mean([x['r5'] for x in core])
    if vix>=30 and trend<0:return 'panic'
    if breadth<=.25 and trend<0:return 'bearish'
    if breadth>=.75 and trend>0:return 'bullish'
    if trend<0 and np.mean([x['r1'] for x in core])>0 and m['^VIX']['r1']<0:return 'recovery'
    return 'neutral'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--top',type=int,default=30); p.add_argument('--min-price',type=float,default=5); p.add_argument('--min-volume',type=float,default=500000); p.add_argument('--min-dollar-volume',type=float,default=25e6); p.add_argument('--min-rvol',type=float,default=1.0); p.add_argument('--output',default='market_scan'); p.add_argument('--stocks-only',action='store_true'); p.add_argument('--sector-check',type=int,default=60,help='Fetch sector metadata for this many finalists'); p.add_argument('--allow-partial',action='store_true'); a=p.parse_args()
    if a.top<1: p.error('--top must be positive')
    universe=discover(); refs=list(dict.fromkeys(REFERENCES+SECTORS+list(INVERSE.values())+list(BULL.values())))
    data=download(refs); m={s:x for s,d in data.items() if (x:=metrics(d)) is not None}
    for s in ['SPY','QQQ','DIA','IWM','^VIX']:
        if s not in m:raise RuntimeError('Missing mandatory market reference: '+s)
    date=m['SPY']['date']; now=pd.Timestamp.now(tz='America/New_York'); today=str(now.date())
    if date==today and now.hour<17:raise RuntimeError('Current daily bar may be incomplete. Run after 5 PM ET or before the next open.')
    if (pd.Timestamp(today)-pd.Timestamp(date)).days>4:raise RuntimeError('Market reference data appears stale')
    if any(m[s]['date']!=date for s in ['QQQ','DIA','IWM','^VIX']):raise RuntimeError('Reference dates disagree')
    state=regime(m); print('Market regime:',state,'as of',date)
    symbols=universe.index.tolist(); data=download(symbols); coverage=len(data)/len(symbols)
    if coverage<.75 and not a.allow_partial:raise RuntimeError(f'Low download coverage: {coverage:.1%}; retry or explicitly allow partial results')
    rows=[]
    for s,d in data.items():
        x=metrics(d)
        if not x or x['date']!=date:continue
        etf=universe.loc[s,'etf']=='Y'
        if a.stocks_only and etf:continue
        if x['price']<a.min_price or x['adv']<a.min_volume or x['dollar_volume']<a.min_dollar_volume or x['rvol']<a.min_rvol:continue
        if x['r5']<=0 or not x['above_sma20'] or not x['di_plus']>x['di_minus']:continue
        if not all(np.isfinite(x[k]) for k in ['rsi','adx','atr_pct','rvol']):continue
        if etf and s not in INVERSE and s not in BULL and s not in REFERENCES and s not in SECTORS:continue # explicit ETF allowlist prevents unknown leverage
        if s in INVERSE or s in BULL:
            underlying=m.get((INVERSE if s in INVERSE else BULL)[s]);
            if not underlying or underlying['date']!=date:continue
            if s in INVERSE and (underlying['r5']>=0 or underlying['above_sma20']):continue
            if s in BULL and (underlying['r5']<=0 or not underlying['above_sma20'] or state=='panic'):continue
        rows.append(dict(symbol=s,etf=etf,**x,rs_spy=x['r5']-m['SPY']['r5'],rs_qqq=x['r5']-m['QQQ']['r5']))
    df=pd.DataFrame(rows)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    context={'as_of':date,'regime':state,'coverage':coverage,'discovered':len(symbols),'references':m,'sector_ranking':sorted([{'symbol':s,**m[s]} for s in SECTORS if s in m],key=lambda x:x['r5'],reverse=True),'limitations':['Daily bars only; liquidity is a dollar-volume proxy, not a measured spread.','Current listing directory has survivorship bias; not suitable for historical backtests.','Weights and filters are heuristic and unvalidated.']}
    (out/'market_context.json').write_text(json.dumps(context,indent=2,default=str))
    if df.empty:
        (out/'ranked.csv').write_text('symbol,score\n');(out/'universe.txt').write_text('');print('No eligible instruments');return
    rank=lambda k:df[k].rank(pct=True)
    df['score']=.25*rank('rs_spy')+.15*rank('rs_qqq')+.20*rank('r20')+.15*rank('dollar_volume')+.15*rank('rvol')+.10*rank('adx')
    df=df.sort_values('score',ascending=False).reset_index(drop=True)
    df['sector']='Unknown';df['sector_rs']=np.nan
    for i in range(min(len(df),max(a.top*2,a.sector_check))):
        if df.loc[i,'etf']:continue
        try:
            sector=yf.Ticker(df.loc[i,'symbol']).get_info().get('sector','Unknown');df.loc[i,'sector']=sector
            ref=m.get(SECTOR_MAP.get(sector,''))
            if ref and ref['date']==date:df.loc[i,'sector_rs']=df.loc[i,'r5']-ref['r5']
        except Exception:pass
    # Sector adjustment is bounded, missing sector metadata is neutral.
    df['score']=(df['score']+.05*np.tanh(df['sector_rs'].fillna(0)/.05)).clip(0,1)
    df=df.sort_values('score',ascending=False);df.to_csv(out/'ranked.csv',index=False)
    selected=[]; groups={}; sectors={}
    for r in df.itertuples():
        group=INVERSE.get(r.symbol) or BULL.get(r.symbol)
        if group and groups.get(group,0)>=1:continue
        if r.sector!='Unknown' and sectors.get(r.sector,0)>=max(1,int(a.top*.3)):continue
        selected.append(r.symbol)
        if group:groups[group]=groups.get(group,0)+1
        sectors[r.sector]=sectors.get(r.sector,0)+1
        if len(selected)>=a.top:break
    line=', '.join(repr(s) for s in selected);(out/'universe.txt').write_text(line+'\n');print(line);print(f'Selected {len(selected)} / {a.top}; download coverage {coverage:.1%}')
if __name__=='__main__':main()
