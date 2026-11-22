"""Read-only asynchronous Alpaca scanner. Run alongside daily market_analyzer.py."""
import asyncio, json, math, time
from types import SimpleNamespace
from collections import deque
from pathlib import Path
import pandas as pd
import websockets
from live_market_analyzer import Alpaca, atomic
from market_analyzer import discover, INVERSE, BULL, SECTORS

REFS=list(dict.fromkeys(['SPY','QQQ','DIA','IWM']+list(INVERSE.values())+list(BULL.values())))

def stamp(x): return pd.Timestamp(x).timestamp()

class State:
    def __init__(self):
        self.quote=None; self.trades=deque(); self.bars={}; self.last=None; self.side=0; self.seen=set()
    def event(self,e):
        t=stamp(e['t']); k=e['T']
        if k=='q':
            if self.quote is None or t>=stamp(self.quote['t']): self.quote=e
        elif k in ('b','u'):
            self.bars[t]=e
        elif k=='t':
            key=(e.get('x'),e.get('i'),t,e['p'],e['s'])
            if key in self.seen or (self.trades and t<self.trades[-1][0]): return
            p=e['p']; q=self.quote; side=0
            if q and 0<=t-stamp(q['t'])<=2 and 0<q['bp']<q['ap']:
                mid=(q['bp']+q['ap'])/2; side=int(p>mid)-int(p<mid)
            if not side and self.last is not None: side=int(p>self.last)-int(p<self.last) or self.side
            self.last=p; self.side=side; self.trades.append((t,p,e['s'],side,key)); self.seen.add(key)
            self.prune(t)
    def prune(self,now):
        while self.trades and self.trades[0][0]<now-300:
            self.seen.discard(self.trades.popleft()[-1])
    def metrics(self,now):
        self.prune(now); q=self.quote
        if not q or not 0<=now-stamp(q['t'])<=10 or not 0<q['bp']<=q['ap']: return None
        if min(q['bs'],q['as'])<=0: return None
        bars=[(t,b) for t,b in sorted(self.bars.items()) if t+60<=now]
        if len(bars)<5 or now-bars[-1][0]>150: return None
        volume=sum(b['v'] for _,b in bars)
        if not volume:return None
        mid=(q['bp']+q['ap'])/2; vwap=sum(b['v']*b['vw'] for _,b in bars)/volume
        def window(lo,hi): return [r for r in self.trades if now-hi<=r[0]<now-lo]
        def delta(rows):
            v=sum(r[2] for r in rows); return sum(r[2]*r[3] for r in rows)/v if v else 0
        recent=window(0,60); prior=window(60,120); d=delta(recent)
        return dict(mid=mid,vwap=vwap,spread=(q['ap']-q['bp'])/mid,volume=volume,depth=min(q['bs'],q['as']),r5=mid/bars[-5][1]['c']-1,delta_60=d,cvd_5m=sum(r[2]*r[3] for r in self.trades),delta_10=delta(window(0,10)),delta_30=delta(window(0,30)),delta_300=delta(window(0,300)),trade_count=len(self.trades),recent_count=len(recent),exhaustion=bool(recent and prior and delta(prior)<-.1 and d>.1 and recent[-1][1]>=prior[-1][1]))

def discover_pool(api,pool):
    directory=discover(); symbols=[s for s in directory.index if s not in REFS and '-' not in s and (directory.loc[s,'etf']=='N' or s in INVERSE or s in BULL)]
    scores=[]
    for i in range(0,len(symbols),200):
        batch=symbols[i:i+200]; data=api.get('/v2/stocks/snapshots',dict(symbols=','.join(batch),feed=api.feed))
        for s,v in data.items():
            d=v.get('dailyBar') or {}; prev=v.get('prevDailyBar') or {}; p=d.get('c',0); vol=d.get('v',0); old=prev.get('c',0)
            if p>=5 and old>0 and max(vol*p,prev.get('v',0)*old)>=25e6 and max(vol,prev.get('v',0))>=500000:
                scores.append((s,p/old-1,vol*p))
    df=pd.DataFrame(scores,columns=['symbol','gain','dollars'])
    if df.empty: raise RuntimeError('Discovery returned no liquid candidates')
    df['score']=.65*df.gain.rank(pct=True)+.35*df.dollars.rank(pct=True)
    return df.sort_values('score',ascending=False).head(pool).symbol.tolist()

def bootstrap(api,symbols,session):
    end=pd.Timestamp.now(tz='UTC').floor('min'); start=pd.Timestamp(session,unit='s',tz='UTC'); states={s:State() for s in symbols}
    if end>start:
        for s,rows in api.pages('bars',symbols,start,end,timeframe='1Min',adjustment='raw').items():
            for b in rows: states[s].event(dict(b,T='b'))
    return states

async def run(a):
    api=Alpaca(a.feed); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    def publish(rows,selected,regime,status):
        atomic(out/'live_ranked.csv',pd.DataFrame(rows).to_csv(index=False) if rows else 'symbol,live_score\n')
        atomic(out/'live_context.json',json.dumps(dict(as_of=pd.Timestamp.now(tz='UTC').isoformat(),selected=selected,regime=regime,status=status,feed=a.feed)))
        atomic(out/'live_universe.txt',', '.join(map(repr,selected)))
    delay=1
    print(f'IEX only: candidate pool={a.pool}, reference symbols={len(REFS)}, subscription budget={a.max_symbols}',flush=True)
    while True:
        try:
            publish([],[],'unknown','warming')
            clock=await asyncio.to_thread(api.get,'/v2/clock',trading=True)
            if not clock['is_open']:
                await asyncio.sleep(30);continue
            day=pd.Timestamp(clock['timestamp']).tz_convert('America/New_York'); session=(day.normalize()+pd.Timedelta(hours=9,minutes=30)).timestamp()
            pool=await asyncio.to_thread(discover_pool,api,a.pool); symbols=list(dict.fromkeys(REFS+pool)); states=await asyncio.to_thread(bootstrap,api,symbols,session)
            async with websockets.connect('wss://stream.data.alpaca.markets/v2/'+a.feed,ping_interval=20,max_queue=1024) as ws:
                await ws.recv(); await ws.send(json.dumps(dict(action='auth',key=api.s.headers['APCA-API-KEY-ID'],secret=api.s.headers['APCA-API-SECRET-KEY'])))
                auth=json.loads(await ws.recv());
                if not any(e.get('msg')=='authenticated' for e in auth):raise RuntimeError('Authentication failed')
                await ws.send(json.dumps(dict(action='subscribe',trades=symbols,quotes=symbols,bars=symbols,updatedBars=symbols)))
                subscribed=False; started=time.time(); next_rank=0; next_clock=0; refresh_task=None; delay=1
                try:
                    while True:
                        try: events=json.loads(await asyncio.wait_for(ws.recv(),timeout=1))
                        except asyncio.TimeoutError: events=[]
                        for e in events:
                            if e['T']=='error':raise RuntimeError(str(e))
                            if e['T']=='subscription':
                                if not set(symbols)<=set(e.get('quotes',[])) or not set(symbols)<=set(e.get('trades',[])):raise RuntimeError('Incomplete subscription')
                                subscribed=True
                            if e.get('S') in states and e['T'] in ('q','t','b','u'):states[e['S']].event(e)
                        now=time.time()
                        if now>=next_clock:
                            clock=await asyncio.to_thread(api.get,'/v2/clock',trading=True); next_clock=now+15
                            if not clock['is_open']:break
                        if now-started>=a.rediscover and refresh_task is None:refresh_task=asyncio.create_task(asyncio.to_thread(discover_pool,api,a.pool))
                        if refresh_task and refresh_task.done():
                            new=refresh_task.result(); refresh_task=None; started=now
                            if set(new)!=set(pool):break # reconnect and backfill new candidates; fail closed during warmup
                        if not subscribed or now<next_rank:continue
                        next_rank=now+a.refresh; ctx={s:st.metrics(now) for s,st in states.items()}; core=[ctx[s] for s in REFS[:4]]
                        if any(x is None for x in core):publish([],[],'unknown','stale');continue
                        bull=sum(x['mid']>x['vwap'] and x['r5']>0 for x in core)>=3; bear=sum(x['mid']<x['vwap'] and x['r5']<0 for x in core)>=3
                        regime='bullish' if bull else 'bearish' if bear else 'mixed'; rows=[]
                        for s in pool:
                            x=ctx[s]
                            if not x or x['spread']>a.max_spread or x['mid']<=x['vwap'] or x['r5']<=0 or x['delta_60']<=0 or x['recent_count']<10:continue
                            ref=INVERSE.get(s) or BULL.get(s); u=ctx.get(ref)
                            if ref and (not u or (s in INVERSE and not (u['mid']<u['vwap'] and u['r5']<0)) or (s in BULL and not (u['mid']>u['vwap'] and u['r5']>0))):continue
                            rows.append(dict(symbol=s,**x,rs=x['r5']-core[0]['r5']))
                        df=pd.DataFrame(rows); selected=[]
                        if rows:
                            rank=lambda k:df[k].rank(pct=True)
                            df['live_score']=.25*rank('rs')+.20*rank('r5')+.20*rank('delta_60')+.15*rank('volume')+.10*(-df.spread).rank(pct=True)+.05*rank('depth')+.05*df.exhaustion.astype(float)
                            df=df.sort_values('live_score',ascending=False); groups=set()
                            for s in df.symbol:
                                g=INVERSE.get(s) or BULL.get(s)
                                if g and g in groups:continue
                                selected.append(s)
                                if g:groups.add(g)
                                if len(selected)>=a.top:break
                        publish(df.to_dict('records') if rows else [],selected,regime,'ready');print(regime,selected,flush=True)
                finally:
                    if refresh_task:refresh_task.cancel()
        except Exception as e:
            print(type(e).__name__,str(e),flush=True);publish([],[],'unknown','error');await asyncio.sleep(delay);delay=min(delay*2,60)
        finally:publish([],[],'unknown','disconnected')

# Internal settings. IEX Basic budget includes benchmark subscriptions.
MAX_SYMBOLS = 30

def main():
    a = SimpleNamespace(feed='iex', max_symbols=MAX_SYMBOLS,
                        pool=MAX_SYMBOLS-len(REFS), top=30,
                        refresh=3, rediscover=300, max_spread=.001,
                        output=str(Path(__file__).resolve().parent / 'market_scan'))
    if a.pool <= 0:
        raise RuntimeError('IEX subscription budget exhausted by references')
    asyncio.run(run(a))

if __name__ == '__main__':
    main()
