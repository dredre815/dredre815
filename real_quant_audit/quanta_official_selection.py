#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def daily_rank_ic(x: pd.Series, y: pd.Series) -> float:
    frame=pd.concat([x.rename('x'),y.rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    vals=[]
    for _,g in frame.groupby(level='datetime',sort=False):
        if len(g)>=20 and g.x.nunique()>=3 and g.y.nunique()>=3:
            vals.append(g.x.corr(g.y,method='spearman'))
    return float(np.nanmean(vals)) if vals else float('nan')


def ci(values: np.ndarray) -> tuple[float,float]:
    values=values[np.isfinite(values)]
    m=float(values.mean())
    if len(values)<2:return m,m
    d=float(stats.t.ppf(.975,len(values)-1)*stats.sem(values))
    return m-d,m+d


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--provider',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--trials',type=int,default=1000)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    import qlib
    from qlib.config import REG_CN
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP
    qlib.init(provider_uri=str(args.provider),region=REG_CN)
    h=Alpha158(instruments='csi300',start_time='2016-01-01',end_time='2025-12-26',infer_processors=[],learn_processors=[],fit_start_time='2016-01-01',fit_end_time='2020-12-31')
    data=h.fetch(col_set=['feature','label'],data_key=DataHandlerLP.DK_R)
    f=data['feature'].astype(float); y=data['label'].iloc[:,0].astype(float)
    dates=pd.to_datetime(data.index.get_level_values('datetime'))
    visible=(dates>=pd.Timestamp('2021-01-01'))&(dates<=pd.Timestamp('2021-12-31'))
    sealed=(dates>=pd.Timestamp('2022-01-01'))&(dates<=pd.Timestamp('2025-12-26'))
    if not visible.any() or not sealed.any():
        raise RuntimeError({'min':str(dates.min()),'max':str(dates.max()),'visible':int(visible.sum()),'sealed':int(sealed.sum())})

    rows=[]
    for col in f.columns:
        rows.append({'feature':str(col),'visible_2021_rankic':daily_rank_ic(f.loc[visible,col],y.loc[visible]),'sealed_2022_2025_rankic':daily_rank_ic(f.loc[sealed,col],y.loc[sealed])})
    score=pd.DataFrame(rows).dropna().reset_index(drop=True)
    score.to_csv(args.out/'official_alpha158_scores.csv',index=False)
    rng=np.random.default_rng(20260802)
    budgets=sorted(set(k for k in [1,2,6,18,50,100,len(score)] if 1<=k<=len(score)))
    trials=[]
    for k in budgets:
        for t in range(args.trials):
            idx=rng.choice(len(score),size=k,replace=False)
            subset=score.iloc[idx]
            winner=subset.loc[subset.visible_2021_rankic.idxmax()]
            random_pick=subset.iloc[int(rng.integers(0,len(subset)))]
            trials.append({'candidate_count':k,'trial':t,'winner_feature':winner.feature,'visible_rankic':float(winner.visible_2021_rankic),'sealed_rankic':float(winner.sealed_2022_2025_rankic),'decay':float(winner.visible_2021_rankic-winner.sealed_2022_2025_rankic),'random_sealed_rankic':float(random_pick.sealed_2022_2025_rankic)})
    td=pd.DataFrame(trials); td.to_csv(args.out/'official_selection_trials.csv',index=False)
    out=[]
    for k,p in td.groupby('candidate_count'):
        v=p.visible_rankic.to_numpy(float); s=p.sealed_rankic.to_numpy(float); d=p.decay.to_numpy(float)
        vlo,vhi=ci(v); slo,shi=ci(s); dlo,dhi=ci(d)
        out.append({'candidate_count':int(k),'winner_visible_rankic_mean':float(v.mean()),'visible_ci95_low':vlo,'visible_ci95_high':vhi,'winner_sealed_rankic_mean':float(s.mean()),'sealed_ci95_low':slo,'sealed_ci95_high':shi,'decay_mean':float(d.mean()),'decay_ci95_low':dlo,'decay_ci95_high':dhi,'random_sealed_rankic_mean':float(p.random_sealed_rankic.mean()),'false_promotion_rate_visible_positive_sealed_nonpositive':float(((v>0)&(s<=0)).mean()),'sealed_below_random_rate':float((s<p.random_sealed_rankic.to_numpy(float)).mean())})
    summary=pd.DataFrame(out); summary.to_csv(args.out/'official_selection_summary.csv',index=False)
    meta={'data_min':str(dates.min().date()),'data_max':str(dates.max().date()),'visible':['2021-01-01','2021-12-31'],'sealed':['2022-01-01','2025-12-26'],'visible_rows':int(visible.sum()),'sealed_rows':int(sealed.sum()),'feature_count':len(score),'budgets':budgets,'trials_per_budget':args.trials,'mapping_note':'candidate_count=6 is a legitimate-factor multiple-selection proxy for the shipped six trajectory backtests; it is not a byte-for-byte simulation of QuantaAlpha trajectory mutation/crossover.'}
    (args.out/'official_selection_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(summary.to_string(index=False)); print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
