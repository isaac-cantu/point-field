"""Controlled timing using the repository's public benchmark factories."""
import argparse, sys, time, json, platform, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import trustworthiness
from threadpoolctl import threadpool_limits
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--repeats',type=int,default=5);p.add_argument('--warmup',type=int,default=1);p.add_argument('--threads',type=int,default=1);a=p.parse_args(); assert a.repeats>0 and a.warmup>=1
repo=a.repo.resolve();sys.path[:0]=[str(repo/'benchmarks'),str(repo/'src')]
from common import METHODS, MAIN_CASES, HELD_OUT_CASES, VARIANTS, load_case
from benchmark_embeddings_2d import EMBEDDERS, CASES, rolled_case, embed
out=repo/'runs'/'tiempos_controlados';out.mkdir(parents=True,exist_ok=True);rows=[]
with threadpool_limits(limits=a.threads):
 for task in ['clustering','embedding']:
  cases=MAIN_CASES+HELD_OUT_CASES if task=='clustering' else CASES
  methods=METHODS if task=='clustering' else EMBEDDERS
  for case in cases:
   for seed in [100,101,102]:
    for variant in VARIANTS if task=='clustering' else ['base']:
     if task=='clustering':
      ds=load_case(case,variant,seed);X,y=ds.X,ds.y;k=len(np.unique(y[y>=0]));
     else:X,Z,y=rolled_case(case,seed)
     for method in methods:
      def run():
       if task=='clustering':return METHODS[method][0](k).fit_predict(X)
       return embed(method,X)
      try:
       for _ in range(a.warmup):run()
       for rep in range(a.repeats):
        t=time.perf_counter();pred=run();dt=time.perf_counter()-t
        q=adjusted_rand_score(y,pred) if task=='clustering' else trustworthiness(Z,pred,n_neighbors=10)
        rows.append(dict(task=task,case=case,seed=seed,variant=variant,method=method,n_points=len(X),repeat=rep,time_s=dt,quality=q,status='OK'))
      except Exception as e:rows.append(dict(task=task,case=case,seed=seed,variant=variant,method=method,status='ERROR',error=str(e)))
     pd.DataFrame(rows).to_csv(out/'runs.csv',index=False)
df=pd.DataFrame(rows);v=df[df.status=='OK'];v.groupby(['task','method']).agg(n=('time_s','size'),median_s=('time_s','median'),p25_s=('time_s',lambda x:x.quantile(.25)),p75_s=('time_s',lambda x:x.quantile(.75)),mean_quality=('quality','mean')).to_csv(out/'summary.csv');(out/'metadata.json').write_text(json.dumps(dict(system=platform.platform(),python=sys.version,threads=a.threads,warmup=a.warmup,repeats=a.repeats,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()),indent=2));print(out)
