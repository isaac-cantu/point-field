#!/usr/bin/env python3
"""Development-only sensitivity; does not tune on official test sets."""
import argparse, sys, time, json, subprocess, os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--out',type=Path,default=Path('runs/sensibilidad_desarrollo'));p.add_argument('--smoke',action='store_true');a=p.parse_args();root=a.repo.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
sys.path[:0]=[str(root/'src'),str(root/'benchmarks')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from pointfield.learning import CDGlobalClustering,CDLocalSpectral,CDDescriptor,per_query_ap,nn_accuracy
from pointfield.objects import make_object_dataset
from common import MAIN_CASES,DEV_SEEDS,VARIANTS,load_case
from benchmark_embeddings_2d import rolled_case
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
rows=[]; erows=[]; orows=[]
def save(name, data):pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False)
def timed(fn):
 t=time.perf_counter();z=fn();return z,time.perf_counter()-t
with threadpool_limits(1):
 for case in MAIN_CASES[:1] if a.smoke else MAIN_CASES:
  for seed in DEV_SEEDS[:1] if a.smoke else DEV_SEEDS:
   for variant in ['base'] if a.smoke else VARIANTS:
    ds=load_case(case,variant,seed)
    for degree in (1,2) if a.smoke else (1,2,3,4):
     for method,k in [('V0',None)]+[('V1',k) for k in ([10] if a.smoke else [5,10,20,30])]:
      r=dict(case=case,seed=seed,variant=variant,method=method,degree=degree,k=k,receives_K=True)
      try:
       model=CDGlobalClustering(ds.n_groups,degree=degree) if method=='V0' else CDLocalSpectral(ds.n_groups,n_neighbors=k,degree=degree)
       lab,dt=timed(lambda:model.fit_predict(ds.X));r.update(ARI=adjusted_rand_score(ds.y,lab),NMI=normalized_mutual_info_score(ds.y,lab),time_s=dt,status='OK')
      except Exception as e:r.update(status='ERROR',error=repr(e))
      rows.append(r)
    save('clustering_development',rows)
   X,Z,y=rolled_case(case,seed)
   for degree in (1,2) if a.smoke else (1,2,3,4):
    for k in ([10] if a.smoke else [5,10,20,30]):
     r=dict(case=case,seed=seed,degree=degree,k=k)
     try:
      E,dt=timed(lambda:CDLocalSpectral(n_neighbors=k,degree=degree).embed(X,2));r.update(T=trustworthiness(Z,E,n_neighbors=10),C=trustworthiness(E,Z,n_neighbors=10),T_input=trustworthiness(X,E,n_neighbors=10),knn_acc=cross_val_score(KNeighborsClassifier(5),E,y,cv=5).mean(),time_s=dt,status='OK')
     except Exception as e:r.update(status='ERROR',error=repr(e))
     erows.append(r)
   save('embedding_development',erows)
 for dim in [2,3]:
  for seed in ([0] if a.smoke else [0,1,2]):
   R,yr,_=make_object_dataset(dim,4 if a.smoke else 20,128 if a.smoke else 512,seed=20260000+2*seed)
   Q,yq,_=make_object_dataset(dim,2 if a.smoke else 10,128 if a.smoke else 512,seed=20260001+2*seed)
   for degree in ([1,2] if a.smoke else [1,2,3,4]):
    for reg in ([.01] if a.smoke else [1e-6,1e-4,.01,.1]):
     r=dict(dim=dim,seed=seed,degree=degree,reg=reg)
     try:
      cd=CDDescriptor(degree=degree,reg=reg);t=time.perf_counter();F=np.array([cd(x) for x in R]);G=np.array([cd(x) for x in Q]);dt=time.perf_counter()-t
      # Same block scaling as retrieval protocol, fitted only on gallery.
      from pointfield.learning import BlockScaler
      scaler=BlockScaler(cd.block_slices(dim)).fit(F);F=scaler.transform(F);G=scaler.transform(G)
      r.update(mAP=float(np.nanmean(per_query_ap(G,yq,F,yr))),nn=nn_accuracy(G,yq,F,yr),dimension=F.shape[1],extract_s=dt,status='OK')
     except Exception as e:r.update(status='ERROR',error=repr(e))
     orows.append(r)
   save('descriptor_development',orows)
for name,data,keys,metric in [('clustering',rows,['method','degree','k'],'ARI'),('embedding',erows,['degree','k'],'T'),('descriptor',orows,['dim','degree','reg'],'mAP')]:
 df=pd.DataFrame(data);v=df[df.status=='OK'];
 if len(v):v.groupby(keys,dropna=False)[metric].agg(['mean','std','count']).to_csv(out/(name+'_summary.csv'))
(out/'metadata.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),smoke=a.smoke,python=sys.version,protocol='development sensitivity only; not a retrospective proof of original parameter selection'),indent=2));print(out)
