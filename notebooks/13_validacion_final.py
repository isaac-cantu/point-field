#!/usr/bin/env python3
"""Final experimental validation. Run from point-field; never overwrite archived results."""
import argparse, hashlib, json, os, platform, subprocess, sys, time, traceback, shutil, re, tarfile, io, warnings
from pathlib import Path
from datetime import datetime, timezone
from math import comb

P=argparse.ArgumentParser(description=__doc__)
P.add_argument('--repo',type=Path,default=Path.cwd())
P.add_argument('--mode',choices=['smoke','full'],default='full')
P.add_argument('--modelnet10',type=Path); P.add_argument('--modelnet40',type=Path)
P.add_argument('--flare-notebook',type=Path,help='Notebook FLARE configured with absolute dataset path')
P.add_argument('--blocks',nargs='+',default=['all'],choices=['all','archive','math','support','clustering','embedding','objects','modelnet','notebooks','flare','memory','tests'])
P.add_argument('--threads',type=int,default=1); P.add_argument('--repeats',type=int,default=5)
P.add_argument('--out',type=Path); P.add_argument('--resume',action='store_true')
P.add_argument('--accept-test',action='store_true',help='Confirm frozen protocol before rerunning official test')
A=P.parse_args(); ROOT=A.repo.resolve(); FULL=A.mode=='full'
assert (ROOT/'src/pointfield').is_dir(), 'Use --repo pointing to point-field'
assert A.threads>=1 and A.repeats>=1
OUT=(A.out or ROOT/'runs'/('validacion_final_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
if OUT.exists() and not A.resume: raise SystemExit('Output exists; choose another --out or --resume')
OUT.mkdir(parents=True,exist_ok=True)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']: os.environ[k]=str(A.threads)
os.environ['MPLBACKEND']='Agg'
import numpy as np, pandas as pd
from scipy.linalg import solve
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, silhouette_score, accuracy_score, f1_score
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits, threadpool_info
import matplotlib.pyplot as plt
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'benchmarks')]
from pointfield.learning import ChristoffelDarboux, CDDescriptor, moment_descriptor
from pointfield.objects import make_object_dataset
LIMIT=threadpool_limits(A.threads)
STATUS=OUT/'status.csv'; records=[]
if A.resume and STATUS.exists(): records=pd.read_csv(STATUS).fillna('').to_dict('records')
def record(block,item,status,detail=''):
    records.append(dict(block=block,item=item,status=status,detail=str(detail)))
    pd.DataFrame(records).to_csv(STATUS,index=False); print(block,item,status,detail,flush=True)
def check(block,item,ok,detail=''): record(block,item,'PASS' if bool(ok) else 'FAIL',detail)
def save(name,rows):
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows); df.to_csv(OUT/(name+'.csv'),index=False); return df
def run(cmd,log,cwd=ROOT):
    with (OUT/(log+'.log')).open('w') as f:
        env=os.environ.copy();env['PYTHONPATH']=str(Path(cwd)/'src')+os.pathsep+str(Path(cwd)/'benchmarks')+os.pathsep+env.get('PYTHONPATH','');r=subprocess.run([str(x) for x in cmd],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,env=env)
    if r.returncode: raise RuntimeError(f'Exit {r.returncode}; see {log}.log')
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
commit=git('rev-parse','HEAD'); dirty=git('status','--porcelain','--untracked-files=no')
meta=dict(commit=commit,tracked_changes=dirty,mode=A.mode,python=sys.version,platform=platform.platform(),cpu=platform.processor(),threads=A.threads,repeats=A.repeats,threadpools=threadpool_info(),modelnet10=str(A.modelnet10),modelnet40=str(A.modelnet40),flare_notebook=str(A.flare_notebook),accept_test=A.accept_test)
import importlib.metadata as im
meta['versions']={}
for p in ['numpy','scipy','pandas','scikit-learn','umap-learn','point-field','psutil','nbclient','pytest']:
    try: meta['versions'][p]=im.version(p)
    except im.PackageNotFoundError: meta['versions'][p]='not installed'
if A.resume and (OUT/'metadata.json').exists():
    old=json.loads((OUT/'metadata.json').read_text()); assert all(old.get(k)==meta.get(k) for k in ['commit','tracked_changes','mode','threads','repeats','modelnet10','modelnet40','flare_notebook','accept_test']), 'Resume configuration changed'
(OUT/'metadata.json').write_text(json.dumps(meta,indent=2,default=str)); (OUT/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)); (OUT/'tracked.diff').write_text(git('diff','HEAD'))
def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def archive():
    b='archive'; paths=['benchmarks/results/cd_2d_degree_diagnostics.csv','benchmarks/results/clustering_2d_eval.csv','benchmarks/results/embeddings_2d_eval.csv','benchmarks/results/objetos_2d.csv','benchmarks/results/objetos_3d.csv','benchmarks/results/modelnet10_test.csv','benchmarks/results/modelnet40_test.csv','benchmarks/results/aplicacion_FLARE_nivel1.csv','benchmarks/results/aplicacion_FLARE_nivel2.csv','benchmarks/results/aplicacion_FLARE_contrastes.csv','benchmarks/results/aplicacion_FLARE_robustez.csv','benchmarks/results/aplicacion_FLARE_envolventes.csv','benchmarks/results/aplicacion_FLARE_agrupamiento_formas.csv','runs/20261119T210929036674Z/scenes_3d.csv','runs/20261119T210929036674Z/reconstruction_per_object.csv','runs/tesis_figuras/09_metricas_soporte_annulus.csv','runs/tesis_figuras/09_diagnostico_annulus.csv','runs/auditoria_final/checks.csv','runs/tiempos_controlados/runs.csv','runs/tiempos_controlados/summary.csv']
    manifest=[]
    for rel in paths:
        p=ROOT/rel
        if not p.exists():record(b,rel,'MISSING');continue
        d=pd.read_csv(p); manifest.append(dict(path=rel,sha256=hashfile(p),rows=len(d),columns='|'.join(d.columns)));check(b,rel,len(d)>0,f'{len(d)} rows; existence only')
    save('archive_manifest',manifest)
    p=ROOT/'runs/auditoria_final/checks.csv'
    if p.exists():
        d=pd.read_csv(p);check(b,'previous audit',len(d)==39 and d.status.eq('PASS').all(),f'{len(d)} checks')
    for n in [10,40]:
        d=pd.read_csv(ROOT/f'benchmarks/results/modelnet{n}_test.csv'); models=d[d.nn.notna()];check(b,f'MN{n} retrieval',len(models)==35 and not models.duplicated(['condition','method']).any() and models.mAP.between(0,1).all() and models.nn.between(0,1).all())
        for _,r in d[d.nn.isna()].iterrows():
            x,y=r.method.split(' - ');g=models[models.condition==r.condition].set_index('method');check(b,f'MN{n} {r.condition} {r.method}',abs(g.loc[x,'mAP']-g.loc[y,'mAP']-r.mAP)<1e-9)
        folder=ROOT/f'runs/linear_probe/ModelNet{n}_seed20261025_p512_tr0_te0'
        if not folder.exists():record(b,f'MN{n} SVM','MISSING');continue
        preds=[]
        for m in ['CD_U','CD_W','M_U','D2']:
            p=folder/f'{m}_predictions.csv';d=pd.read_csv(p);y,z=d.iloc[:,-2],d.iloc[:,-1];preds.append(dict(method=m,n=len(y),accuracy=accuracy_score(y,z),f1_macro=f1_score(y,z,average='macro')))
        dnew=save(f'MN{n}_svm_recomputed',preds); old=pd.read_csv(folder/'results.csv'); comparison=dnew.merge(old,on='method',suffixes=('_new','_old'),validate='one_to_one');check(b,f'MN{n} SVM prediction metrics',len(comparison)==4 and np.allclose(comparison.accuracy_new,comparison.accuracy_old,atol=1e-12,rtol=0) and np.allclose(comparison.f1_macro,comparison.macro_f1,atol=1e-12,rtol=0) and (comparison.n==comparison.test).all())
        m=pd.read_csv(folder/'manifest.csv'); h=next((c for c in m if 'sha' in c.lower() or 'hash' in c.lower()),None)
        check(b,f'MN{n} manifest unique hashes',h is not None and m[h].notna().all() and m[h].is_unique,f'{len(m)} rows')
        record(b,f'MN{n} prediction metrics','INFO','Recomputed from individual predictions; see CSV; compare with results.csv')
    d=pd.read_csv(ROOT/'runs/tesis_figuras/09_metricas_soporte_annulus.csv');check(b,'annulus IoU counts',len(d)==64 and np.allclose(d.IoU,d.TP/(d.TP+d.FP+d.FN)))
    f=pd.read_csv(ROOT/'benchmarks/results/aplicacion_FLARE_nivel1.csv');check(b,'FLARE scene summary',len(f)==36 and f.indice_muestra.nunique()==6 and f['órganos'].eq(13).all() and f.puntos.eq(5200).all() and f.ARI.between(-1,1).all());f.groupby('método')[['ARI','NMI','homogeneidad','completitud','tiempo (s)']].mean().to_csv(OUT/'flare_scene_archive_summary.csv')
    f=pd.read_csv(ROOT/'benchmarks/results/aplicacion_FLARE_nivel2.csv');check(b,'FLARE retrieval summary',len(f)==5 and f.descriptor.is_unique and f.mAP.between(0,1).all() and f['1-NN'].between(0,1).all());record(b,'FLARE individual predictions','INFO','Aggregated archive only; cannot independently recover AP or verify patient split from this CSV')
    d=pd.read_csv(ROOT/'runs/20261119T210929036674Z/reconstruction_per_object.csv');check(b,'decoder archived per-object records',len(d)==234 and not d.duplicated(['dataset','method','object_id']).any() and d.Chamfer2.ge(0).all() and d.Hellinger2.between(0,1+1e-12).all());d.groupby(['dataset','method'])[['Chamfer2','Hellinger2']].agg(['mean','std','count']).to_csv(OUT/'decoder_archive_summary.csv')
    d=pd.read_csv(ROOT/'runs/20261119T210929036674Z/scenes_3d.csv');check(b,'3D archived scene metrics',len(d)==6 and d.ARI.between(-1,1).all());record(b,'3D archive scope','INFO','Only six rows in this historical file; not evidence of a full multi-seed experiment')

def math_checks():
    rows=[];rng=np.random.default_rng(20261125)
    def err(name,a,b,tol=1e-8,**kw):
        a,b=np.asarray(a),np.asarray(b);e=float(np.linalg.norm(a-b)/max(1.,np.linalg.norm(b))); rows.append(dict(property=name,error=e,tolerance=tol,**kw));check('math',name,e<=tol,f'{e:.3e}')
    for dim in [2,3]:
      for degree in [1,2,3,4]:
        X=rng.uniform(-1,1,(500,dim));Z=rng.uniform(-1,1,(30,dim)); cd=ChristoffelDarboux(degree,reg=0).fit(X);V=cd.design(X);coef=rng.normal(size=V.shape[1]);K=cd.kernel(Z,X)
        kw=dict(dim=dim,degree=degree);err('reproduction',K@(V@coef)/len(X),cd.design(Z)@coef,**kw);err('trace',cd.score(X).mean(),comb(dim+degree,dim),**kw)
        err('kernel_diagonal',np.diag(cd.kernel(Z)),cd.score(Z),**kw);err('reciprocal',cd.christoffel(Z)*cd.score(Z),np.ones(len(Z)),**kw)
        check('math','kernel PSD',np.linalg.eigvalsh(cd.kernel(Z)).min()>=-1e-8,str(kw))
        T=np.eye(dim)+.15*rng.normal(size=(dim,dim));t=rng.uniform(-.2,.2,dim);err('affine_unregularized',ChristoffelDarboux(degree,reg=0).fit(X@T+t).score(Z@T+t),cd.score(Z),**kw)
        Q=np.linalg.qr(rng.normal(size=(dim,dim)))[0];Q[:,0]*=np.linalg.det(Q) # proper rotation
        w=ChristoffelDarboux(degree,reg=1e-5,basis='weighted').fit(X)
        for kind in ['rotation','reflection']:
            if kind=='reflection':Q[:,0]*=-1
            wr=ChristoffelDarboux(degree,reg=1e-5,basis='weighted').fit(X@Q);err(kind+'_regularized',wr.kernel(Z@Q),w.kernel(Z),**kw)
        err('regularized_trace',w.score(X).mean(),np.trace(solve(w.M_reg_,w.moment_matrix_,assume_a='pos')),**kw)
        a=solve(w.M_reg_,w.design(Z[:1])[0],assume_a='pos'); a/=w.design(Z[:1])[0]@a;err('variational',a@w.M_reg_@a,w.christoffel(Z[:1])[0],**kw)
        if degree==1:
            D=Z-X.mean(0);err('Mahalanobis',cd.score(Z),1+np.einsum('ij,ji->i',D,solve(np.cov(X.T,bias=True),D.T)),**kw)
      clouds,_,_=make_object_dataset(dim,2,300,seed=100)
      for i,X in enumerate(clouds):
        Q=np.linalg.qr(rng.normal(size=(dim,dim)))[0];Y=(2.3*X@Q+3)[rng.permutation(len(X))];err('descriptor_similarity',CDDescriptor()(Y),CDDescriptor()(X),tol=1e-7,dim=dim,object=i);err('descriptor_permutation',CDDescriptor()(X[rng.permutation(len(X))]),CDDescriptor()(X),tol=1e-7,dim=dim,object=i);err('moments_similarity',moment_descriptor(Y),moment_descriptor(X),tol=1e-7,dim=dim,object=i)
    save('math_checks',rows)

def support():
    # exact volumetric 2D supports; independent uniform query points; nested training draws
    rng=np.random.default_rng(999);U=rng.uniform(-1.35,1.35,(30000 if FULL else 3000,2));r=np.linalg.norm(U,axis=1)
    def inside(X,name):
        x,y=X.T;rr=x*x+y*y
        if name=='annulus':return (rr>=.25)&(rr<=1)
        if name=='disk':return rr<=1
        if name=='rectangle':return (abs(x)<=.9)&(abs(y)<=.55)
        if name=='cross':return ((abs(x)<=.3)&(abs(y)<=1))|((abs(x)<=1)&(abs(y)<=.3))
        return ((x-.6)**2+y*y<=.35**2)|((x+.6)**2+y*y<=.35**2)
    def sample(name,n,seed):
        g=np.random.default_rng(seed); chunks=[]; count=0
        while count<n:
            X=g.uniform(-1,1,(max(n,1000),2));X=X[inside(X,name)];chunks.append(X);count+=len(X)
        return np.concatenate(chunks)[:n]
    deg=[1,2,4,6,8,12,16,32] if FULL else [1,2,4]; ns=[100,200,500,1000,2000,4000,8000,16000] if FULL else [100,500]; seeds=range(100,105) if FULL else [100]; rows=[]
    # High degrees only for annulus; other supports through degree 16 to bound workload.
    for name in ['annulus','disk','rectangle','cross','two_disks'] if FULL else ['annulus']:
      truth=inside(U,name)
      for seed in seeds:
        Xall=sample(name,max(ns),seed)
        for n in ns:
          X=Xall[:n]
          for d in deg if name=='annulus' else [v for v in deg if v<=16]:
            try:
                t=time.perf_counter();cd=ChristoffelDarboux(d,reg=1e-8).fit(X);fit=time.perf_counter()-t;thr=np.quantile(cd.score(X),.95);t=time.perf_counter();pred=np.concatenate([cd.score(U[i:i+2048]) for i in range(0,len(U),2048)])<=thr;elapsed=time.perf_counter()-t
                tp=np.sum(pred&truth);fp=np.sum(pred&~truth);fn=np.sum(~pred&truth);tn=np.sum(~pred&~truth)
                rows.append(dict(shape=name,seed=seed,n_points=n,degree=d,basis=cd.basis_size_,IoU=tp/(tp+fp+fn),precision=tp/max(tp+fp,1),recall=tp/(tp+fn),FPR=fp/(fp+tn),TP=tp,FP=fp,FN=fn,TN=tn,threshold=thr,cond_M=cd.condition_number_,cond_regularized=np.linalg.cond(cd.M_reg_),effective_dof=cd.effective_dof_,fit_s=fit,eval_s=elapsed,status='OK'))
            except Exception as e:rows.append(dict(shape=name,seed=seed,n_points=n,degree=d,status='ERROR',error=str(e)))
          save('support_runs',rows)
    df=save('support_runs',rows);check('support','all configurations',df.status.eq('OK').all(),len(df));v=df[df.status=='OK'];v.groupby(['shape','degree','n_points']).IoU.agg(['mean','std','count']).to_csv(OUT/'support_summary.csv')
    for name,g in v.groupby('shape'):
        h=g.pivot_table(index='degree',columns='n_points',values='IoU');fig,ax=plt.subplots(figsize=(9,5));im=ax.imshow(h,vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(len(h.columns)),h.columns);ax.set_yticks(range(len(h.index)),h.index);ax.set(xlabel='N',ylabel='Grado',title=name+' — IoU medio');fig.colorbar(im);fig.tight_layout();fig.savefig(OUT/f'support_{name}.pdf');plt.close(fig)

def benchmark(task):
    from common import METHODS, MAIN_CASES, HELD_OUT_CASES, VARIANTS, load_case
    from benchmark_embeddings_2d import EMBEDDERS, CASES, rolled_case, embed
    import psutil, threading
    proc=psutil.Process();rows=[];seeds=range(100,105) if FULL else [100];reps=A.repeats if FULL else 1
    def measure(fn):
        base=proc.memory_info().rss; peak=[base];stop=threading.Event()
        def poll():
            while not stop.wait(.002):peak[0]=max(peak[0],proc.memory_info().rss)
        th=threading.Thread(target=poll);th.start();t=time.perf_counter()
        try:result=fn();dt=time.perf_counter()-t
        finally:stop.set();th.join()
        return result,dt,base,max(peak[0],proc.memory_info().rss)
    for case in (MAIN_CASES+HELD_OUT_CASES if task=='clustering' else CASES):
      for seed in seeds:
        for variant in (VARIANTS if task=='clustering' else ['base']):
          if task=='clustering':ds=load_case(case,variant,seed);X,y,k=ds.X,ds.y,ds.n_groups
          else:X,Z,y=rolled_case(case,seed)
          for method in (METHODS if task=='clustering' else EMBEDDERS):
            key=dict(case=case,seed=seed,variant=variant,method=method,n_points=len(X));fn=(lambda:METHODS[method][0](k).fit_predict(X)) if task=='clustering' else (lambda:embed(method,X))
            try:
                fn() # warmup, not measured
                for rep in range(reps):
                    z,dt,base,peak=measure(fn);r=dict(**key,repeat=rep,time_s=dt,rss_baseline_MiB=base/2**20,rss_peak_sampled_MiB=peak/2**20,status='OK')
                    if task=='clustering':
                        mask=z!=-1;labels=z[mask];r.update(receives_k=METHODS[method][1],ARI=adjusted_rand_score(y,z),NMI=normalized_mutual_info_score(y,z),homogeneity=homogeneity_score(y,z),completeness=completeness_score(y,z),V_measure=v_measure_score(y,z),noise_frac=1-mask.mean(),n_found=len(set(labels)),silhouette=silhouette_score(X[mask],labels) if 1<len(set(labels))<len(labels) else np.nan)
                    else:r.update(T=trustworthiness(Z,z,n_neighbors=10),C=trustworthiness(z,Z,n_neighbors=10),T_input=trustworthiness(X,z,n_neighbors=10),kNN_acc=cross_val_score(KNeighborsClassifier(5),z,y,cv=5).mean())
                    rows.append(r)
            except Exception as e:rows.append(dict(**key,status='ERROR',error=str(e)))
          save(task+'_runs',rows)
    df=save(task+'_runs',rows);check(task,'all executions',df.status.eq('OK').all(),len(df));v=df[df.status=='OK'];quality='ARI' if task=='clustering' else 'T';v.groupby('method').agg(n=('time_s','size'),median_s=('time_s','median'),p25_s=('time_s',lambda s:s.quantile(.25)),p75_s=('time_s',lambda s:s.quantile(.75)),quality_mean=(quality,'mean'),peak_sampled_MiB=('rss_peak_sampled_MiB','max')).to_csv(OUT/(task+'_summary.csv'))
    # Repeated runs are timing repeats, not independent geometric realizations.
    first=v[v.repeat==0];old=pd.read_csv(ROOT/f'benchmarks/results/{"clustering_2d" if task=="clustering" else "embeddings_2d"}_eval.csv');keys=['case','seed','method']+(['variant'] if task=='clustering' else []);cols=['ARI','NMI'] if task=='clustering' else ['T','C','T_input','kNN_acc'];cmp=first.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one');
    for c in cols:cmp[c+'_delta']=cmp[c+'_new']-cmp[c+'_old']
    save(task+'_archive_comparison',cmp);record(task,'archive metric comparison','INFO',f'{len(cmp)} matched; inspect deltas, not forced equality across environments')

def objects():
    for d in [2,3]:
        cmd=[sys.executable,ROOT/'benchmarks/objetos_invariantes.py','--dim',d,'--out',OUT/f'objects_{d}d.csv']
        if not FULL:cmd+=['--n-ref',4,'--n-query',2,'--points',96,'--n-boot',100]
        run(cmd,f'objects_{d}d');record('objects',str(d),'PASS','script completed; compare protocol and metrics')

def modelnet():
    if not A.accept_test:record('modelnet','official test','SKIP','--accept-test required; freeze all decisions first');return
    for n,path in [(10,A.modelnet10),(40,A.modelnet40)]:
        if path is None or not path.is_dir():record('modelnet',str(n),'MISSING','dataset path required');continue
        # Official script writes some auxiliary files in its working tree; use isolated tracked snapshot.
        w=snapshot();root=path.resolve();cmd=[sys.executable,w/'benchmarks/modelnet10.py','--root',root,'--split','test','--n-ref',400,'--n-dev',100,'--n-test',200 if n==10 else 400,'--out',OUT/f'modelnet{n}_retrieval.csv']
        if not FULL:cmd+=['--n-ref',40,'--n-dev',20,'--n-test',40,'--points',96,'--n-boot',100]
        try:run(cmd,f'modelnet{n}_retrieval',w);record('modelnet',f'MN{n} retrieval','PASS','new run; protocol may differ from archived')
        except Exception as e:record('modelnet',f'MN{n} retrieval','ERROR',e)
        try:
            cmd=[sys.executable,w/'notebooks/11_modelnet_svm.py','--root',root,'--classes',n,'--repo',w]
            if not FULL:cmd+=['--train-per-class',10,'--test-per-class',5,'--points',96]
            run(cmd,f'modelnet{n}_svm',w);record('modelnet',f'MN{n} SVM','PASS','outputs in snapshot/runs/linear_probe')
        except Exception as e:record('modelnet',f'MN{n} SVM','ERROR',e)

def snapshot():
    w=OUT/'snapshot'
    if not w.exists():
        subprocess.run(['git','worktree','add','--detach',str(w),commit],cwd=ROOT,check=True)
        if (ROOT/'data').exists() and not (w/'data').exists(): (w/'data').symlink_to(ROOT/'data',target_is_directory=True)
        if dirty: record('snapshot','tracked local changes','INFO','Snapshot uses committed source; other blocks use working tree. Do not compare as identical if source differs.')
    return w

def execute_nb(path,name,replace=None):
    import nbformat
    from nbclient import NotebookClient
    w=snapshot();nb=nbformat.read(path,as_version=4)
    if replace:
        for c in nb.cells:
            if c.cell_type=='code':c.source=replace(c.source)
    target=OUT/(name+'.ipynb');nbformat.write(nb,target)
    # Use kernel installed for this Python environment; ensure user invokes ipykernel install as documented.
    try:NotebookClient(nb,timeout=None,kernel_name='pointfield-cierre',resources={'metadata':{'path':str(w)}}).execute()
    finally:nbformat.write(nb,target)
    record('notebooks',name,'PASS','all cells executed; inspect generated numerical results')
def notebooks():
    w=snapshot()
    def cfg(s):
        s=re.sub(r"(?m)^MODE\s*=\s*['\"]rapido['\"]", "MODE='amplio'" if FULL else "MODE='rapido'",s);return s
    try:execute_nb(w/'notebooks/CD_desde_cero.ipynb','CD_desde_cero_reexecuted',cfg)
    except Exception as e:record('notebooks','CD_desde_cero','ERROR',e)
    # Historical notebook exercises 3D contact and independent reconstruction; preserve smoke/full distinction.
    def cfg2(s):
        s=re.sub(r"(?m)^MODE\s*=\s*['\"]smoke['\"]", "MODE='full'" if FULL else "MODE='smoke'",s);s=re.sub(r'(?m)^RUN_MODELNET\s*=\s*True','RUN_MODELNET = False',s);return s
    execute_nb(w/'notebooks/verificacion_tesis_CD_corregida.ipynb','verificacion_reexecuted',cfg2)
def flare():
    if not A.flare_notebook or not A.flare_notebook.is_file():record('flare','rerun','MISSING','provide configured notebook with absolute dataset path');return
    execute_nb(A.flare_notebook.resolve(),'FLARE_reexecuted');record('flare','rerun','PASS','Notebook completed; verify its printed dataset counts and CSV before scientific acceptance');record('flare','scope','INFO','Synthetic multi-organ scenes and transductive shape retrieval; no clinical or patient-level claim')
def memory():
    # Separate processes: ru_maxrss is lifetime RSS high-water mark (Linux KiB), including imports.
    worker=Path(__file__).with_name('14_memoria_worker.py');rows=[]
    for n in ([200,400,800,1600,3200] if FULL else [200,400]):
        for method in ['spectral','cd_local','cd_descriptor','moments']:
            try:
                cmd=[sys.executable,str(worker),'--repo',str(ROOT),'--n',str(n),'--method',method,'--threads',str(A.threads)];r=subprocess.run(cmd,capture_output=True,text=True,check=True);rows.append(json.loads(r.stdout.strip().splitlines()[-1]))
            except Exception as e:rows.append(dict(n=n,method=method,status='ERROR',error=str(e)))
            save('memory_scaling',rows)
    check('memory','all workers',all(r['status']=='OK' for r in rows),len(rows))
def tests():run([sys.executable,'-m','pytest','-q',ROOT/'tests'],'pytest');record('tests','pytest','PASS','see pytest.log')
blocks={'archive':archive,'math':math_checks,'support':support,'clustering':lambda:benchmark('clustering'),'embedding':lambda:benchmark('embedding'),'objects':objects,'modelnet':modelnet,'notebooks':notebooks,'flare':flare,'memory':memory,'tests':tests}
for b,f in blocks.items():
    if 'all' not in A.blocks and b not in A.blocks:continue
    if A.resume and any(r['block']==b and r['item']=='BLOCK_END' and r['status']=='PASS' for r in records):continue
    start=len(records)
    try:
        f();bad=any(r['status'] in ['FAIL','ERROR','MISSING','SKIP'] for r in records[start:]);record(b,'BLOCK_END','INCOMPLETE' if bad else 'PASS')
    except Exception as e:
        (OUT/(b+'_error.log')).write_text(traceback.format_exc());record(b,'BLOCK_END','ERROR',e)
print('Results:',OUT);table=pd.DataFrame(records);print(table.groupby(['block','status']).size().to_string()); latest=table[table.item=='BLOCK_END'].drop_duplicates('block',keep='last');latest.to_csv(OUT/'block_summary.csv',index=False);sys.exit(0 if len(latest) and latest.status.eq('PASS').all() else 1)
