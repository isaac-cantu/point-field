"""Frozen descriptors + supervised linear probe, official train/test splits.
Run from point-field: python /path/11_modelnet_svm.py --root data/ModelNet10 --classes 10
"""
import argparse, sys, time, json, platform, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from threadpoolctl import threadpool_limits

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--classes',type=int,choices=[10,40],required=True); p.add_argument('--repo',type=Path,default=Path.cwd()); p.add_argument('--points',type=int,default=512); p.add_argument('--train-per-class',type=int,default=0,help='0: all available'); p.add_argument('--test-per-class',type=int,default=0); p.add_argument('--seed',type=int,default=20261025); a=p.parse_args()
    repo=a.repo.resolve(); sys.path.insert(0,str(repo/'benchmarks')); sys.path.insert(0,str(repo/'src'))
    from modelnet10 import index,load
    from objetos_invariantes import features
    from pointfield.geometry.mesh import file_sha256
    root=a.root.resolve(); out=repo/'runs'/'linear_probe'/f'ModelNet{a.classes}_seed{a.seed}_p{a.points}_tr{a.train_per_class}_te{a.test_per_class}'; out.mkdir(parents=True,exist_ok=True)
    train,classes=index(root,'train'); test,_=index(root,'test'); assert len(classes)==a.classes,classes
    hs={file_sha256(f) for f,c in train}; test=[(f,c) for f,c in test if file_sha256(f) not in hs]
    rng=np.random.default_rng(a.seed)
    def subset(items,cap):
        if not cap:return items
        return [pool[i] for c in classes for pool in [[it for it in items if it[1]==c]] for i in rng.choice(len(pool),min(cap,len(pool)),replace=False)]
    train=subset(train,a.train_per_class); test=subset(test,a.test_per_class)
    assert set(c for _,c in train)==set(classes)==set(c for _,c in test)
    pd.DataFrame([(sp,str(f.relative_to(root)),c,file_sha256(f)) for sp,items in [('train',train),('test',test)] for f,c in items],columns=['split','path','class','sha256']).to_csv(out/'manifest.csv',index=False)
    with threadpool_limits(limits=1):
        R,yr=load(train,classes,a.points,rng); Q,yq=load(test,classes,a.points,rng)
        # features returns descriptor arrays and slice definitions; see repository implementation.
        t=time.perf_counter(); fr=features(R,3); train_extract=time.perf_counter()-t
        t=time.perf_counter(); fq=features(Q,3); test_extract=time.perf_counter()-t
        if isinstance(fr,tuple):fr=fr[0];fq=fq[0]
        rows=[]; cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=a.seed)
        for method in ('CD_U','CD_W','M_U','D2'):
            X,Z=fr[method],fq[method]; estimator=make_pipeline(StandardScaler(),LinearSVC(dual='auto',max_iter=20000,random_state=a.seed)); search=GridSearchCV(estimator,{'linearsvc__C':[.001,.01,.1,1,10,100]},cv=cv,scoring='accuracy',n_jobs=1)
            t=time.perf_counter();search.fit(X,yr); fit_s=time.perf_counter()-t
            t=time.perf_counter();pred=search.predict(Z);pred_s=time.perf_counter()-t
            rows.append(dict(method=method,dimension=X.shape[1],train=len(yr),test=len(yq),C=search.best_params_['linearsvc__C'],cv_accuracy=search.best_score_,accuracy=accuracy_score(yq,pred),macro_f1=f1_score(yq,pred,average='macro'),search_and_fit_s=fit_s,predict_s=pred_s))
            pd.DataFrame({'path':[str(f.relative_to(root)) for f,c in test],'true':yq.tolist(),'prediction':pred}).to_csv(out/f'{method}_predictions.csv',index=False)
            pd.DataFrame(confusion_matrix(yq,pred,labels=range(len(classes))),index=classes,columns=classes).to_csv(out/f'{method}_confusion.csv')
        pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(dict(seed=a.seed,points=a.points,system=platform.platform(),python=sys.version,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),all_descriptors_train_s=train_extract,all_descriptors_test_s=test_extract,threads=1,pose='random similarity including reflection; different from typical canonical DNN protocols'),indent=2))
    print(pd.DataFrame(rows).to_string(index=False));print(out)
if __name__=='__main__':main()
