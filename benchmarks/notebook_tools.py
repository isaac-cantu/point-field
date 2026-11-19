"""Auditable protocol v1 for the thesis notebook. Not a historical replica."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from pointfield.geometry.mesh import TriangleMesh, file_sha256
from pointfield.learning import CDDescriptor, BlockScaler, MultiViewMetric, make_view
from pointfield.learning.invariant import moment_descriptor, moment_block_slices, normalize_similarity, knn_density_weights
from pointfield.learning.evaluation import per_query_ap, nn_accuracy, paired_bootstrap

PROTOCOL = 'verification-v1-20261119'

def stable_seed(seed, identity):
    return int.from_bytes(hashlib.sha256(f'{seed}:{identity}'.encode()).digest()[:4], 'little')

def modelnet_manifest(root, expected_classes):
    """OFF official tree only; fail loudly on unsupported layouts or ambiguity.
    Train wins over test for byte-identical duplicates. Geometric duplicates
    with different serialization are NOT detected by SHA-256.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for p in sorted(root.rglob('*.off')):
        if p.parent.name not in ('train', 'test'):
            continue
        rows.append(dict(path=str(p), object_id=p.relative_to(root).as_posix(),
                         label=p.parent.parent.name, official_split=p.parent.name,
                         sha256=file_sha256(p)))
    if not rows:
        raise ValueError('Expected <root>/<class>/{train,test}/*.off. TXT/HDF5 need an explicit adapter; never infer their splits.')
    df = pd.DataFrame(rows)
    if df.label.nunique() != expected_classes:
        raise ValueError(f'Expected {expected_classes} classes, found {df.label.nunique()}')
    conflicts = df.groupby('sha256').label.nunique()
    mask = df.sha256.isin(conflicts[conflicts > 1].index)
    ambiguous = df.loc[mask].assign(exclusion_reason='conflicting_class_labels')
    df = df.loc[~mask].copy()
    if df.label.nunique() != expected_classes:
        raise ValueError('An entire class was removed by the conflicting-label audit')
    df['priority'] = df.official_split.map({'train':0, 'test':1})
    df = df.sort_values(['priority', 'object_id'])
    rejected = df[df.duplicated('sha256')].copy()
    df = df.drop_duplicates('sha256').drop(columns='priority').reset_index(drop=True)
    for _, g in df.groupby('label'):
        if set(g.official_split) != {'train','test'}:
            raise ValueError('A class lacks a train or test split after deduplication')
    rejected = rejected.drop(columns="priority").assign(exclusion_reason="duplicate_same_label")
    return df, pd.concat([ambiguous, rejected], ignore_index=True)

def freeze_splits(df, seed, per_class=(40,10,20)):
    """New deterministic split. No silent undersampling; no A8/S2 claim."""
    rows = []
    for label, g in df.groupby('label', sort=True):
        rng = np.random.default_rng(stable_seed(seed,label))
        tr = g[g.official_split.eq('train')].sort_values('object_id')
        te = g[g.official_split.eq('test')].sort_values('object_id')
        nr,nv,nt = per_class
        if len(tr)<nr+nv or len(te)<nt:
            raise ValueError(f'{label}: insufficient objects for requested split {per_class}')
        tr=tr.iloc[rng.permutation(len(tr))];te=te.iloc[rng.permutation(len(te))]
        for part, sub in [('reference',tr.iloc[:nr]),('validation',tr.iloc[nr:nr+nv]),('test',te.iloc[:nt])]:
            z=sub.copy();z['partition']=part;rows.append(z)
    out=pd.concat(rows,ignore_index=True)
    assert not out.sha256.duplicated().any()
    out['sampling_seed']=[stable_seed(seed,x) for x in out.object_id]
    return out

def load_partition(manifest, partition, n_points):
    sub=manifest[manifest.partition.eq(partition)]
    X=[TriangleMesh.from_off(p).sample(n_points,seed=int(s)) for p,s in zip(sub.path,sub.sampling_seed)]
    return X,sub.label.to_numpy(),sub.object_id.to_numpy()

def exact_d2(X, bins=32):
    """All pairs; fixed bins + overflow. Avoid stochastic permutation error."""
    Y,*_=normalize_similarity(X)
    edges=np.r_[np.linspace(0,4,bins+1),np.inf]
    h=np.histogram(pdist(Y),edges)[0].astype(float)
    return h/h.sum()

def extract(clouds, kind, angular=True):
    if kind=='D2_exact':return np.array([exact_d2(x) for x in clouds])
    if kind=='PCA_spectrum':
        return np.array([np.linalg.eigvalsh(normalize_similarity(x)[0].T@normalize_similarity(x)[0]/len(x)) for x in clouds])
    if kind=='M_U':return np.array([moment_descriptor(x,angular=angular) for x in clouds])
    if kind in ('CD_U','CD_W'):
        cd=CDDescriptor(angular=angular)
        return np.array([cd(x,None if kind=='CD_U' else knn_density_weights(x)) for x in clouds])
    raise ValueError(kind)

def fit_models(reference, seed=20261119, angular=True, n_views=3):
    """No category labels accepted. Inner scaler fitted only on inner train.
    Refit scaler/metric on all reference identities after selecting eta.
    """
    dim=reference[0].shape[1]; rng=np.random.default_rng(seed)
    views=[[make_view(x,rng,n_sub=min(128,len(x))) for _ in range(n_views)] for x in reference]
    models={}; timing=[]
    for kind in ['CD_U','CD_W','M_U','D2_exact','PCA_spectrum']:
        t=time.perf_counter();F=extract(reference,kind,angular)
        slices=(CDDescriptor(angular=angular).block_slices(dim) if kind.startswith('CD') else
                moment_block_slices(dim,angular=angular) if kind=='M_U' else {'all':slice(0,F.shape[1])})
        scaler=BlockScaler(slices).fit(F)
        models[kind]={'kind':kind,'scaler':scaler,'W':None,'angular':angular,'gallery':scaler.transform(F)}
        if kind in ('CD_U','M_U'):
            raw=np.array([extract(vs,kind,angular) for vs in views])
            n=len(F)
            if n<6:raise ValueError('At least six reference objects required for nested metric validation')
            perm=np.random.default_rng(seed).permutation(n);nh=min(n-2,max(2,round(.2*n)))
            ho,tr=perm[:nh],perm[nh:]
            inner=BlockScaler(slices).fit(F[tr]); V=inner.transform(raw)
            C=MultiViewMetric._cov(V[tr]); scores={}
            for eta in (.1,1.,10.):
                W=MultiViewMetric._whitener(C,eta)
                dist=cdist(V[ho,0]@W,V[ho,1:].mean(1)@W)
                scores[eta]=float((dist.argmin(1)==np.arange(nh)).mean())
            eta=max(scores,key=lambda e:(scores[e],e)) # conservative tie: more regularization
            W=MultiViewMetric._whitener(MultiViewMetric._cov(scaler.transform(raw)),eta)
            models[kind+'_metric']={**models[kind],'W':W,'eta':eta,'inner_identity_scores':scores,
                                    'gallery':scaler.transform(F)@W}
        timing.append({'kind':kind,'fit_seconds':time.perf_counter()-t,'reference_objects':len(F)})
    return models,pd.DataFrame(timing)

def transform_models(models, query):
    raw={k:extract(query,k,m['angular']) for k,m in models.items() if m['W'] is None}
    return {name:(m['scaler'].transform(raw[m['kind']]) if m['W'] is None else
                  m['scaler'].transform(raw[m['kind']])@m['W']) for name,m in models.items()}

def evaluate(models, query, yq, yg, ids, seed=0, n_boot=300, cluster_seeds=(0,1,2)):
    Z=transform_models(models,query);rows=[];individual=[];aps={}
    for name,z in Z.items():
        G=models[name]['gallery'];ap=per_query_ap(z,yq,G,yg)
        if not np.isfinite(ap).all():raise ValueError('Some query classes are absent from gallery')
        aps[name]=ap
        # K is supplied by the experimental design. Labels do not fit the descriptor.
        k=len(np.unique(yq));ari=[];ami=[]
        for s in cluster_seeds:
            lab=KMeans(n_clusters=k,n_init=20,random_state=s).fit_predict(z)
            ari.append(adjusted_rand_score(yq,lab));ami.append(adjusted_mutual_info_score(yq,lab))
        rows.append(dict(method=name,mAP=ap.mean(),nn=nn_accuracy(z,yq,G,yg),ARI=np.mean(ari),
                         ARI_seed_sd=np.std(ari),AMI=np.mean(ami),K_given=k))
        individual.extend(dict(method=name,object_id=i,label=c,AP=float(a)) for i,c,a in zip(ids,yq,ap))
    ci=paired_bootstrap(aps['CD_U_metric'],aps['M_U_metric'],strata=yq,n_boot=n_boot,seed=seed)
    return pd.DataFrame(rows),pd.DataFrame(individual),ci,Z

def perturb_cloud(X, condition, seed, intensity=None):
    rng=np.random.default_rng(seed);Y,c,r=normalize_similarity(X)
    if condition=='clean':return X.copy()
    if condition=='noise':return X+(0.03 if intensity is None else intensity)*r*rng.normal(size=X.shape)
    if condition=='density':
        u=rng.normal(size=X.shape[1]);u/=np.linalg.norm(u)
        p=np.exp((2 if intensity is None else intensity)*np.clip(Y@u,-2,2));p/=p.sum()
        return X[rng.choice(len(X),max(16,len(X)//2),replace=False,p=p)]
    if condition=='subsample':return X[rng.choice(len(X),max(16,len(X)//2),replace=False)]
    if condition=='outliers':
        n=max(1,round((.05 if intensity is None else intensity)*len(X)))
        return np.vstack([X,c+r*rng.uniform(-5,5,(n,X.shape[1]))])
    if condition=='occlusion':
        u=rng.normal(size=X.shape[1]);u/=np.linalg.norm(u);t=Y@u
        return X[t<=np.quantile(t,.8 if intensity is None else 1-intensity)]
    raise ValueError(condition)

def chamfer2(X,Y):
    D=cdist(X,Y,'sqeuclidean');return .5*(D.min(0).mean()+D.min(1).mean())

def radial_field(X, edges):
    Y,*_=normalize_similarity(X);h=np.histogram(np.linalg.norm(Y,axis=1),edges)[0].astype(float)
    return h/h.sum()

def hellinger2(p,q):return .5*np.sum((np.sqrt(p)-np.sqrt(q))**2)

def save_metric_state(models,path):
    state={}
    for name,m in models.items():
        for key,val in [('mean',m['scaler'].mean_),('scale',m['scaler'].scale_),('gallery',m['gallery'])]:state[name+'_'+key]=val
        if m['W'] is not None:state[name+'_W']=m['W']
    np.savez_compressed(path,**state)

def canonical_cloud(X):
    """PCA pose for a diagnostic decoder only; unstable for repeated eigenvalues
    or zero third moments. This is NOT an inverse of an invariant descriptor."""
    Y,*_=normalize_similarity(X)
    ev,U=np.linalg.eigh(Y.T@Y/len(Y));U=U[:,::-1];ev=ev[::-1]
    Z=Y@U;sg=np.sign(np.mean(Z**3,axis=0));sg[sg==0]=1
    return Z*sg, float(np.min(np.abs(np.diff(ev))))

def voxel_targets(clouds,bins=12,extent=3.):
    d=clouds[0].shape[1];edges=[np.linspace(-extent,extent,bins+1)]*d
    fields=[];canonical=[];diagnostics=[]
    for X in clouds:
        Z,gap=canonical_cloud(X);H=np.histogramdd(Z,bins=edges)[0].ravel()
        if H.sum()==0:raise ValueError('Empty voxel window')
        fields.append(H/H.sum());canonical.append(Z)
        diagnostics.append({'pca_gap':gap,'outside_fraction':float(np.mean(np.any(np.abs(Z)>extent,axis=1)))})
    return np.array(fields),canonical,pd.DataFrame(diagnostics)

def decode_fields(G,H,V,Hv,Q,alphas=(.01,.1,1.,10.)):
    """Linear ridge with intercept, alpha selected on separate validation clouds.
    Clip negative predictions, normalize. No labels; all candidates same grid."""
    mu=G.mean(0);mh=H.mean(0);A=G-mu;B=H-mh
    fits={};scores={}
    def predict(Z,W):
        out=np.maximum((Z-mu)@W+mh,0);den=out.sum(1)
        out[den<=0]=mh;return out/out.sum(1,keepdims=True)
    for a in alphas:
        W=np.linalg.solve(A.T@A+a*np.eye(A.shape[1]),A.T@B)
        fits[a]=W;P=predict(V,W)
        scores[a]=float(np.mean([hellinger2(p,h) for p,h in zip(P,Hv)]))
    alpha=min(scores,key=lambda a:(scores[a],-a))
    return predict(Q,fits[alpha]),alpha,scores

def sample_voxels(p,dim,bins=12,extent=3.,n=256,seed=0):
    rng=np.random.default_rng(seed);flat=rng.choice(len(p),n,p=p)
    cells=np.array(np.unravel_index(flat,(bins,)*dim)).T
    return -extent+(cells+rng.uniform(size=(n,dim)))*(2*extent/bins)
