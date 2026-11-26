import argparse,os,sys,time,json,resource
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--n',type=int,required=True);p.add_argument('--method',required=True);p.add_argument('--threads',type=int,default=1);a=p.parse_args()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]=str(a.threads)
sys.path.insert(0,str(a.repo/'src'))
import numpy as np
from sklearn.cluster import SpectralClustering
from pointfield.learning import CDLocalSpectral,CDDescriptor,moment_descriptor
from threadpoolctl import threadpool_limits
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from pointfield.geometry.shapes3d import Sphere
# Linux ru_maxrss is KiB. Mac uses bytes.
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform=='darwin' else 1024)
with threadpool_limits(a.threads):
    if a.method in ['spectral','cd_local']:X,y=make_moons(a.n,noise=.04,random_state=100)
    else:X=Sphere(1.).sample(a.n,seed=100)
    baseline=rss();t=time.perf_counter()
    if a.method=='spectral':z=SpectralClustering(2,affinity='nearest_neighbors',n_neighbors=10,random_state=0).fit_predict(X)
    elif a.method=='cd_local':z=CDLocalSpectral(2,n_neighbors=10,degree=1).fit_predict(X)
    elif a.method=='cd_descriptor':z=CDDescriptor()(X)
    elif a.method=='moments':z=moment_descriptor(X)
    else:raise ValueError(a.method)
    elapsed=time.perf_counter()-t;peak=rss()
print(json.dumps(dict(n=a.n,method=a.method,time_s=elapsed,rss_peak_MiB=peak,baseline_highwater_MiB=baseline,highwater_increment_MiB=max(0,peak-baseline),ARI=adjusted_rand_score(y,z) if a.method in ['spectral','cd_local'] else None,status='OK')))
