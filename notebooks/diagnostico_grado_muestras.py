"""Diagnóstico complementario independiente, no réplica del benchmark histórico.
Ejecutar desde la raíz: python scripts/diagnostico_grado_muestras.py
Dependencias: numpy, scipy, matplotlib, pandas.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import itertools
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'/'experimentos'; OUT.mkdir(parents=True,exist_ok=True)
RES=ROOT/'resultados'; RES.mkdir(exist_ok=True)
N_VALUES=[50,150,500]; DEGREES=[1,2,4,6]; SEEDS=[100,101,102,103,104]
SHAPES=['Disco','Anillo','Elipse','Rectángulo','Dos discos']
def inside(x,shape):
    a,b=x.T; r=a*a+b*b
    if shape=='Disco': return r<=0.85**2
    if shape=='Anillo': return (r<=0.9**2)&(r>=0.48**2)
    if shape=='Elipse': return (a/1.05)**2+(b/0.5)**2<=1
    if shape=='Rectángulo': return (abs(a)<=0.9)&(abs(b)<=0.55)
    return ((a-.65)**2+b*b<=.47**2)|((a+.65)**2+b*b<=.47**2)
def sample(rng,shape,n):
    chunks=[]; total=0
    while total<n:
        x=rng.uniform(-1.4,1.4,size=(max(256,3*(n-total)),2)); x=x[inside(x,shape)]
        chunks.append(x); total+=len(x)
    return np.concatenate(chunks)[:n]
def basis(x,n):
    exps=[(i,k-i) for k in range(n+1) for i in range(k,-1,-1)]
    return np.stack([x[:,0]**i*x[:,1]**j for i,j in exps],axis=1)
def fit(X,n,c=1e-8):
    center=X.mean(axis=0); rms=np.sqrt(np.mean(np.sum((X-center)**2,axis=1)))
    V=basis((X-center)/rms,n); M=V.T@V/len(X); A=M+c*np.trace(M)/len(M)*np.eye(len(M)); L=np.linalg.cholesky(A)
    def score(Z):
        B=basis((Z-center)/rms,n); z=solve_triangular(L,B.T,lower=True,check_finite=False); return np.sum(z*z,axis=0)
    return score,np.linalg.cond(A)
def main():
    g=np.linspace(-1.4,1.4,120); xx,yy=np.meshgrid(g,g); G=np.c_[xx.ravel(),yy.ravel()]; records=[]; example={}
    for shape in SHAPES:
        truth=inside(G,shape)
        for seed in SEEDS:
            rng=np.random.default_rng(seed); train=sample(rng,shape,max(N_VALUES)); cal=sample(rng,shape,1000)
            for N,n in itertools.product(N_VALUES,DEGREES):
                X=train[:N]; score,cond=fit(X,n); threshold=np.quantile(score(cal),.95); field=score(G); pred=field<=threshold
                iou=np.sum(pred&truth)/np.sum(pred|truth)
                records.append(dict(geometria=shape,semilla=seed,N=N,grado=n,iou=iou,condicion=cond,umbral=threshold))
                if seed==SEEDS[0] and shape=='Anillo': example[N,n]=(X,pred.reshape(xx.shape),iou)
    df=pd.DataFrame(records); df.to_csv(RES/'diagnostico_grado_muestras.csv',index=False)
    sm=df.groupby(['N','grado']).agg(iou_media=('iou','mean'),iou_sd=('iou','std'),condicion_mediana=('condicion','median')).reset_index(); sm.to_csv(RES/'diagnostico_resumen.csv',index=False)
    fig,axes=plt.subplots(len(N_VALUES),len(DEGREES),figsize=(9.2,6.8),constrained_layout=True)
    for i,N in enumerate(N_VALUES):
        for j,n in enumerate(DEGREES):
            ax=axes[i,j]; X,pred,iou=example[N,n]; ax.contourf(xx,yy,pred.astype(int),levels=[.5,1.5],colors=['#b5d5e8']); ax.contour(xx,yy,inside(G,'Anillo').reshape(xx.shape).astype(int),levels=[.5],colors=['#a05b28'],linestyles='--',linewidths=.8); ax.scatter(*X.T,s=2,c='#233d56',alpha=.7); ax.set(aspect='equal',xticks=[],yticks=[],xlim=(-1.15,1.15),ylim=(-1.15,1.15)); ax.set_title(f'n={n} · IoU={iou:.2f}',fontsize=9); 
            if j==0: ax.set_ylabel(f'N={N}')
    fig.savefig(OUT/'grado_muestras_anillo.pdf'); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,5.5),constrained_layout=True)
    for ax,shape in zip(axes.flat,SHAPES):
        d=df[df.geometria==shape]
        for N in N_VALUES:
            a=d[d.N==N].groupby('grado').iou.agg(['mean','std']); ax.errorbar(a.index,a['mean'],yerr=a['std'],marker='o',capsize=2,label=f'N={N}')
        ax.set(title=shape,xticks=DEGREES,ylim=(0,1.05),xlabel='Grado',ylabel='IoU'); ax.grid(alpha=.2)
    axes.flat[-1].axis('off'); h,l=axes.flat[0].get_legend_handles_labels(); axes.flat[-1].legend(h,l,loc='center',frameon=False); fig.savefig(OUT/'grado_muestras_iou.pdf'); plt.close(fig)
    print(sm.to_string(index=False))
if __name__=='__main__': main()
