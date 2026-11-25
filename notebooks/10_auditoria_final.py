"""Auditoría de resultados point-field, sin reentrenar ni sobrescribir archivos.
Uso: python 10_auditoria_final.py --repo /ruta/point-field
Opcional: --tesis /ruta/proyecto_Overleaf
"""
import argparse, hashlib, json, platform, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,default=Path.cwd()); p.add_argument('--tesis',type=Path); args=p.parse_args()
root=args.repo.resolve(); out=root/'runs'/'auditoria_final'; out.mkdir(parents=True,exist_ok=True)
checks=[]; manifest=[]
def check(name, ok, detail=''):
    checks.append(dict(check=name,status='PASS' if ok else 'FAIL',detail=str(detail)))
def read(rel):
    f=root/rel
    if not f.exists():
        checks.append(dict(check=rel,status='MISSING',detail='Archivo ausente')); return None
    b=f.read_bytes(); manifest.append(dict(file=rel,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
    return pd.read_csv(f)
def save(fig,name):
    for ext in ('pdf','png'): fig.savefig(out/f'{name}.{ext}',bbox_inches='tight',dpi=220)
    plt.close(fig)
def export(d,name): d.to_csv(out/f'{name}.csv',index=False)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
methods=['D2','M_U','CD_U','CD_W','M_U_metric','CD_U_metric','CD_W_metric']
mn=[]; contrasts=[]
for ds in ('modelnet10','modelnet40'):
    d=read(f'benchmarks/results/{ds}_test.csv')
    if d is None: continue
    actual=d[d.method.isin(methods)].copy(); actual['dataset']=ds
    check(ds+' filas únicas',not actual.duplicated(['condition','method']).any())
    check(ds+' cobertura',len(actual)==35 and set(actual.method)==set(methods) and set(actual.condition)=={'clean','noise','density','occlusion','outliers'})
    check(ds+' métricas acotadas',actual[['mAP','nn']].apply(lambda x:x.between(0,1)).all().all())
    mn.append(actual)
    for _,r in d[~d.method.isin(methods)].iterrows():
        names=r.method.split(' - ')
        if len(names)!=2 or not all(n in methods for n in names):
            check(ds+' contraste reconocido',False,r.method);continue
        a=actual[actual.condition.eq(r.condition)].set_index('method').mAP
        delta=a[names[0]]-a[names[1]]
        check(ds+' '+r.condition+' '+r.method,np.isclose(delta,r.mAP,atol=1e-9),f'{delta:.9f} vs {r.mAP:.9f}')
        contrasts.append(dict(dataset=ds,condition=r.condition,contrast=r.method,delta=r.mAP,lo=r.ci_low,hi=r.ci_high))
    prev=read(f'runs/tesis_figuras/05_{ds}_test_tabla.csv')
    if prev is not None:
        a=actual.pivot(index='method',columns='condition',values='mAP').sort_index().sort_index(axis=1)
        b=prev.set_index('method').reindex(index=a.index,columns=a.columns)
        check(ds+' tabla notebook 05',np.allclose(a,b,atol=1e-10))
if mn:
    m=pd.concat(mn,ignore_index=True);export(m,'modelnet_resultados_exactos');export(m[m.condition.eq('clean')],'modelnet_limpio')
if contrasts: export(pd.DataFrame(contrasts),'modelnet_contrastes')
for name,metric,source in [('clustering','ARI','clustering_2d_eval'),('embedding','T','embeddings_2d_eval')]:
    d=read(f'benchmarks/results/{source}.csv')
    if d is None: continue
    group=d.groupby('method'); summary=group.time_s.agg(n='count',mean='mean',median='median',sd='std',p25=lambda x:x.quantile(.25),p75=lambda x:x.quantile(.75)).reset_index()
    summary=summary.merge(group[metric].mean().rename(metric),on='method');export(summary,name+'_tiempos')
    check(name+' tiempos no negativos',d.time_s.notna().all() and (d.time_s>=0).all())
    check(name+' métricas válidas',d[metric].between(-1 if metric=='ARI' else 0,1).all())
    fig,ax=plt.subplots(figsize=(8,4.5)); s=summary.sort_values('median'); y=np.arange(len(s));ax.errorbar(s['median']*1000,y,xerr=np.vstack([(s['median']-s.p25)*1000,(s.p75-s['median'])*1000]),fmt='o',capsize=3);ax.set_yticks(y,s.method);ax.set_xscale('log');ax.set_xlabel('Tiempo por ejecución (ms; mediana e IQR)');ax.grid(axis='x',alpha=.25);save(fig,name+'_tiempos')
    fig,ax=plt.subplots(figsize=(8,5));
    for _,r in summary.iterrows(): ax.scatter(r['mean']*1000,r[metric],label=r.method)
    ax.set(xscale='log',xlabel='Tiempo medio por ejecución (ms)',ylabel=f'{metric} medio',title='Mismas filas para tiempo y calidad');ax.legend(fontsize=8,loc='best');ax.grid(alpha=.2);save(fig,name+'_calidad_tiempo')
    if name=='clustering':
        d['scope']=np.where(d.held_out.astype(str).str.lower().eq('true'),'reservado','principal');d['condition_group']=np.where(d.variant.eq('base'),'base','perturbado');export(d.groupby(['scope','condition_group','method']).agg(n=('ARI','size'),ARI=('ARI','mean'),NMI=('NMI','mean'),median_ms=('time_s',lambda x:1000*x.median())).reset_index(),'clustering_por_condicion')
a=read('runs/tesis_figuras/09_metricas_soporte_annulus.csv'); b=read('runs/tesis_figuras/09_diagnostico_annulus.csv')
if a is not None and b is not None:
    check('Anillo 64 configuraciones',len(a)==len(b)==64 and not a.duplicated(['degree','n_points']).any() and not b.duplicated(['degree','n_points']).any())
    check('Anillo ejecuciones OK',a.status.eq('OK').all() and b.status.eq('OK').all())
    j=a.merge(b,on=['degree','n_points'],suffixes=('','_grid'),validate='one_to_one')
    check('Anillo umbrales iguales',np.allclose(j.threshold,j.threshold_grid));check('Anillo IoU desde conteos',np.allclose(j.IoU,j.TP/(j.TP+j.FP+j.FN)));check('Anillo Dice desde conteos',np.allclose(j.Dice,2*j.TP/(2*j.TP+j.FP+j.FN)));check('Anillo 30000 consultas',((j.TP+j.TN+j.FP+j.FN)==30000).all())
    check('Anillo base polinomial',((j.degree+1)*(j.degree+2)/2==j.basis_size).all());export(j,'anillo_verificado')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for n in (100,1000,2000,16000):
        g=j[j.n_points.eq(n)].sort_values('degree');axs[0].plot(g.degree,g.IoU,'o-',label=f'N={n}');axs[1].plot(g.degree,g.eval_time_s_grid,'o-',label=f'N={n}')
    for ax in axs:ax.set_xscale('log',base=2);ax.set_xticks(sorted(j.degree.unique()),labels=sorted(j.degree.unique()));ax.set_xlabel('Grado');ax.grid(alpha=.2)
    axs[0].set(ylabel='IoU',ylim=(0,1));axs[1].set(ylabel='Tiempo de evaluación de rejilla (s)',yscale='log');axs[0].legend();save(fig,'anillo_calidad_costo')
for rel,keys,metrics in [('runs/20261119T210929036674Z/reconstruction_per_object.csv',['dataset','method'],['Hellinger2','Chamfer2']),('notebooks/runs_desde_cero/20261119T212430343107Z/11_decoder_por_objeto.csv',['dataset','modelo'],['H2','Chamfer2'])]:
    d=read(rel)
    if d is not None:
        s=d.groupby(keys)[metrics].agg(['count','mean','std']);s.columns=['_'.join(c) for c in s.columns];export(s.reset_index(),'decoder_'+('historico' if 'reconstruction_per' in rel else 'desde_cero'))
flare=read('benchmarks/results/aplicacion_FLARE_nivel2.csv')
if flare is not None:export(flare,'flare_nivel2')
# Resultados externos: protocolos separados. No promediar con resultados propios.
external=[('FoldingNet','SVM lineal',94.4,88.4,'https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_FoldingNet_Point_Cloud_CVPR_2018_paper.pdf'),('Latent-GAN','SVM lineal',95.3,85.7,'https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_FoldingNet_Point_Cloud_CVPR_2018_paper.pdf'),('Point-M2AE','SVM lineal',np.nan,92.9,'https://proceedings.neurips.cc/paper_files/paper/2022/hash/ad1d7a4df30a9c0c46b387815a774a84-Abstract-Conference.html'),('I2P-MAE','SVM lineal',np.nan,93.4,'https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Learning_3D_Representations_From_2D_Pre-Trained_Models_via_Image-to-Point_Masked_CVPR_2023_paper.html'),('Point-MAE','Fine-tuning',np.nan,94.04,'https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/800_ECCV_2022_paper.php')]
export(pd.DataFrame(external,columns=['method','protocol','MN10_accuracy_pct','MN40_accuracy_pct','source']),'literatura_clasificacion')
if args.tesis:
    import re
    t=args.tesis.resolve(); rows=[]
    for f in sorted(t.rglob('*.tex')):
        for match in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',f.read_text(errors='replace')):
            rel=match.group(1); q=t/rel; exists=q.exists() or any(q.with_suffix(x).exists() for x in ('.pdf','.png','.jpg'));rows.append(dict(tex=str(f.relative_to(t)),line=f.read_text(errors='replace')[:match.start()].count('\n')+1,figure=rel,exists=exists))
    if rows:export(pd.DataFrame(rows),'inventario_figuras_tex')
export(pd.DataFrame(checks),'checks');export(pd.DataFrame(manifest),'manifest')
try: sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
except Exception:sha='unknown'
(out/'metadata.json').write_text(json.dumps({'commit':sha,'system':platform.platform(),'python':platform.python_version(),'note':'Tiempos archivados; no medidos en esta auditoría.'},indent=2))
print(pd.DataFrame(checks).to_string(index=False));print('Salidas:',out)
