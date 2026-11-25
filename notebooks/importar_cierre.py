"""Import verified original figures from a local point-field clone; no screenshots."""
import argparse, shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args(); dst=Path(__file__).resolve().parents[1]/'figures'/'experimentos'; src=a.repo/'runs'/'tesis_figuras'
for name in ['09_grid_grado_puntos_annulus.pdf','01_primitivas_2d.pdf','01_primitivas_3d.pdf','01_escenarios.pdf','01_perturbaciones.pdf','03_inicial.pdf','03_evolucion.pdf','04_heatmap_principales.pdf','04_heatmap_reservados.pdf','05_embedding_sintetico.pdf','07_modelnet_comparacion_corregida.pdf']:
    f=src/name
    if f.exists():shutil.copy2(f,dst/name);print('copied',name)
for name in ['11_reconstruccion_PF2D.png','11_reconstruccion_PF3D.png']:
    f=a.repo/'notebooks/runs_desde_cero/20261119T212430343107Z'/name
    if f.exists():shutil.copy2(f,dst/name);print('copied',name)
