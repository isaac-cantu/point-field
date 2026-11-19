# point-field

Generación de nubes de puntos sintéticas con **referencia conocida** y herramientas de
aprendizaje geométrico basadas en la función de **Christoffel–Darboux (CD)**.
Proyecto PEF *Aprendizaje geométrico no supervisado mediante la función de Christoffel–Darboux* (UDEM).

```bash
pip install -e .            # numpy, matplotlib
pip install scikit-learn scikit-image pandas umap-learn pytest   # benchmarks
python -m pytest -q         # 77 pruebas
```

## Librería

| Módulo | Contenido |
|---|---|
| `pointfield.geometry.shapes2d` | `Circle`, `Disk`, `Annulus`, `Ellipse`, `Rectangle`, `Triangle`, `Polygon`, `RegularPolygon`, `Star`, `Line`, `Crescent`, `Spiral` (arquimediana / logarítmica). Muestreo uniforme en longitud de arco o área, `bounds`, `contains`, `signed_distance`. |
| `pointfield.geometry.shapes3d` | `Sphere`, `Cube` (caja), `Torus`, `Cylinder`, `Cone`, `Ellipsoid`, `Helix` (superficies, muestreo uniforme en área). |
| `pointfield.geometry.manifolds` | `GaussianBlob`, `HyperSphere`, `SwissRoll`, `roll` (enrollar coordenadas 2D en 3D). |
| `pointfield.core` | `Geometry` (base), `Pose`, `PointDataset` (X, y, latentes, metadatos reproducibles), `compose` (escenas etiquetadas). |
| `pointfield.transforms` | `translate`, `rotate`, `scale`, `affine`, `extrude`, `embed_linear`, `embed_roll`. |
| `pointfield.perturbations` | `add_noise`, `add_occlusion`, `add_variable_density`, `add_outliers` (registradas en `meta["perturbations"]`). |
| `pointfield.datasets` | Escenarios del benchmark: `compacto`, `no_convexo`, `entrelazado`, `anidado` + reservados `espiral_3brazos`, `espiral_log`. |
| `pointfield.learning` | `ChristoffelDarboux` (M_d, K_d, P_d, Λ_d, diagnóstico de grado), `CDGlobalClustering` (CD inicial), `CDLocalSpectral` (mejora 1), `SzegoPhase` + `CDSzegoSpectral` (mejora 2), descriptores 3D (`descriptors`). |

```python
from pointfield.datasets import make
from pointfield.perturbations import add_occlusion
from pointfield.learning import ChristoffelDarboux, CDSzegoSpectral

ds = add_occlusion(make("entrelazado", seed=100), fraction=0.25, seed=1).normalized()
P = ChristoffelDarboux(degree=4).fit(ds.X).score(ds.X)          # tipicidad de cada punto
labels = CDSzegoSpectral(n_clusters=ds.n_groups).fit_predict(ds.X)
```

## Benchmarks y evidencias del medio término (`benchmarks/`)

Protocolo completo en [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md). Todo se ejecuta desde la raíz del repositorio; cada script es independiente y escribe sus CSV en `benchmarks/results/` y sus figuras en `benchmarks/figures/`.

```bash
bash benchmarks/run_all.sh               # todo, en orden (~45 min en 2 núcleos)
python benchmarks/verificar_cifras.py     # comprueba las 68 cifras citadas en la presentación y el documento
```

| Bloque | Script | Qué produce |
|---|---|---|
| 01 | `fig_explicativas.py cd grado` · `fig_nube.py` | Cálculo de P_d paso a paso; efecto del grado |
| 02 | `fig_cd_2d.py` | Conjuntos de nivel para d = 1, 2, 4, 6; diagnóstico de grado |
| 03 | `fig_libreria.py` · `fig_explicativas.py perturbaciones` | Primitivas, escenarios, perturbaciones |
| 04 | `fig_explicativas.py casos metricas trust knee` | Casos del benchmark, ARI, trustworthiness, regla de la rodilla |
| 04–07 | `benchmark_clustering_2d.py [--dev]` · `fig_benchmark_clustering.py` | Benchmark de agrupamiento (semillas 0–2 y 100–104) + oráculo |
| 05 | `benchmark_embeddings_2d.py` · `fig_benchmark_embeddings.py` · `fig_explicativas.py enrollado` | Benchmark de embeddings |
| 06 | `fig_mejoras.py` · `fig_explicativas.py afinidad szego` | Mejoras de CD paso a paso |
| 07 | `fig_explicativas.py heatmap` | Mapa de calor con miniatura de cada caso |
| 08 | `exploratorio_3d.py` · `fig_explicativas.py descriptor corte3d` | 3D exploratorio |

## Presentación (`presentacion/`)

```bash
python presentacion/preparar_datos.py                   # CSV -> data.json
node presentacion/build_deck.js Medio_Termino.pptx      # requiere pptxgenjs
```
