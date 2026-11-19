# point-field

Generación de nubes de puntos sintéticas con **referencia conocida** y herramientas de
aprendizaje geométrico basadas en la función de **Christoffel–Darboux (CD)**.
Proyecto PEF *Aprendizaje geométrico no supervisado mediante la función de Christoffel–Darboux* (UDEM).

```bash
pip install -e ".[learning]"   # numpy, matplotlib, scipy, scikit-learn
pip install -e ".[bench,dev]"  # benchmarks (scikit-image, pandas, umap-learn) y pytest
python -m pytest -q            # 91 pruebas
```

## Librería

| Módulo | Contenido |
|---|---|
| `pointfield.geometry.shapes2d` | `Circle`, `Disk`, `Annulus`, `Ellipse`, `Rectangle`, `Triangle`, `Polygon`, `RegularPolygon`, `Star`, `Line`, `Crescent`, `Spiral` (arquimediana / logarítmica). Muestreo uniforme en longitud de arco o área, `bounds`, `contains`, `signed_distance`. |
| `pointfield.geometry.shapes3d` | `Sphere`, `Cube` (caja), `Torus`, `Cylinder`, `Cone`, `Ellipsoid`, `Helix`, `Pyramid` (superficies, muestreo uniforme en área). |
| `pointfield.geometry.mesh` | `read_off` (incluye el encabezado defectuoso `OFF490 518 0` de ModelNet), `TriangleMesh` (muestreo uniforme en área), `file_sha256` (deduplicación). |
| `pointfield.objects` | Familias sintéticas de objetos 2D/3D con variación intraclase y similitud aleatoria (rotación, **reflexión**, escala, traslación): `make_object_dataset`. |
| `pointfield.geometry.manifolds` | `GaussianBlob`, `HyperSphere`, `SwissRoll`, `roll` (enrollar coordenadas 2D en 3D). |
| `pointfield.core` | `Geometry` (base), `Pose`, `PointDataset` (X, y, latentes, metadatos reproducibles), `compose` (escenas etiquetadas). |
| `pointfield.transforms` | `translate`, `rotate`, `scale`, `affine`, `extrude`, `embed_linear`, `embed_roll`. |
| `pointfield.perturbations` | `add_noise`, `add_occlusion`, `add_variable_density`, `add_outliers` (registradas en `meta["perturbations"]`). |
| `pointfield.datasets` | Escenarios del benchmark: `compacto`, `no_convexo`, `entrelazado`, `anidado` + reservados `espiral_3brazos`, `espiral_log`. |
| `pointfield.learning` | `ChristoffelDarboux` (M, ridge ε = reg·tr(M)/s, Cholesky; base `"monomial"` o `"weighted"` √(k!/α!) compatible con rotaciones; pesos por muestra), `CDGlobalClustering` (CD inicial, heurística), `CDLocalSpectral` (mejora 1), `PhaseFeatures`/`CDPhaseSpectral` (mejora 2; alias de `SzegoPhase`/`CDSzegoSpectral`). |
| `pointfield.learning.invariant` | Descriptor de objeto `CDDescriptor` (CD_U, 78 dims en 3D; con `sample_weight` = CD_W), control `moment_descriptor` (M_U), `angular_block`, `knn_density_weights`, `geometric_outlier_filter`, `BlockScaler`, `MultiViewMetric`, `make_view`. |
| `pointfield.learning.evaluation` | mAP, 1-NN, precisión@k, bootstrap pareado estratificado. |
| `pointfield.learning.descriptors` | Descriptores del medio término (Reynolds + Log-Euclidiana, D2). **Histórico**: ver advertencias en el módulo. |

```python
from pointfield.datasets import make
from pointfield.perturbations import add_occlusion
from pointfield.learning import ChristoffelDarboux, CDSzegoSpectral

ds = add_occlusion(make("entrelazado", seed=100), fraction=0.25, seed=1).normalized()
P = ChristoffelDarboux(degree=4).fit(ds.X).score(ds.X)          # tipicidad de cada punto
labels = CDSzegoSpectral(n_clusters=ds.n_groups).fit_predict(ds.X)

# descriptor invariante de un objeto completo
from pointfield.objects import make_object_dataset
from pointfield.learning import CDDescriptor
clouds, y, families = make_object_dataset(dim=3, n_per_class=10, n_points=512, seed=0)
f = CDDescriptor()(clouds[0])     # (78,) invariante a O(3), escala, traslación y permutación
```

> **Nota de nombres.** "Szegő" se conserva solo por compatibilidad: la mejora 2 usa
> la fase angular alrededor de un centro de devanado, no el núcleo de Szegő. Las espirales
> las resuelve la fase sola; el componente CD no es el responsable.

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

## Trabajo posterior al medio término: objetos completos

Protocolo y resultados en [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md#posterior-al-medio-término-reconocimiento-de-objetos-completos).

```bash
python benchmarks/objetos_invariantes.py --dim 3        # familias sintéticas, 5 condiciones
python benchmarks/objetos_invariantes.py --dim 2
python benchmarks/modelnet10.py --root data/ModelNet10 --split dev    # requiere ModelNet10 local
python benchmarks/modelnet10.py --root data/ModelNet10 --split test
```

ModelNet10 no se descarga automáticamente; `data/` está en `.gitignore`.

Resultado principal (ModelNet10, prueba): **CD_U + métrica multivista 0.402 mAP vs. control
de momentos M_U + métrica 0.398 (empate)**. CD aporta solo bajo sesgo de densidad
(CD_W − CD_U: +0.017/+0.020, IC > 0); con 5 % de valores atípicos todos los métodos sin
filtro caen a ≈ 0.10. En las familias sintéticas CD supera a M_U con más margen, pero esa
ventaja **no se transfiere** a ModelNet10: reportar siempre el control M_U y D2.

## Fe de erratas respecto al medio término

* El núcleo usaba `pinv` con M singular; ahora ridge + Cholesky (en el círculo de grado 2,
  `pinv` da q(0) ≈ 0.45 —el centro parece *más* típico que los datos— y el ridge da ≈ 67).
* La afinidad local CD también usaba `pinv`; ahora usa Cholesky con ridge.
* `shapes3d/pyramid.py` estaba vacío; ahora existe `Pyramid`.
* `CDGlobalClustering` es una heurística de reasignación, no EM.
* Descriptor de Reynolds (`descriptors.cd_moment_descriptor`): el promedio conserva solo
  momentos radiales (cubo y octaedro colisionan). Reemplazado por `learning.invariant`.

Las cifras del medio término siguen reproduciéndose: `verificar_cifras.py` → 68/68.
