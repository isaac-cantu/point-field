# Benchmarks — medio término PEF (octubre 2026)

Todo lo reportado aquí se generó con los scripts de esta carpeta; cada CSV en `results/`
tiene una fila por (caso, variante, semilla, método). Entorno: Python 3.13, numpy 2.5.3,
scipy 1.18.1, scikit-learn 1.9.1, scikit-image 0.26, umap-learn 0.5.12.

## Protocolo

| Elemento | Decisión |
|---|---|
| Datos | 4 escenarios principales (`compacto`, `no_convexo`, `entrelazado`, `anidado`) + 2 reservados (`espiral_3brazos`, `espiral_log`), 150 puntos por grupo. Variantes: base, ruido (σ = 0.06·RMS), oclusión direccional (25 %), densidad variable (p_keep 1 → 0.15). |
| Normalización | Centrado + un único factor de escala (radio RMS = 1). Isotrópica a propósito: la fase de Szegő requiere conservar ángulos. |
| Semillas | Desarrollo: 0, 1, 2 (`clustering_2d_dev.csv`), usadas para fijar la compuerta de Szegő. Evaluación: 100–104 (todo lo reportado). Perturbación: semilla + 1000. |
| Igualdad de datos | Todos los métodos reciben exactamente la misma realización. |
| Número de grupos | K-Means, Spectral y los tres CD reciben k verdadero. DBSCAN y OPTICS no. |
| Hiperparámetros | Fijos a priori: K-Means n_init=10; DBSCAN min_samples=5, eps = rodilla de la curva k-distancia (sin etiquetas); OPTICS min_samples=7, xi=0.05, min_cluster_size=0.1 (configuración de la comparación de scikit-learn); Spectral kNN n_neighbors=10; CD global grado 2; CD local k=10 grado 1; compuerta Szegő 0.20. |
| Etiquetas | Solo para ARI/NMI. Excepción declarada: `clustering_2d_oracle.csv` elige el mejor hiperparámetro con las etiquetas y se reporta solo como cota superior (respaldo). |
| Tiempos | Medidos con otro proceso en paralelo (2 núcleos): indicativos, no comparativos finos. |

## Resultados principales (ARI medio, semillas 100–104)

| Método | Principales base | Principales perturbados | Reservados base | Reservados perturbados |
|---|---|---|---|---|
| K-Means | 0.31 | 0.31 | 0.04 | 0.04 |
| DBSCAN (rodilla) | 0.49 | 0.55 | 0.18 | 0.21 |
| OPTICS | 0.33 | 0.25 | 0.27 | 0.16 |
| Spectral (kNN) | 0.79 | 0.71 | 0.36 | 0.19 |
| CD global (inicial) | 0.34 | 0.31 | 0.04 | 0.04 |
| CD local-espectral (mejora 1) | 0.78 | 0.72 | 0.48 | 0.22 |
| CD + fase Szegő (mejora 2) | **0.99** | **0.75** | **0.96** | **0.27** |

Detalle por caso: `results/tabla_clustering_resumen.csv`, `figures/fig_heatmap_ari.png`.

## Lectura

* CD inicial (asignación por menor P_d) se comporta como K-Means: bien en `compacto` (0.96), falla en no convexos y anidados.
* La afinidad local CD (mejora 1) iguala a Spectral kNN (0.78 vs 0.79): resuelve lunas y anillos, no espirales con k = 10.
* La fase de Szegő con compuerta (mejora 2) resuelve las espirales limpias, incluidos los casos reservados (3 brazos 1.00, logarítmica 0.92), y no se activa en `compacto`/`anidado`.
* **Límite**: con oclusión o densidad variable el centro de devanado se desplaza, la compuerta se cierra y el resultado vuelve al de la mejora 1. Con ruido la compuerta queda en el umbral (activación 40 % en `entrelazado + ruido`, ARI 0.42 ± 0.51). En `no_convexo + ruido` se activa y cuesta 0.06 de ARI.
* El oráculo muestra que DBSCAN con el eps correcto también resuelve `entrelazado` (0.94): la diferencia está en elegir el hiperparámetro sin etiquetas.

## Embeddings (escenarios enrollados en 3D → 2D, k = 10)

| Método | T (vs. coords. 2D reales) | C | Exactitud 5-NN |
|---|---|---|---|
| PCA | 0.81 | 0.96 | 0.80 |
| Kernel PCA (RBF) | 0.80 | 0.95 | 0.80 |
| Isomap | 0.86 | 0.96 | 0.88 |
| Laplacian Eigenmaps | 0.86 | 0.95 | 0.83 |
| UMAP | **0.95** | 0.96 | **0.92** |
| CD local-espectral | 0.86 | 0.95 | 0.83 |

## 3D exploratorio (1-NN LOO, 7 clases × 10 instancias)

| Método | base | ruido | oclusión 20 % |
|---|---|---|---|
| CD + Log-Euclidiana d=2, 30 rot. | 0.74 | 0.74 | 0.47 |
| CD + Log-Euclidiana d=4, 30 rot. | 0.89 | 0.87 | 0.48 |
| CD + Log-Euclidiana d=4, 300 rot.* | 0.99 | 0.99 | 0.66 |
| D2 (Osada 2002) | **1.00** | **1.00** | **0.96** |
| Eigen-rasgos PCA | 0.86 | 0.85 | 0.62 |

\* 300 rotaciones se añadió tras un diagnóstico en la semilla de desarrollo 7 (la varianza Monte Carlo del promedio de Reynolds limitaba el resultado).

## Contraste con el reporte consolidado previo

| Afirmación previa | Verificación en este corte |
|---|---|
| CD local-espectral resuelve espirales (ARI 0.99–1.00 con k diagnosticado) | Con k = 10: 0.17. Mejor k elegido con etiquetas: 0.64. No se reproduce con estos datos. |
| Fase de Szegő: ARI 1.00 en espirales | Se reproduce (1.00 en entrelazado, 3 brazos; 0.92 logarítmica). Nuevo: falla con oclusión/densidad variable. |
| CD + Log-Euclidiana supera a D2 en clasificación de objetos | No se reproduce: D2 = 1.00 y CD = 0.99 (con 300 rotaciones); con oclusión D2 0.96 vs CD 0.66. |
| Λ_d separa soporte y hueco del anillo | Se reproduce (d ≥ 2, figura del Bloque 02). |

## Posterior al medio término: reconocimiento de objetos completos

Scripts: `objetos_invariantes.py` (sintético) y `modelnet10.py` (real). Código en
`pointfield.learning.invariant` y `pointfield.learning.evaluation`.

**Protocolo.** Galería de referencia y consultas disjuntas; cada objeto con similitud
aleatoria (rotación, reflexión, escala ∈ [0.5, 2], traslación). Las consultas se
perturban (`clean`, `noise` 3 % RMS, `density` sesgo de muestreo, `occlusion` 20 %,
`outliers` 5 %); la galería queda limpia. Estandarización por bloques y métrica
multivista se ajustan **solo con referencias** (η ∈ {0.1, 1, 10} por recuperación de
identidad en 20 % de ellas). mAP, 1-NN y bootstrap pareado estratificado (3000).

**ModelNet10** (semilla 20261025; 400 referencias / 100 desarrollo de `train` /
200 prueba de `test`; 512 puntos; deduplicación SHA-256):

| Comparación (prueba) | mAP |
|---|---|
| CD_U + métrica | 0.402 |
| M_U + métrica (control sin inversión) | 0.398 — IC de la diferencia incluye 0; no demuestra equivalencia |
| CD_U_metric − M_U_metric con sesgo de densidad | +0.017 / +0.020, IC > 0 |
| Cualquier método sin filtro, 5 % atípicos | ≈ 0.10 |
| Filtro geométrico de atípicos | rescata la condición con atípicos pero erosiona partes delgadas en objetos limpios |

**Sintético** (`results/objetos_{2,3}d.csv`, 20 referencias + 10 consultas por clase):
CD_U + métrica supera a M_U + métrica en todas las condiciones (3D limpio 0.940 vs 0.885),
y la métrica multivista recupera casi todo el sesgo de densidad (3D 0.477 → 0.924). D2
es el más robusto a atípicos sin métrica. Estas familias son más fáciles que ModelNet10
y la ventaja de CD no se transfirió: la conclusión de la tesis se basa en ModelNet10.

**Lectura.** La contribución defendible es el descriptor invariante con controles y la
evidencia acotada de que la normalización CD ayuda bajo sesgo de densidad; no una
superioridad general sobre momentos o D2.


Nota de auditoría (19 nov): las cifras ModelNet anteriores pertenecen a A8. Los scripts públicos actuales y la notebook v1 no son una réplica exacta: cambian escalado, vistas, perturbaciones y selección. No exigir coincidencia numérica sin recuperar los manifiestos y scripts originales.
