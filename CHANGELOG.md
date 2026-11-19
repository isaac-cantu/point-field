# Changelog

## 0.3.0 — posterior al medio término (noviembre 2026)

### Corregido
- `ChristoffelDarboux`: ridge ε = reg·tr(M)/s y factorización de Cholesky en lugar de
  `pinv` (con M singular, `pinv` hacía que puntos fuera del soporte parecieran típicos).
  Valida datos finitos; añade `kernel`, `condition_number_`, `effective_dof_`.
- `local_cd_affinity`: Cholesky con ridge en lugar de `pinv`.
- `shapes3d/pyramid.py` estaba vacío: nueva `Pyramid`.
- Docstrings: `CDGlobalClustering` (heurística, no EM), fase "Szegő" (nombre histórico),
  descriptores de Reynolds (solo momentos radiales), muestreo exacto del elipsoide.

### Añadido
- Base ponderada √(k!/α!) (`basis="weighted"`), compatible con rotaciones; pesos por muestra.
- `pointfield.learning.invariant`: `CDDescriptor` (CD_U/CD_W), control `moment_descriptor`
  (M_U), bloque angular, pesos de densidad kNN, filtro geométrico de atípicos,
  `BlockScaler`, `MultiViewMetric`, `make_view`.
- `pointfield.learning.evaluation`: mAP, 1-NN, precisión@k, bootstrap pareado estratificado.
- `pointfield.geometry.mesh`: lector OFF (ModelNet), `TriangleMesh`, SHA-256.
- `pointfield.objects`: familias sintéticas de objetos 2D/3D con similitud aleatoria.
- Benchmarks `objetos_invariantes.py` y `modelnet10.py`; resultados sintéticos en
  `benchmarks/results/objetos_{2,3}d.csv`.
- Alias `PhaseFeatures`/`CDPhaseSpectral`.
- 14 pruebas nuevas (91 en total): invarianzas a 1e-9, identidad de la base ponderada,
  Cholesky vs. inversa, grado 1 = Mahalanobis, círculo ridge vs `pinv`, mAP, bootstrap,
  métrica multivista, `Pyramid`, lector OFF.

## 0.2.0 — medio término (octubre 2026)
- Benchmarks de agrupamiento y embeddings 2D, mejoras CD local y fase, 3D exploratorio.
