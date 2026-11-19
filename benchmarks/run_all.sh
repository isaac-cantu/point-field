#!/usr/bin/env bash
# Ejecuta todos los experimentos y figuras del medio término, en orden.
# Tiempo aproximado en una laptop de 2 núcleos: 45 min (el benchmark de
# clustering con oráculo es lo más lento). Cada paso se puede correr solo.
set -euo pipefail
cd "$(dirname "$0")"
step() { echo; echo "=== $1"; }

step "Bloque 02 · CD en 2D";                  python fig_cd_2d.py
step "Bloque 03 · librería y escenarios";     python fig_libreria.py
step "Bloque 04 · semillas de desarrollo";    python benchmark_clustering_2d.py --dev
step "Bloque 05/07 · semillas de evaluación"; python benchmark_clustering_2d.py
step "Bloque 05/07 · tablas y figuras";       python fig_benchmark_clustering.py
step "Bloque 05 · embeddings";                python benchmark_embeddings_2d.py
python fig_benchmark_embeddings.py
step "Bloque 06 · mejoras";                   python fig_mejoras.py
step "Bloque 08 · 3D exploratorio";           python exploratorio_3d.py
step "Figuras explicativas";                  python fig_explicativas.py
python fig_nube.py
step "Verificación de cifras";                python verificar_cifras.py
