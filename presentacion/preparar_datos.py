"""Extrae de benchmarks/results/ las cifras que usa build_deck.js -> data.json."""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "benchmarks", "results")
out = {}
df = pd.read_csv(os.path.join(R, "clustering_2d_eval.csv"))
df["grupo"] = df.held_out.map({True: "reservado", False: "principal"})
df["cond"] = (df.variant == "base").map({True: "base", False: "perturbado"})
pm = df.groupby(["method", "grupo", "cond"]).ARI.mean().round(3).unstack(["grupo", "cond"]).to_dict("index")
out["por_metodo"] = {k: {f"{a}|{b}": v for (a, b), v in d.items()} for k, d in pm.items()}
out["oracle"] = pd.read_csv(os.path.join(R, "tabla_oracle.csv")).round(2).to_dict("records")
e = pd.read_csv(os.path.join(R, "embeddings_2d_eval.csv"))
out["emb"] = e.groupby("method")[["T", "C", "kNN_acc"]].mean().round(2).to_dict("index")
c = pd.read_csv(os.path.join(R, "clasificacion_3d.csv"))
out["c3d"] = c.groupby(["method", "variant"]).accuracy_1nn_loo.agg(["mean", "std"]).round(3).reset_index().to_dict("records")
out["gate_eval"] = pd.read_csv(os.path.join(R, "tabla_compuerta_szego.csv")).round(2).to_dict("records")
d = pd.read_csv(os.path.join(R, "clustering_2d_dev.csv"))
d = d[d.method.str.contains("Szeg")]
out["gate_dev"] = d.groupby(["case", "variant"], sort=False).agg(g=("winding_gain", "mean"), on=("szego_active", "mean")).round(2).reset_index().to_dict("records")
out["deg"] = pd.read_csv(os.path.join(R, "cd_2d_degree_diagnostics.csv")).to_dict("records")
txt = json.dumps(out, indent=0, default=float).replace("NaN", "null")
open(os.path.join(HERE, "data.json"), "w").write(txt)
print("data.json ok")
