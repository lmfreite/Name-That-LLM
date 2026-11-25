"""Genera notebooks/submission_XX.ipynb autocontenidos para Kaggle (Internet apagado):
leen test.csv, entrenan con train.csv y escriben submission.parquet."""
import json, os, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
lib = open(f"{ROOT}/src/lib.py").read()
exp = open(f"{ROOT}/src/experiments.py").read()
body = exp.split("SPEC = [")[0].replace("from lib import *\n", "")
import sys; sys.path.insert(0, f"{ROOT}/src")
from experiments import SPEC
os.makedirs(f"{ROOT}/notebooks", exist_ok=True)
res = json.load(open(f"{ROOT}/results.json")) if os.path.exists(f"{ROOT}/results.json") else {}
def cell(t, src): return {"cell_type": t, "metadata": {}, "source": src.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {})}
for (i, week, name, fp, assign, hyp) in SPEC:
    cv = res.get(str(i), {}).get("cv_bal_acc", "n/d")
    md = (f"# Envío {i:02d} — {name}\n\nSemana {week}. Hipótesis: {hyp}\n\n"
          f"CV local (GroupKFold por prompt, 5 folds): **{cv}** balanced accuracy.\n\n"
          "Notebook autocontenido: celda 1 = librería común, celda 2 = definición de los experimentos, celda 3 = ejecución. "
          "Parte de la idea de los notebooks públicos heliosli/name-that-llm-lv681-a15 y omkarpattanaik123/name-that-llm-v28-sovereign-titan-apex; "
          "el aporte propio son el centrado por prompt, la asignación húngara y los ensambles descritos en el informe.")
    run = f'''tr, te = load()
le = LabelEncoder().fit(tr["model"])
proba = {fp.__name__}(tr, te)
labels = prompt_assign(te["text"].tolist(), proba, len(le.classes_)) if {assign} else proba.argmax(1)
sub = write_submission(te, le.classes_[labels], "submission.parquet")
print(sub.head(), sub["model"].value_counts().head())
'''
    nb = {"cells": [cell("markdown", md), cell("code", lib), cell("code", body), cell("code", run)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(f"{ROOT}/notebooks/submission_{i:02d}.ipynb", "w"), indent=1)
print("ok")
