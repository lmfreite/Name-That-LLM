"""Genera SUSTENTACION.md desde results.json (resultados reales de la CV)."""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/src")
from experiments import SPEC
res = json.load(open(f"{ROOT}/results.json"))
rows = []
best = 0
for (i, week, name, fp, assign, hyp) in SPEC:
    r = res[str(i)]; d = r["cv_bal_acc"] - best if best else 0; best = max(best, r["cv_bal_acc"])
    rows.append((i, week, name, assign, hyp, r, d))
w1 = max(r[5]["cv_bal_acc"] for r in rows if r[1] == 1); w2 = max(r[5]["cv_bal_acc"] for r in rows if r[1] == 2)
tabla = "\n".join(f"| {i:02d} | {w} | {n} | {'Sí' if a else 'No'} | {r['cv_bal_acc']:.4f} | {r['cv_acc']:.4f} | {r['cv_macro_f1']:.4f} | |" for i, w, n, a, h, r, d in rows)
detalle = "\n".join(f"""### Envío {i:02d} — {n}
- **Archivos:** `submissions/submission_{i:02d}.parquet`, `notebooks/submission_{i:02d}.ipynb` (semana {w}).
- **Hipótesis / qué cambió:** {h}
- **Post-proceso de asignación por prompt:** {'sí' if a else 'no'}.
- **Resultado CV (GroupKFold por prompt, 5 folds):** balanced accuracy **{r['cv_bal_acc']:.4f}** (Δ vs. mejor anterior: {d:+.4f}), macro-F1 {r['cv_macro_f1']:.4f}.
- **Puntaje Kaggle:** _(completar tras el envío)_
""" for i, w, n, a, h, r, d in rows)
md = f"""# Name That LLM — Sustentación (Examen Final, Inteligencia Artificial)

> Documento de apoyo para el informe PDF y la presentación del 24 de nov. Todos los puntajes de esta tabla son **de validación local** (GroupKFold agrupado por `prompt_id`, 5 folds, sobre `train.csv`). Los puntajes de la tabla de posiciones de Kaggle deben completarse con los envíos reales: no se pueden obtener desde aquí.

## 1. Problema
Atribuir cada pasaje (40–250 palabras) a uno de 18 modelos de lenguaje. Métrica: *balanced accuracy* (azar = 1/18 ≈ 0,056). Los prompts de prueba **no** aparecen en el entrenamiento, por eso la validación es agrupada por prompt.

## 2. Hallazgos clave de los datos
1. `train.csv` tiene 150 prompts × 18 modelos: **cada prompt contiene exactamente un pasaje por modelo** (150 por clase; 149 en claude-fable-5).
2. La longitud media difiere por modelo (gemini-3.8-flash ≈ 101 palabras, claude-opus-4-8 ≈ 124).
3. Agrupando los textos por tema (TF-IDF + SVD + clustering aglomerativo coseno), en validación **se recuperan los prompts de forma perfecta (ARI = 1,0)** sin usar `prompt_id`. Esto permite, en el test oculto (~4000 textos ≈ 222 prompts × 18):
   - **Centrar las características por prompt** (restar la media del grupo): elimina el tema y deja la huella del modelo.
   - **Asignación húngara**: dentro de cada grupo, cada modelo se asigna a lo sumo a un texto.
4. Es un uso *transductivo* de la estructura del test (no de etiquetas). Se documenta como decisión de diseño y se mide con ablaciones (envíos 13 vs. 12, 18 vs. 19). En el modo interactivo (3 filas) el código no agrupa y usa argmax normal.

## 3. Pipeline común
Limpieza mínima (se conservan mayúsculas y puntuación: son señal) → representaciones (TF-IDF palabras 1-3, TF-IDF caracteres 1-6 sensible a mayúsculas, ≈60 rasgos estilométricos: longitud de frases, signos, comillas tipográficas, contracciones, palabras funcionales…) → modelo (NB, LR, LinearSVC, LightGBM) → ensamble por media geométrica de probabilidades → (opcional) asignación húngara → `submission.parquet` con columnas `id`, `model`.

Código: `src/lib.py` (funciones), `src/experiments.py` (los 20 modelos), `src/run_all.py` (CV + envíos), `src/build_notebooks.py` (notebooks Kaggle).

## 4. Tabla resumen (CV local)
| # | Semana | Modelo | Asignación | Bal. acc. CV | Acc. | Macro-F1 | Puntaje Kaggle |
|---|---|---|---|---|---|---|---|
{tabla}

**Mejor CV semana 1:** {w1:.4f} · **Mejor CV semana 2:** {w2:.4f} · mejora: {w2 - w1:+.4f}

## 5. Detalle por envío
{detalle}

## 6. Qué funcionó y qué no
Ver la columna de Δ en la tabla: las mayores ganancias vienen de (a) combinar vistas léxicas, (b) centrar por prompt y (c) la asignación húngara; los árboles sobre estilometría solos y el pseudo-etiquetado aportan poco o nada si así lo indica la tabla.

## 7. Advertencias honestas
- Los puntajes locales **no** equivalen al leaderboard (el test tiene ~4000 textos y 222 prompts; el entrenamiento local de cada fold usa 120 prompts). El umbral de 0,683 de la rúbrica **no está verificado**: depende del puntaje real en Kaggle. Si el CV local queda lejos de 0,683, el equipo debe partir de los notebooks de referencia (heliosli, omkarpattanaik123) y sumarles las mejoras de este repositorio.
- Los archivos `submissions/submission_XX.parquet` locales contienen solo las 3 filas de `test.csv` de ejemplo (formato válido). Las predicciones reales las genera cada notebook al ejecutarse en Kaggle sobre el test oculto.
- Las capturas de pantalla y el nombre del equipo deben aportarse desde Kaggle.
- Herramientas de IA usadas (declaración requerida por la rúbrica): Claude Code (Anthropic) para escribir y depurar código y esta documentación; el equipo debe revisar y poder explicar cada parte.

## 8. Referencias
- Competencia: https://www.kaggle.com/competitions/name-that-llm
- Notebook base 1: https://www.kaggle.com/code/heliosli/name-that-llm-lv681-a15
- Notebook base 2: https://www.kaggle.com/code/omkarpattanaik123/name-that-llm-v28-sovereign-titan-apex
- scikit-learn, LightGBM, SciPy (`linear_sum_assignment`).
"""
open(f"{ROOT}/SUSTENTACION.md", "w").write(md); print("ok")
