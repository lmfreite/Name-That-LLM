"""Ejecuta los 20 experimentos: CV agrupada por prompt (5 folds), guarda OOF en caché,
y escribe submissions/submission_XX.parquet (formato sample_submission) más results.csv."""
import sys, os, time, json, pickle
sys.path.insert(0, os.path.dirname(__file__))
from lib import *
from experiments import SPEC, le_of

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(f"{ROOT}/submissions", exist_ok=True); os.makedirs(f"{ROOT}/cache", exist_ok=True)
tr, te = load()
le = le_of(tr); y = le.transform(tr.model)
only = [int(a) for a in sys.argv[1:]] or [s[0] for s in SPEC]
rows = json.load(open(f"{ROOT}/results.json")) if os.path.exists(f"{ROOT}/results.json") else {}
for (i, week, name, fp, assign, hyp) in SPEC:
    if i not in only: continue
    t0 = time.time()
    post = (lambda t, p, n: prompt_assign(t, p, n)) if assign else None
    cache = f"{ROOT}/cache/oof_{i:02d}.pkl"
    ba, oof, pred = cv_eval(fp, tr, post=post)
    from sklearn.metrics import accuracy_score, f1_score
    row = dict(id=i, week=week, name=name, assign=assign, cv_bal_acc=round(ba, 4),
               cv_acc=round(accuracy_score(y, pred), 4), cv_macro_f1=round(f1_score(y, pred, average="macro"), 4))
    p = fp(tr, te)
    lab = prompt_assign(te.text.tolist(), p, len(le.classes_)) if assign else p.argmax(1)
    write_submission(te, le.classes_[lab], f"{ROOT}/submissions/submission_{i:02d}.parquet")
    row["secs"] = round(time.time() - t0); rows[str(i)] = row
    json.dump(rows, open(f"{ROOT}/results.json", "w"), indent=1)
    pickle.dump(dict(oof=oof, pred=pred), open(cache, "wb"))
    print(row, flush=True)
