"""Librería común para la competencia Name That LLM.

Contiene: carga de datos, características estilométricas, constructores de
representaciones, modelos base (todos devuelven probabilidades), utilidades de
ensamblaje y el post-procesado por clusters de prompt (asignación húngara).
Este archivo se incrusta tal cual dentro de cada notebook de Kaggle.
"""
import os, re, warnings
import numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.cluster import AgglomerativeClustering

warnings.filterwarnings("ignore")
SEED = 42


# ----------------------------------------------------------------- datos
def find_dir():
    for d in [os.environ.get("DATA_DIR", "."), "/kaggle/input/name-that-llm", "/kaggle/input/competitions/name-that-llm", ".."]:
        if os.path.exists(os.path.join(d, "train.csv")):
            return d
    for root, _, files in os.walk("/kaggle/input"):
        if "train.csv" in files:
            return root
    raise FileNotFoundError("train.csv no encontrado")


def load():
    d = find_dir()
    return pd.read_csv(f"{d}/train.csv"), pd.read_csv(f"{d}/test.csv")


def write_submission(test, labels, path="submission.parquet"):
    sub = pd.DataFrame({"id": test["id"].values, "model": np.asarray(labels)})
    assert sub["model"].notna().all() and len(sub) == len(test)
    sub.to_parquet(path, index=False)
    return sub


# ------------------------------------------------------- estilometría
_PUNCT = list(",.;:!?—–-\"'“”‘’()[]…*#/%$&_`") + ["\n"]
_FUNC = ("the of and to a in that is it for as with was on be by this which "
         "not but or are from at an have has more than their its we our can "
         "however thus yet while rather").split()


def stylo_one(t):
    n = max(len(t), 1)
    words = re.findall(r"[A-Za-zÀ-ÿ']+", t)
    nw = max(len(words), 1)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]
    sl = np.array([len(s.split()) for s in sents]) if sents else np.array([0])
    wl = np.array([len(w) for w in words]) if words else np.array([0])
    low = [w.lower() for w in words]
    paras = [p for p in t.split("\n") if p.strip()]
    f = [
        np.log1p(n), np.log1p(nw), len(sents), sl.mean(), sl.std(), sl.max(), sl.min(),
        wl.mean(), wl.std(), len(set(low)) / nw,
        sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n,
        sum(ord(c) > 127 for c in t) / n, sum(c == " " for c in t) / n,
        len(paras), t.count("\n\n"), len(re.findall(r"\b\w+n't\b|\b\w+'(s|re|ve|ll|d|m)\b", t)) / nw,
        len(re.findall(r"^\s*([-*•]|\d+[.)])\s", t, flags=re.M)),
        sum(len(w) > 9 for w in words) / nw,
        len(re.findall(r"\b[A-Z][a-z]+\b", t)) / nw,
    ]
    f += [t.count(p) / n * 100 for p in _PUNCT]
    cnt = {}
    for w in low:
        cnt[w] = cnt.get(w, 0) + 1
    f += [cnt.get(w, 0) / nw * 100 for w in _FUNC]
    return f


def stylo_matrix(texts):
    return np.array([stylo_one(t) for t in texts], dtype=float)


# ----------------------------------------------------- representaciones
def tfidf_word(ngram=(1, 2), min_df=2, lower=True, max_features=None):
    return TfidfVectorizer(ngram_range=ngram, min_df=min_df, sublinear_tf=True, lowercase=lower,
                           token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=max_features)


def tfidf_char(ngram=(2, 5), min_df=2, lower=False, analyzer="char", max_features=None):
    return TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min_df, sublinear_tf=True,
                           lowercase=lower, max_features=max_features)


def fit_transform_many(vecs, tr_text, te_text):
    A, B = [], []
    for v in vecs:
        A.append(v.fit_transform(tr_text)); B.append(v.transform(te_text))
    return sparse.hstack(A).tocsr(), sparse.hstack(B).tocsr()


# ------------------------------------------------------------ modelos
def softmax(z, T=1.0):
    z = z * T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def proba_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return softmax(model.decision_function(X), 4.0)


def fit_predict_sparse(model, Xtr, ytr, Xte):
    model.fit(Xtr, ytr)
    return proba_of(model, Xte)


def fit_predict_stylo(model, tr_text, ytr, te_text, scale=True):
    A, B = stylo_matrix(tr_text), stylo_matrix(te_text)
    if scale:
        sc = StandardScaler().fit(A); A, B = sc.transform(A), sc.transform(B)
    model.fit(A, ytr)
    return proba_of(model, B)


def blend(ps, w=None):
    w = np.ones(len(ps)) if w is None else np.asarray(w, float)
    out = sum(wi * np.log(np.clip(p, 1e-9, 1)) for wi, p in zip(w, ps)) / w.sum()
    return softmax(out)


def blend_arith(ps, w=None):
    w = np.ones(len(ps)) if w is None else np.asarray(w, float)
    return sum(wi * p for wi, p in zip(w, ps)) / w.sum()


# ------------------------------- post-proceso: 1 texto por modelo y prompt
def cluster_prompts(texts, n_models=18, min_n=None):
    """Cada prompt tiene exactamente un pasaje por modelo, así que n/18 = nº de
    prompts. Agrupa los textos por tema (TF-IDF de contenido + SVD + clustering
    aglomerativo coseno). Con pocos textos (3 ejemplos interactivos) cada texto
    queda en su propio grupo."""
    n = len(texts)
    if n < (min_n or 2 * n_models):
        return np.arange(n)
    k = max(1, int(round(n / n_models)))
    v = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.3)
    X = v.fit_transform(texts)
    Z = TruncatedSVD(min(100, X.shape[1] - 1), random_state=SEED).fit_transform(X)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(Z)


def group_center(M, groups):
    """Resta la media del grupo (prompt) a cada fila: elimina el efecto del tema."""
    M = np.asarray(M, float); out = M.copy()
    for g in np.unique(groups):
        i = groups == g
        out[i] = M[i] - M[i].mean(0)
    return out


def group_rank(M, groups):
    """Rango percentil de cada texto dentro de su grupo, columna a columna."""
    M = np.asarray(M, float); out = np.zeros_like(M)
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        if len(i) < 2:
            out[i] = 0.5; continue
        out[i] = M[i].argsort(0).argsort(0) / (len(i) - 1)
    return out


def assign_in_groups(proba, groups, n_models):
    """Asignación húngara dentro de cada grupo: cada modelo a lo sumo una vez."""
    pred = proba.argmax(1).copy()
    lp = np.log(np.clip(proba, 1e-9, 1))
    for c in np.unique(groups):
        idx = np.where(groups == c)[0]
        if len(idx) < 2 or len(idx) > 2 * n_models:
            continue
        r, col = linear_sum_assignment(-lp[idx])
        pred[idx[r]] = col
    return pred


def prompt_assign(texts, proba, n_models, groups=None):
    g = cluster_prompts(texts, n_models) if groups is None else groups
    return assign_in_groups(proba, g, n_models)


# ------------------------------------------------------------- validación
def cv_eval(fit_predict, train, n_splits=5, post=None, seed=SEED, verbose=True):
    """GroupKFold por prompt_id. fit_predict(tr_df, te_df, classes) -> proba (n,18).
    Devuelve (balanced_accuracy, oof_proba). Si post!=None aplica el post-proceso
    por fold sobre los textos retenidos (sin usar prompt_id)."""
    le = LabelEncoder().fit(train["model"])
    y = le.transform(train["model"])
    oof = np.zeros((len(train), len(le.classes_)))
    pred = np.zeros(len(train), int)
    for f, (a, b) in enumerate(GroupKFold(n_splits).split(train, y, train["prompt_id"])):
        p = fit_predict(train.iloc[a].reset_index(drop=True), train.iloc[b].reset_index(drop=True))
        oof[b] = p
        pred[b] = post(train.iloc[b]["text"].tolist(), p, len(le.classes_)) if post else p.argmax(1)
    ba = balanced_accuracy_score(y, pred)
    return ba, oof, pred
