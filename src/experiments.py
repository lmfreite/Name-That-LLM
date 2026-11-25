"""Los 20 experimentos. Cada uno es una función  fit_predict(tr, te) -> proba (len(te), 18)
con columnas ordenadas como LabelEncoder().fit(train.model).classes_.
EXPERIMENTS[i] = dict(id, name, week, fp, assign(bool), hyp)  — assign=True aplica el
post-proceso de clusters de prompt + asignación húngara sobre las probabilidades."""
import numpy as np
from lib import *
import lightgbm as lgb

_LE = None
def le_of(tr):
    return LabelEncoder().fit(tr["model"])

def _y(tr): return le_of(tr).transform(tr["model"])

def _groups(tr, te):
    return tr["prompt_id"].values, cluster_prompts(te["text"].tolist())

def _dense_centered(vecs, tr, te):
    X, Z = fit_transform_many(vecs, tr.text, te.text)
    ga, gb = _groups(tr, te)
    return group_center(X.toarray(), ga), group_center(Z.toarray(), gb)

def _stylo(tr, te, mode="center"):
    A, B = stylo_matrix(tr.text), stylo_matrix(te.text)
    ga, gb = _groups(tr, te)
    if mode == "raw": return A, B
    if mode == "center": return group_center(A, ga), group_center(B, gb)
    return (np.hstack([A, group_center(A, ga), group_rank(A, ga)]),
            np.hstack([B, group_center(B, gb), group_rank(B, gb)]))

def _scaled(A, B):
    sc = StandardScaler().fit(A); return sc.transform(A), sc.transform(B)

# ------------------------------------------------------------ SEMANA 1
def e01(tr, te):   # NB sobre TF-IDF de palabras
    X, Z = fit_transform_many([tfidf_word((1, 2))], tr.text, te.text)
    return fit_predict_sparse(MultinomialNB(alpha=0.05), X, _y(tr), Z)

def e02(tr, te):   # LR sobre TF-IDF de palabras
    X, Z = fit_transform_many([tfidf_word((1, 2))], tr.text, te.text)
    return fit_predict_sparse(LogisticRegression(C=20, max_iter=300), X, _y(tr), Z)

def e03(tr, te):   # SVM lineal sobre n-gramas de caracteres (con mayúsculas y puntuación)
    X, Z = fit_transform_many([tfidf_char((2, 5))], tr.text, te.text)
    return fit_predict_sparse(LinearSVC(C=0.5), X, _y(tr), Z)

def e04(tr, te):   # palabras + caracteres, SVM lineal
    X, Z = fit_transform_many([tfidf_word((1, 2)), tfidf_char((2, 5))], tr.text, te.text)
    return fit_predict_sparse(LinearSVC(C=0.5), X, _y(tr), Z)

def e05(tr, te):   # palabras + caracteres, regresión logística
    X, Z = fit_transform_many([tfidf_word((1, 2)), tfidf_char((2, 5))], tr.text, te.text)
    return fit_predict_sparse(LogisticRegression(C=30, max_iter=300), X, _y(tr), Z)

def e06(tr, te):   # solo estilometría (36+ rasgos) + LR
    A, B = _scaled(*_stylo(tr, te, "raw"))
    return fit_predict_sparse(LogisticRegression(C=0.3, max_iter=1000), A, _y(tr), B)

def e07(tr, te):   # TF-IDF + estilometría apilada como columnas dispersas
    X, Z = fit_transform_many([tfidf_word((1, 2)), tfidf_char((2, 5))], tr.text, te.text)
    A, B = _scaled(*_stylo(tr, te, "raw"))
    X = sparse.hstack([X, sparse.csr_matrix(A * 0.15)]).tocsr(); Z = sparse.hstack([Z, sparse.csr_matrix(B * 0.15)]).tocsr()
    return fit_predict_sparse(LogisticRegression(C=30, max_iter=300), X, _y(tr), Z)

def e08(tr, te):   # ensamble: LR(word+char) + NB + estilometría
    return blend([e05(tr, te), e01(tr, te), e06(tr, te)], [2, 1, 1])

def e09(tr, te):   # + asignación 1 texto por modelo y prompt sobre e08
    return e08(tr, te)

def e10(tr, te):   # centrado por prompt de la estilometría (LR)
    A, B = _scaled(*_stylo(tr, te, "center"))
    return fit_predict_sparse(LogisticRegression(C=0.3, max_iter=1000), A, _y(tr), B)

# ------------------------------------------------------------ SEMANA 2
def e11(tr, te):   # TF-IDF palabras+chars centrado por prompt, LinearSVC
    X, Z = _dense_centered([tfidf_word((1, 2), 3, max_features=15000), tfidf_char((2, 5), 3, max_features=25000)], tr, te)
    return fit_predict_sparse(LinearSVC(C=0.3), X, _y(tr), Z)

def e12(tr, te):   # idem con LR
    X, Z = _dense_centered([tfidf_word((1, 2), 3, max_features=15000), tfidf_char((2, 5), 3, max_features=25000)], tr, te)
    return fit_predict_sparse(LogisticRegression(C=30, max_iter=300), X, _y(tr), Z)

def e13(tr, te):   # + asignación sobre e12
    return e12(tr, te)

def e14(tr, te):   # ensamble centrado: TF-IDF centrado + estilometría centrada
    return blend([e12(tr, te), e10(tr, te)], [3, 1])

def e15(tr, te):   # estilometría completa (cruda+centrada+rango) con LightGBM
    A, B = _stylo(tr, te, "all")
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, verbose=-1, n_jobs=-1, random_state=SEED)
    m.fit(A, _y(tr)); return m.predict_proba(B)

def e16(tr, te):   # ensamble e12 + e10 + LightGBM
    return blend([e12(tr, te), e10(tr, te), e15(tr, te)], [3, 1, 1])

def e17(tr, te):   # LR centrada con n-gramas de caracteres de 1-6 y palabras 1-3
    X, Z = _dense_centered([tfidf_word((1, 3), 2, max_features=20000), tfidf_char((1, 6), 3, max_features=40000)], tr, te)
    return fit_predict_sparse(LogisticRegression(C=50, max_iter=400), X, _y(tr), Z)

def e18(tr, te):   # ensamble grande: e12+e17+e10+e15+e04(no centrado)
    return blend([e12(tr, te), e17(tr, te), e10(tr, te), e15(tr, te), e04(tr, te)], [3, 3, 1, 1, 1])

def e19(tr, te):   # e18 + asignación
    return e18(tr, te)

def e20(tr, te):   # e18 + auto-entrenamiento (pseudo-etiquetas) + asignación
    base = e18(tr, te)
    ga, gb = _groups(tr, te)
    if len(te) < 4 * 18:
        return base
    pl = assign_in_groups(base, gb, 18)
    conf = base[np.arange(len(te)), pl]
    keep = conf >= np.quantile(conf, 0.5)
    tr2 = pd.concat([tr, te[keep].assign(model=le_of(tr).classes_[pl[keep]], prompt_id=["T%d" % g for g in gb[keep]])], ignore_index=True)
    p2 = e18(tr2, te)
    return blend([base, p2], [1, 1])

SPEC = [
 (1, 1, "TF-IDF palabras + Naive Bayes", e01, False, "Línea base mínima: bolsa de palabras."),
 (2, 1, "TF-IDF palabras + Regresión Logística", e02, False, "Un clasificador discriminativo mejora a NB."),
 (3, 1, "TF-IDF caracteres (2-5, con mayúsculas) + LinearSVC", e03, False, "La puntuación/espaciado estilístico vive en los caracteres."),
 (4, 1, "Palabras + caracteres + LinearSVC", e04, False, "Las dos vistas son complementarias."),
 (5, 1, "Palabras + caracteres + Regresión Logística", e05, False, "LR da probabilidades útiles para ensamblar."),
 (6, 1, "Estilometría (≈60 rasgos) + LR", e06, False, "Rasgos de forma (longitud, signos) independientes del tema."),
 (7, 1, "TF-IDF + estilometría en un solo espacio (LR)", e07, False, "Unir contenido léxico y forma."),
 (8, 1, "Ensamble LR + NB + estilometría", e08, False, "Promediar modelos diversos reduce varianza."),
 (9, 1, "Ensamble e08 + asignación húngara por prompt", e09, True, "Cada prompt tiene 1 texto por modelo: restricción explotable."),
 (10, 1, "Estilometría centrada por prompt + asignación", e10, True, "Quitar el efecto del tema comparando textos del mismo prompt."),
 (11, 2, "TF-IDF centrado por prompt + LinearSVC + asignación", e11, True, "Centrar el TF-IDF aísla la huella del modelo."),
 (12, 2, "TF-IDF centrado por prompt + LR + asignación", e12, True, "LR rinde mejor que SVM en el espacio centrado."),
 (13, 2, "e12 sin asignación (ablación)", e13, False, "Mide cuánto aporta solo la asignación."),
 (14, 2, "Ensamble centrado (TF-IDF + estilometría) + asignación", e14, True, "Combinar contenido centrado y forma centrada."),
 (15, 2, "LightGBM sobre estilometría cruda+centrada+rango + asignación", e15, True, "Árboles capturan interacciones no lineales (p. ej. longitud relativa)."),
 (16, 2, "Ensamble e12 + e10 + LightGBM + asignación", e16, True, "Añadir diversidad no lineal."),
 (17, 2, "LR centrada con n-gramas más amplios + asignación", e17, True, "Palabras 1-3 y caracteres 1-6."),
 (18, 2, "Ensamble grande (5 modelos) sin asignación", e18, False, "Tope del ensamble antes del post-proceso."),
 (19, 2, "Ensamble grande + asignación húngara", e19, True, "Combina todas las mejoras."),
 (20, 2, "Ensamble grande + pseudo-etiquetas del test + asignación", e20, True, "Aprendizaje semisupervisado transductivo sobre el test."),
]
