#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import csv
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering,
    Birch, AffinityPropagation, MeanShift, OPTICS, DBSCAN
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    f1_score, adjusted_rand_score, normalized_mutual_info_score
)

from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA)
PAIRS = [a + b for a in AA for b in AA]

def read_fasta(path):
    seqs = {}
    cur_id, cur_seq = None, []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur_seq)
                header = line[1:].strip()
                cur_id = header.split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None:
        seqs[cur_id] = "".join(cur_seq)
    return seqs

def features_2x2_binary(seq, skips, headers_index):
    s = seq.upper()
    n = len(s)
    vec = np.zeros(len(headers_index), dtype=np.float64)
    for x in skips:
        step = x + 1
        if n <= step:
            continue
        for i in range(n - step):
            a = s[i]
            b = s[i + step]
            if a in AA_SET and b in AA_SET:
                key = f"{a}{b}|skip={x}"
                j = headers_index.get(key)
                if j is not None:
                    vec[j] = 1.0
    return vec

def load_labels_csv(path):
    df = pd.read_csv(path)
    need = {"seq_id", "label"}
    if not need.issubset(df.columns):
        raise ValueError("labels_csv deve conter colunas: seq_id,label")
    return dict(zip(df["seq_id"].astype(str), df["label"].astype(str)))

def _best_f1_by_hungarian(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cost = np.zeros((len(labels_true), len(labels_pred)))
    for i, lt in enumerate(labels_true):
        for j, lp in enumerate(labels_pred):
            ti = (y_true == lt).astype(int)
            pj = (y_pred == lp).astype(int)
            f1_pair = f1_score(ti, pj, zero_division=0)
            cost[i, j] = -f1_pair
    ri, cj = linear_sum_assignment(cost)
    mapping = {labels_pred[j]: labels_true[i] for i, j in zip(ri, cj)}
    y_pred_aligned = np.array([mapping.get(y, y) for y in y_pred])
    f1_macro = f1_score(y_true, y_pred_aligned, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred_aligned, average="weighted", zero_division=0)
    return f1_macro, f1_weighted

def _internal_metrics(X, labels):
    n_eff = len(set(labels)) - (1 if -1 in labels else 0)
    if n_eff < 2 or len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "calinski": np.nan, "davies": np.nan}
    return {
        "silhouette": silhouette_score(X, labels),
        "calinski": calinski_harabasz_score(X, labels),
        "davies": davies_bouldin_score(X, labels),
    }

def _external_metrics_if_any(labels_true, labels_pred):
    if labels_true is None:
        return {"f1_macro": np.nan, "f1_weighted": np.nan, "ari": np.nan, "nmi": np.nan}
    labels_true = np.asarray(labels_true, dtype=object)
    labels_pred = np.asarray(labels_pred, dtype=object)
    mask = np.array([lt is not None for lt in labels_true])
    if mask.sum() < 2:
        return {"f1_macro": np.nan, "f1_weighted": np.nan, "ari": np.nan, "nmi": np.nan}
    y_t = labels_true[mask]
    y_p = labels_pred[mask]
    if np.unique(y_t).size < 2 or np.unique(y_p).size < 2:
        return {"f1_macro": np.nan, "f1_weighted": np.nan, "ari": np.nan, "nmi": np.nan}
    f1m, f1w = _best_f1_by_hungarian(y_t, y_p)
    ari = adjusted_rand_score(y_t, y_p)
    nmi = normalized_mutual_info_score(y_t, y_p)
    return {"f1_macro": f1m, "f1_weighted": f1w, "ari": ari, "nmi": nmi}

def run_all_clusterings(X, seq_ids, labels_true=None, max_k=12, run_dbscan=False, run_all=False):
    n, d = X.shape
    max_k = max(2, min(max_k, n - 1))
    results = []
    def add_result(algo, params, labels_pred):
        met_int = _internal_metrics(X, labels_pred)
        met_ext = _external_metrics_if_any(labels_true, labels_pred)
        row = {
            "algo": algo,
            "params": json.dumps(params, ensure_ascii=False),
            "n_clusters_found": int(len(set(labels_pred)) - (1 if -1 in labels_pred else 0)),
            **met_int, **met_ext,
        }
        results.append(row)
    K_RANGE = list(range(2, max_k + 1))
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        add_result("KMeans", {"k": k}, km.fit_predict(X))
        mb = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=min(512, n))
        add_result("MiniBatchKMeans", {"k": k}, mb.fit_predict(X))
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        add_result("Agglomerative(ward)", {"k": k}, agg.fit_predict(X))
        try:
            agg_avg = AgglomerativeClustering(n_clusters=k, linkage="average", metric="euclidean")
            add_result("Agglomerative(average)", {"k": k}, agg_avg.fit_predict(X))
        except TypeError:
            agg_avg = AgglomerativeClustering(n_clusters=k, linkage="average", affinity="euclidean")
            add_result("Agglomerative(average)", {"k": k}, agg_avg.fit_predict(X))
        try:
            spec = SpectralClustering(n_clusters=k, random_state=0, assign_labels="kmeans", n_init=10)
            add_result("Spectral", {"k": k}, spec.fit_predict(X))
        except Exception:
            pass
        bir = Birch(n_clusters=k)
        add_result("Birch", {"k": k}, bir.fit_predict(X))
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=1e-6,
                random_state=0
            )
            add_result("GaussianMixture", {"k": k, "cov": "diag", "reg": 1e-6},
                       gmm.fit(X).predict(X))
        except Exception:
            add_result("GaussianMixture", {"k": k, "cov": "diag", "reg": 1e-6, "status": "failed"},
                       np.full(X.shape[0], -1, dtype=int))
    if run_all:
        try:
            ap = AffinityPropagation(random_state=0)
            add_result("AffinityPropagation", {}, ap.fit_predict(X))
        except Exception:
            pass
        try:
            ms = MeanShift()
            add_result("MeanShift", {}, ms.fit_predict(X))
        except Exception:
            pass
        try:
            op = OPTICS(min_samples=max(5, int(0.02 * n)))
            add_result("OPTICS", {"min_samples": int(max(5, 0.02 * n))}, op.fit_predict(X))
        except Exception:
            pass
    if run_dbscan:
        X_std = np.std(X, axis=0).mean() + 1e-8
        for eps_mult in [0.5, 1.0, 1.5]:
            eps = eps_mult * X_std
            for ms in [3, 5, 10]:
                try:
                    db = DBSCAN(eps=float(eps), min_samples=ms)
                    add_result("DBSCAN", {"eps": round(float(eps), 6), "min_samples": ms}, db.fit_predict(X))
                except Exception:
                    pass
    return pd.DataFrame(results)

def correlate_internal_with_f1(df):
    out_rows = []
    def _corr_block(sub, tag):
        for metric in ["silhouette", "calinski", "davies"]:
            sub_ok = sub[["f1_macro", metric]].dropna()
            if len(sub_ok) >= 3 and sub_ok["f1_macro"].nunique() > 1 and sub_ok[metric].nunique() > 1:
                p = pearsonr(sub_ok["f1_macro"], sub_ok[metric])[0]
                s = spearmanr(sub_ok["f1_macro"], sub_ok[metric])[0]
            else:
                p, s = np.nan, np.nan
            out_rows.append({"scope": tag, "metric": metric, "pearson": p, "spearman": s, "n": len(sub_ok)})
    _corr_block(df, "GLOBAL")
    for algo, sub in df.groupby("algo"):
        _corr_block(sub, f"ALGO::{algo}")
    return pd.DataFrame(out_rows)

def build_and_predict(algo, params, X):
    if algo == "KMeans":
        k = int(params["k"])
        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    if algo == "MiniBatchKMeans":
        k = int(params["k"])
        return MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=min(512, X.shape[0])).fit_predict(X)
    if algo == "Agglomerative(ward)":
        k = int(params["k"])
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if algo == "Agglomerative(average)":
        k = int(params["k"])
        try:
            return AgglomerativeClustering(n_clusters=k, linkage="average", metric="euclidean").fit_predict(X)
        except TypeError:
            return AgglomerativeClustering(n_clusters=k, linkage="average", affinity="euclidean").fit_predict(X)
    if algo == "Spectral":
        k = int(params["k"])
        return SpectralClustering(n_clusters=k, random_state=0, assign_labels="kmeans", n_init=10).fit_predict(X)
    if algo == "Birch":
        k = int(params["k"])
        return Birch(n_clusters=k).fit_predict(X)
    if algo == "GaussianMixture":
        k = int(params["k"])
        cov = params.get("cov", "diag")
        reg = float(params.get("reg", 1e-6))
        try:
            return GaussianMixture(n_components=k, covariance_type=cov, reg_covar=reg, random_state=0).fit(X).predict(X)
        except Exception:
            return np.full(X.shape[0], -1, dtype=int)
    if algo == "AffinityPropagation":
        return AffinityPropagation(random_state=0).fit_predict(X)
    if algo == "MeanShift":
        return MeanShift().fit_predict(X)
    if algo == "OPTICS":
        ms = int(params.get("min_samples", max(5, int(0.02 * X.shape[0]))))
        return OPTICS(min_samples=ms).fit_predict(X)
    if algo == "DBSCAN":
        eps = float(params["eps"])
        ms = int(params["min_samples"])
        return DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
    return np.full(X.shape[0], -1, dtype=int)

def pick_best_configuration(df, select_by="silhouette"):
    df = df.copy()
    df = df[df["n_clusters_found"] >= 2]
    if select_by == "silhouette":
        df["_rank_key"] = list(zip(-df["silhouette"].fillna(-np.inf),
                                   -df["calinski"].fillna(-np.inf),
                                   df["davies"].fillna(np.inf)))
        df_best = df.sort_values("_rank_key").drop(columns=["_rank_key"])
        return df_best.iloc[0].to_dict()
    if select_by == "calinski":
        df["_rank_key"] = list(zip(-df["calinski"].fillna(-np.inf),
                                   -df["silhouette"].fillna(-np.inf),
                                   df["davies"].fillna(np.inf)))
        df_best = df.sort_values("_rank_key").drop(columns=["_rank_key"])
        return df_best.iloc[0].to_dict()
    if select_by == "davies":
        df["_rank_key"] = list(zip(df["davies"].fillna(np.inf),
                                   -df["silhouette"].fillna(-np.inf),
                                   -df["calinski"].fillna(-np.inf)))
        df_best = df.sort_values("_rank_key").drop(columns=["_rank_key"])
        return df_best.iloc[0].to_dict()
    return df.iloc[0].to_dict()

def make_plots(outdir, X_pca, ids, labels_true_map, best_algo, best_params, best_labels, df_var, df_clust):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    os.makedirs(outdir, exist_ok=True)

    # Scree
    fig = plt.figure(figsize=(8, 5))
    y = df_var["explained_variance_ratio"].values
    plt.plot(range(1, len(y)+1), y, marker="o")
    plt.xlabel("PC")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA Scree")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "pca_scree.png"), dpi=150)
    plt.close(fig)

    # Scatter PC1 x PC2 (labels externos se houver)
    fig = plt.figure(figsize=(6, 6))
    x1, x2 = X_pca[:, 0], X_pca[:, 1]
    if labels_true_map:
        labs = [labels_true_map.get(i, None) for i in ids]
        uniq = sorted(list({l for l in labs if l is not None}))
        if len(uniq) >= 2:
            for l in uniq:
                m = [li == l for li in labs]
                plt.scatter(x1[m], x2[m], s=16, label=str(l), alpha=0.8)
            plt.legend(title="Label")
        else:
            plt.scatter(x1, x2, s=16, alpha=0.8)
    else:
        plt.scatter(x1, x2, s=16, alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA PC1×PC2")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "pca_scatter_pc1_pc2.png"), dpi=150)
    plt.close(fig)

    # Melhor configuração: PC1 x PC2 colorido por cluster
    fig = plt.figure(figsize=(6,6))
    plt.scatter(x1, x2, c=best_labels, s=16, alpha=0.85)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Best ({best_algo}) by clusters")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"pca_scatter_best_{best_algo}.png"), dpi=150)
    plt.close(fig)

    # Curvas para KMeans (métricas vs k)
    df_km = df_clust[df_clust["algo"] == "KMeans"].copy()
    if not df_km.empty and "params" in df_km:
        df_km["k"] = df_km["params"].apply(lambda s: json.loads(s).get("k") if isinstance(s, str) else None)
        df_km = df_km.dropna(subset=["k"])
        df_km = df_km.sort_values("k")
        fig = plt.figure(figsize=(8,5))
        if df_km["silhouette"].notna().any():
            plt.plot(df_km["k"], df_km["silhouette"], marker="o", label="Silhouette")
        if df_km["calinski"].notna().any():
            plt.plot(df_km["k"], df_km["calinski"], marker="o", label="Calinski")
        if df_km["davies"].notna().any():
            plt.plot(df_km["k"], df_km["davies"], marker="o", label="Davies")
        plt.xlabel("k")
        plt.ylabel("score")
        plt.title("KMeans: métricas internas vs k")
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, "kmeans_metrics_vs_k.png"), dpi=150)
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="2X2 binário + PCA + Clustering (métricas internas/externas).")
    ap.add_argument("--fasta", required=True, help="Caminho do arquivo FASTA")
    ap.add_argument("--outdir", required=True, help="Diretório de saída")
    ap.add_argument("--skips", default="0,1", help="Skips separados por vírgula (ex.: 0,1,2)")
    ap.add_argument("--limit", type=int, default=0, help="Processa apenas as N primeiras sequências")
    ap.add_argument("--pca_components", type=int, default=300, help="Número de componentes principais (PCA)")
    ap.add_argument("--labels_csv", default=None, help="CSV com colunas seq_id,label para métrica externa (F1). Opcional.")
    ap.add_argument("--max_k", type=int, default=12, help="Máx. clusters/Componentes em modelos que pedem K (padrão: 12).")
    ap.add_argument("--run_dbscan", action="store_true", help="Também roda DBSCAN com pequena grade de eps/min_samples.")
    ap.add_argument("--run_all", action="store_true", help="Inclui AffinityPropagation, MeanShift e OPTICS (pode demorar).")
    ap.add_argument("--select_by", default="silhouette", choices=["silhouette","calinski","davies"], help="Métrica interna-alvo para seleção da melhor configuração")
    ap.add_argument("--plot", action="store_true", help="Gera e salva gráficos PNG no outdir")
    args = ap.parse_args()

    skips = [int(x.strip()) for x in args.skips.split(",") if x.strip() != ""]
    os.makedirs(args.outdir, exist_ok=True)

    seqs = read_fasta(args.fasta)
    ids = list(seqs.keys())
    if args.limit and args.limit > 0:
        ids = ids[:args.limit]

    headers = [f"{p}|skip={x}" for x in skips for p in PAIRS]
    headers_index = {h: i for i, h in enumerate(headers)}

    X = np.zeros((len(ids), len(headers)), dtype=np.float64)
    for r, sid in enumerate(ids):
        X[r, :] = features_2x2_binary(seqs[sid], skips, headers_index)
    X = X.astype(np.float64)

    out_csv = os.path.join(args.outdir, f"kmer_2X2_binary_skips-{'-'.join(map(str,skips))}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq_id"] + headers)
        for r, sid in enumerate(ids):
            w.writerow([sid] + X[r, :].astype(int).tolist())
    print(f"[OK] Matriz binária salva: {out_csv}")
    print(f"  - N sequências: {len(ids)}")
    print(f"  - D atributos:  {len(headers)} (400 pares × {len(skips)} skips)")

    n_samples, n_features = X.shape
    if n_samples < 2:
        print("[AVISO] PCA requer ao menos 2 amostras. Projeção não realizada.")
        return

    k_req = int(args.pca_components)
    k_max = min(k_req, n_samples, n_features)
    if k_max < k_req:
        print(f"[INFO] Ajustando componentes PCA de {k_req} para {k_max} (limitado por N={n_samples}, D={n_features}).")

    pca = PCA(n_components=k_max, svd_solver="auto", random_state=0)
    X_pca = pca.fit_transform(X).astype(np.float64)

    pcs_cols = [f"PC{i}" for i in range(1, k_max + 1)]
    df_scores = pd.DataFrame(X_pca, columns=pcs_cols)
    df_scores.insert(0, "seq_id", ids)
    scores_csv = os.path.join(args.outdir, f"pca_scores_k{k_max}.csv")
    df_scores.to_csv(scores_csv, index=False)

    var_ratio = pca.explained_variance_ratio_
    df_var = pd.DataFrame({
        "PC": pcs_cols,
        "explained_variance_ratio": var_ratio,
        "cumulative_variance_ratio": np.cumsum(var_ratio)
    })
    var_csv = os.path.join(args.outdir, f"pca_explained_variance_k{k_max}.csv")
    df_var.to_csv(var_csv, index=False)

    print(f"[OK] PCA salvo:")
    print(f"  - Scores:    {scores_csv}")
    print(f"  - Variância: {var_csv}")
    print(f"  - Var. acumulada (PC1..PC{k_max}): {df_var['cumulative_variance_ratio'].iloc[-1]:.4f}")

    labels_map = load_labels_csv(args.labels_csv) if args.labels_csv else None
    labels_true = None
    if labels_map is not None:
        labels_true = [labels_map.get(sid, None) for sid in ids]
        if any(lbl is None for lbl in labels_true):
            print("[AVISO] Alguns seq_id não têm rótulo em labels_csv; serão ignorados no cálculo de F1/ARI/NMI.")

    print("[INFO] Rodando algoritmos de clustering…")
    df_clust = run_all_clusterings(
        X=X_pca,
        seq_ids=ids,
        labels_true=None if labels_true is None else np.array(labels_true, dtype=object),
        max_k=args.max_k,
        run_dbscan=args.run_dbscan,
        run_all=args.run_all
    )
    clust_csv = os.path.join(args.outdir, f"clustering_results_kmax{args.max_k}.csv")
    df_clust.to_csv(clust_csv, index=False)
    print(f"[OK] Resultados de clustering salvos em: {clust_csv}")

    df_corr = correlate_internal_with_f1(df_clust)
    corr_csv = os.path.join(args.outdir, f"clustering_internal_vs_f1_correlations.csv")
    df_corr.to_csv(corr_csv, index=False)
    print(f"[OK] Correlações salvas em: {corr_csv}")

    best = pick_best_configuration(df_clust, select_by=args.select_by)
    best_algo = best["algo"]
    best_params = json.loads(best["params"])
    best_labels = build_and_predict(best_algo, best_params, X_pca)
    best_labels_csv = os.path.join(args.outdir, f"best_labels_{args.select_by}.csv")
    pd.DataFrame({"seq_id": ids, "cluster": best_labels}).to_csv(best_labels_csv, index=False)
    best_json = {
        "select_by": args.select_by,
        "algo": best_algo,
        "params": best_params,
        "metrics": {
            "silhouette": float(best.get("silhouette")) if pd.notna(best.get("silhouette")) else None,
            "calinski": float(best.get("calinski")) if pd.notna(best.get("calinski")) else None,
            "davies": float(best.get("davies")) if pd.notna(best.get("davies")) else None,
            "f1_macro": float(best.get("f1_macro")) if pd.notna(best.get("f1_macro")) else None,
            "f1_weighted": float(best.get("f1_weighted")) if pd.notna(best.get("f1_weighted")) else None,
        }
    }
    with open(os.path.join(args.outdir, f"best_config_{args.select_by}.json"), "w", encoding="utf-8") as f:
        json.dump(best_json, f, ensure_ascii=False, indent=2)
    print("[OK] Melhor configuração sugerida:")
    print(f"  - select_by: {args.select_by}")
    print(f"  - algo:      {best_algo}")
    print(f"  - params:    {json.dumps(best_params)}")
    print(f"  - labels:    {best_labels_csv}")

    if args.plot:
        make_plots(
            outdir=args.outdir,
            X_pca=X_pca,
            ids=ids,
            labels_true_map=(labels_map if labels_map else {}),
            best_algo=best_algo,
            best_params=best_params,
            best_labels=best_labels,
            df_var=df_var,
            df_clust=df_clust
        )
        print(f"[OK] Gráficos salvos em: {args.outdir}")

if __name__ == "__main__":
    main()
