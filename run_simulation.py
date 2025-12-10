import os
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
from scrna import ScRNALikelihoodParams

from bulk_dna_likelihood import SNV
from PhylEx import mcmc
from ete3 import Tree
import matplotlib.pyplot as plt

# Path to the cherry folder                                     
TREE0_DIR = "cherry1"   

# Defaults only used if CN columns are missing                
DEFAULT_VARIANT_READS = 10
DEFAULT_TOTAL_READS = 10
DEFAULT_MAJOR_CN = 1
DEFAULT_MINOR_CN = 1

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def load_bulk_snvs(tree0_dir):
    bulk_path = os.path.join(tree0_dir, "genotype_ssm.txt")
    df = pd.read_csv(bulk_path, sep="\t")

    snvs = []
    for _, row in df.iterrows():
        variant_reads = int(row["b"])
        total_reads = int(row["d"])

        if "major_cn" in df.columns:
            major_cn = int(row["major_cn"])
        else:
            major_cn = DEFAULT_MAJOR_CN

        if "minor_cn" in df.columns:
            minor_cn = int(row["minor_cn"])
        else:
            minor_cn = DEFAULT_MINOR_CN

        snv = SNV(
            variant_reads=variant_reads,
            total_reads=total_reads,
            major_cn=major_cn,
            minor_cn=minor_cn,
            clone_index=0,
        )

        snv.id = row["ID"]

        snvs.append(snv)

    return snvs


def load_true_snvs(tree0_dir):
    """
    True SNV→clone assignments derived from datum2node.tsv by
    grouping identical genotype strings into clusters.
    """
    d2n_path = os.path.join(tree0_dir, "datum2node.tsv")
    df = pd.read_csv(d2n_path, sep="\t", header=None,
                     names=["sample", "geno"])

    # Factorize genotype strings into cluster IDs 0..K-1
    df["cluster"], _ = pd.factorize(df["geno"])
    df["cluster"] = df["cluster"] + 1  # make them 1..K

    # Order matches sample order (s0, s1, ...) which matches genotype_ssm
    return df["cluster"].values.astype(int)


def load_true_phi(tree0_dir):
    """
    True clone prevalences from cellular_prev.csv + datum2node.tsv:
    φ_k = mean cellular_prev over all SNVs assigned to clone k, normalized.
    """
    # Labels from datum2node
    d2n_path = os.path.join(tree0_dir, "datum2node.tsv")
    df_labels = pd.read_csv(d2n_path, sep="\t", header=None,
                            names=["sample", "geno"])
    df_labels["cluster"], _ = pd.factorize(df_labels["geno"])
    df_labels["cluster"] = df_labels["cluster"] + 1

    # Prevalences
    prev_path = os.path.join(tree0_dir, "cellular_prev.csv")
    df_prev = pd.read_csv(prev_path, sep="\t", header=None,
                          names=["sample", "prev"])

    # Merge and aggregate
    df = df_labels.merge(df_prev, on="sample")
    phi_series = df.groupby("cluster")["prev"].mean().sort_index()
    phi = phi_series.values.astype(float)

    total = phi.sum()
    if total > 0:
        phi = phi / total

    return phi


def load_phi(tree0_dir):
    """Wrapper in case we need ground-truth φ elsewhere."""
    return load_true_phi(tree0_dir)

def run_mcmc_single(tree0_dir):
    print(f"Running MCMC on {tree0_dir} ...")

    bulk_snvs = load_bulk_snvs(tree0_dir)

    scrna_file = f"{tree0_dir}/simul_sc.txt"
    scrna_df = load_scrna_data(scrna_file)

    # Build S ONLY — do NOT return phi, clone_has_snv, params
    S = process_scrna_data(scrna_df, bulk_snvs)

    # Debug check
    print("Checking S for invalid entries...")
    for c, cell in enumerate(S):
        for n, entry in enumerate(cell):
            if not isinstance(entry, tuple) or len(entry) != 2:
                print("INVALID ENTRY:", c, n, entry)
                raise SystemExit("S IS MALFORMED")

    # The ONLY correct way to call mcmc
    map_result = mcmc(
        bulk_snvs=bulk_snvs,
        scrna_data=S,             # <<< THE FIX
        lamb_0=1.0,
        lamb=0.1,
        gamma=5.0,
        epsilon=0.001,
        num_iterations=1000,
        burnin=250,
        use_fixed_tree=True
    )

    return map_result




import pandas as pd
import numpy as np

def load_scrna_data(file_path):
    return pd.read_csv(file_path, sep="\t", header=0)



def process_scrna_data(scrna_df, bulk_snvs):
    scrna_df = scrna_df.copy()

    scrna_df.columns = [c.strip() for c in scrna_df.columns]

    scrna_df["ID"] = scrna_df["ID"].astype(str).str.strip()
    scrna_df["Cell"] = scrna_df["Cell"].astype(str).str.strip()
    scrna_df["a"] = scrna_df["a"].replace("NA", 0).fillna(0).astype(int)
    scrna_df["d"] = scrna_df["d"].replace("NA", 0).fillna(0).astype(int)

    bulk_ids = [snv.id for snv in bulk_snvs]
    id_to_index = {snv_id: i for i, snv_id in enumerate(bulk_ids)}
    M = len(bulk_ids)

    cells = sorted(scrna_df["Cell"].unique())
    C = len(cells)
    cell_to_index = {cell: i for i, cell in enumerate(cells)}

    S = [[(0, 0) for _ in range(M)] for _ in range(C)]

    for _, row in scrna_df.iterrows():
        snv_id = row["ID"]
        if snv_id not in id_to_index:
            continue

        c = cell_to_index[row["Cell"]]
        m = id_to_index[row["ID"]]
        d_tot = int(row["d"])
        b_ref = int(row["a"])
        b = d_tot - b_ref

        S[c][m] = (b, d_tot)

    # ---- FINAL VALIDATION ----
    for c in range(C):
        for m in range(M):
            entry = S[c][m]
            if not (isinstance(entry, tuple) and len(entry) == 2):
                print("BAD ENTRY:", c, m, entry)
                raise SystemExit("STOP: S contains malformed (b, d) values")

    print("S matrix successfully built.")
    print(f"S shape = {len(S)} cells × {len(S[0])} SNVs")

    return S


def analyze_performance(tree0_dir, map_result):
    """
    Evaluate SNV assignment, clone prevalence, and tree structure.
    Uses Hungarian algorithm to compute true cluster accuracy.
    """

    from sklearn.metrics import confusion_matrix, adjusted_rand_score
    from scipy.optimize import linear_sum_assignment

    def cluster_accuracy(true_labels, pred_labels):
        """
        Computes accuracy after optimal label matching (Hungarian algorithm).
        """
        true_labels = np.asarray(true_labels)
        pred_labels = np.asarray(pred_labels)

        cm = confusion_matrix(true_labels, pred_labels)
        row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
        correct = cm[row_ind, col_ind].sum()

        return correct / len(true_labels)

    print(f"true_snvs length: {len(true_snvs := load_true_snvs(tree0_dir))}, "
          f"inferred_snvs length: {len(inferred_snvs := np.asarray(map_result['z'], dtype=int))}")

    min_len = min(len(true_snvs), len(inferred_snvs))
    if min_len == 0:
        print("No SNVs to compare for accuracy/ARI.")
    else:
        acc = cluster_accuracy(true_snvs[:min_len], inferred_snvs[:min_len])
        ari = adjusted_rand_score(true_snvs[:min_len], inferred_snvs[:min_len])

        print(f"SNV assignment accuracy (Hungarian aligned): {acc:.3f}")
        print(f"Adjusted Rand Index (ARI): {ari:.3f}")

    # Clone prevalence comparison
    true_phi = load_true_phi(tree0_dir)
    inferred_phi = np.asarray(map_result["phi"], dtype=float)

    K_true = len(true_phi)
    K_inf = len(inferred_phi)
    print(f"true_phi length: {K_true}, inferred_phi length: {K_inf}")

    # Pad vectors so RMSE is computable
    if K_true == K_inf:
        true_vec = true_phi
        inf_vec = inferred_phi
    elif K_inf < K_true:
        inf_vec = np.zeros(K_true)
        inf_vec[:K_inf] = inferred_phi
        true_vec = true_phi
    else:
        true_vec = np.zeros(K_inf)
        true_vec[:K_true] = true_phi
        inf_vec = inferred_phi

    rmse = np.sqrt(np.mean((true_vec - inf_vec) ** 2))
    print(f"Clone prevalence RMSE: {rmse:.3f} (true K={K_true}, inferred K={K_inf})")

    # Tree comparison using Robinson-Foulds metric
    true_tree = Tree(os.path.join(tree0_dir, "tree.newick"), format=1)
    inferred_tree_newick = map_result["tree"].to_newick()
    inferred_tree = Tree(inferred_tree_newick, format=1)

    rf, max_rf, *_ = true_tree.robinson_foulds(inferred_tree)
    if max_rf <= 0:
        print(f"Robinson-Foulds distance: {rf} (max_rf undefined: {max_rf})")
    else:
        print(f"Robinson-Foulds distance: {rf} / {max_rf}")


def build_cluster_labels_from_datum2node(tree0_dir, out_name="cluster_labels_from_datum2node.tsv"):
    """
    Read datum2node.tsv (sample, genotype string) and turn it into a
    cluster_labels-style file: (sample, cluster_id).

    Each unique genotype string gets its own cluster_id.
    """
    d2n_path = os.path.join(tree0_dir, "datum2node.tsv")
    df = pd.read_csv(d2n_path, sep="\t", header=None, names=["sample", "geno"])

    # Factorize genotype strings → integer labels 0..K-1
    df["cluster"], uniques = pd.factorize(df["geno"])

    # Optional: make cluster IDs 1..K instead of 0..K-1
    df["cluster"] = df["cluster"] + 1

    out_path = os.path.join(tree0_dir, out_name)
    df[["sample", "cluster"]].to_csv(out_path, sep="\t", header=False, index=False)

    print(f"Wrote {out_path} with {df['cluster'].nunique()} clusters.")
    return out_path


if __name__ == "__main__":
    map_result = run_mcmc_single(TREE0_DIR)
    print(map_result["tree"].to_newick())
    analyze_performance(TREE0_DIR, map_result)
