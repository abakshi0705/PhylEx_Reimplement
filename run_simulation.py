import os
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score

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

"""
    Load bulk SNVs from genotype_ssm.txt, which has columns:
    ID, CHR, POS, REF, ALT, b, d, major_cn, minor_cn.
    """
def load_bulk_snvs(tree0_dir):
    bulk_path = os.path.join(tree0_dir, "genotype_ssm.txt")      
    df = pd.read_csv(bulk_path, sep="\t")                        

    snvs = []
    for _, row in df.iterrows():
        # read counts
        variant_reads = int(row["b"])                          
        total_reads = int(row["d"])                            

        # copy numbers (fall back to defaults if not present)   
        if "major_cn" in df.columns:                             
            major_cn = int(row["major_cn"])                     
        else:                                                    
            major_cn = DEFAULT_MAJOR_CN                         

        if "minor_cn" in df.columns:                            
            minor_cn = int(row["minor_cn"])                  
        else:                                                  
            minor_cn = DEFAULT_MINOR_CN                      

        snvs.append(
            SNV(
                variant_reads=variant_reads,
                total_reads=total_reads,
                major_cn=major_cn,
                minor_cn=minor_cn,
                clone_index=0,   
            )
        )
    return snvs

""" def loadscRNAseqData(tree0_dir):
    scrna_path = os.path.join(tree0_dir, "simul_ssm.txt") """


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
    map_result = mcmc(
        bulk_snvs=bulk_snvs,
        scrna_data=None,   # simul_ssm not wired in yet           # //FIX (keep None for now)
        lamb_0=1.0,
        lamb=0.1,
        gamma=5.0,
        epsilon=0.001,
        num_iterations=1000,
        burnin=250,
        use_fixed_tree=True
    )
    return map_result


def analyze_performance(tree0_dir, map_result):
    # SNV assignment
    true_snvs = load_true_snvs(tree0_dir)
    inferred_snvs = np.asarray(map_result["z"], dtype=int)
    print(f"true_snvs length: {len(true_snvs)}, inferred_snvs length: {len(inferred_snvs)}")

    # ARI is label-invariant, so fine even if label IDs differ
    min_len = min(len(true_snvs), len(inferred_snvs))          
    if min_len == 0:                                            
        print("No SNVs to compare for accuracy/ARI.")           
    else:
        acc = (true_snvs[:min_len] == inferred_snvs[:min_len]).mean()
        ari = adjusted_rand_score(true_snvs[:min_len], inferred_snvs[:min_len])
        print(f"SNV assignment accuracy (first {min_len}): {acc:.3f}")
        print(f"Adjusted Rand Index (ARI, first {min_len}): {ari:.3f}")

    # Clone prevalence
    true_phi = load_true_phi(tree0_dir)
    inferred_phi = np.asarray(map_result["phi"], dtype=float)

    K_true = len(true_phi)
    K_inf = len(inferred_phi)
    print(f"true_phi length: {K_true}, inferred_phi length: {K_inf}")

    # Pad shorter vector with zeros so RMSE is always defined
    if K_true == K_inf:
        true_vec = true_phi
        inf_vec = inferred_phi
    elif K_inf < K_true:
        inf_vec = np.zeros(K_true, dtype=float)
        inf_vec[:K_inf] = inferred_phi
        true_vec = true_phi
    else:  # K_inf > K_true
        true_vec = np.zeros(K_inf, dtype=float)
        true_vec[:K_true] = true_phi
        inf_vec = inferred_phi

    rmse = np.sqrt(np.mean((true_vec - inf_vec) ** 2))
    print(
        f"Clone prevalence RMSE: {rmse:.3f} "
        f"(true K={K_true}, inferred K={K_inf})"
    )

    # Tree comparison (Robinson-Foulds)
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
