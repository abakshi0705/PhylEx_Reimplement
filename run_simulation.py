
import os
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
from bulk_dna_likelihood import SNV
from PhylEx import mcmc  # replace with your module
from ete3 import Tree
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURATION
# -------------------------------
# Path to a single tree0 folder
TREE0_DIR = "/Users/adityabakshi/Downloads/single_region/simul/quadternary_cn_multiregion/rep0/case0/single_region_genotype1/chain0/joint/tree0"

# Default read counts for testing
DEFAULT_VARIANT_READS = 10
DEFAULT_TOTAL_READS   = 100
DEFAULT_MAJOR_CN      = 1
DEFAULT_MINOR_CN      = 1

def load_bulk_snvs(tree0_dir):
    datum2node_file = os.path.join(tree0_dir, "datum2node.tsv")
    df = pd.read_csv(datum2node_file, sep="\t", header=None)
    snvs = []
    for _, row in df.iterrows():
        node_index = int(row[1])
        snvs.append(SNV(
            variant_reads=DEFAULT_VARIANT_READS,
            total_reads=DEFAULT_TOTAL_READS,
            major_cn=DEFAULT_MAJOR_CN,
            minor_cn=DEFAULT_MINOR_CN,
            clone_index=node_index
        ))
    return snvs

def load_true_snvs(tree0_dir):
    df = pd.read_csv(f"{tree0_dir}/datum2node.tsv", sep="\t", header=None)
    return df[1].values

def load_true_phi(tree0_dir):
    df = pd.read_csv(f"{tree0_dir}/cellular_prev.tsv", sep="\t", header=None)
    return df[0].values

def load_phi(tree0_dir):
    phi_file = os.path.join(tree0_dir, "cellular_prev.tsv")
    if os.path.exists(phi_file):
        df = pd.read_csv(phi_file, sep="\t", header=None)
        return df.iloc[:,0].values
    else:
        return None

# -------------------------------
# RUN MCMC
# -------------------------------
def run_mcmc_single(tree0_dir):
    print(f"Running MCMC on {tree0_dir} ...")
    bulk_snvs = load_bulk_snvs(tree0_dir)
    map_result = mcmc(
        bulk_snvs=bulk_snvs,
        scrna_data=None,
        lamb_0=1.0,
        lamb=0.1,
        gamma=5.0,
        epsilon=0.001,
        num_iterations=500,
        burnin=250
    )
    return map_result

# -------------------------------
# PERFORMANCE ANALYSIS
# -------------------------------
def analyze_performance(tree0_dir, map_result):
    # SNV assignment
    true_snvs = load_true_snvs(tree0_dir)
    inferred_snvs = map_result["z"]
    accuracy = (true_snvs == inferred_snvs).mean()
    ari = adjusted_rand_score(true_snvs, inferred_snvs)
    print(f"SNV assignment accuracy: {accuracy:.3f}")
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")

    # Clone prevalence
    true_phi = load_true_phi(tree0_dir)
    inferred_phi = map_result["phi"]
    rmse = np.sqrt(np.mean((true_phi - inferred_phi) ** 2))
    print(f"Clone prevalence RMSE: {rmse:.3f}")

    # Tree comparison (Robinson-Foulds)
    true_tree = Tree(f"{tree0_dir}/tree.newick", format=1)
    inferred_tree_newick = map_result["tree"].to_newick()  # implement `to_newick()` in your Tree class
    inferred_tree = Tree(inferred_tree_newick, format=1)
    rf, max_rf, *_ = true_tree.robinson_foulds(inferred_tree)
    print(f"Robinson-Foulds distance: {rf} / {max_rf}")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    map_result = run_mcmc_single(TREE0_DIR)
    analyze_performance(TREE0_DIR, map_result)