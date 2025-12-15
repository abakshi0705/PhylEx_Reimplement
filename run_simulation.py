import os
import pandas as pd
import numpy as np
#import arviz as az
import pickle
import multiprocessing as mp

#from sklearn.metrics import adjusted_rand_score
from scrna import ScRNALikelihoodParams
#from analyze_convergence import analyze_conv
import math

from bulk_dna_likelihood import SNV
from PhylEx import mcmc, mcmc_fixed_snv
#from ete3 import Tree
#import matplotlib.pyplot as plt

#from sklearn.metrics import confusion_matrix
#from scipy.optimize import linear_sum_assignment


# Path to the cherry folder                                     
TREE0_DIR = "cherry1"   

# Defaults only used if CN columns are missing (as they were in some of our initial test cases)             
DEFAULT_VARIANT_READS = 10
DEFAULT_TOTAL_READS = 10
DEFAULT_MAJOR_CN = 1
DEFAULT_MINOR_CN = 1

"""
Load our bulk snv data from genotype_ssm.txt;
genotype_ssm.txt contains:
    SNV IDs 
    b: variant reads
    d: total reads
    major_cn: major copy number
    minor_cn: minor copy number
"""
def load_bulk_snvs(tree0_dir):
    bulk_path = os.path.join(tree0_dir, "genotype_ssm.txt")

    #read in our bulk snv data
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
    true_snv_path = os.path.join(tree0_dir, "cluster_labels.txt")
    df_temp = pd.read_csv(true_snv_path, header = None, names=["ID", "Clone_Number"])

    # True SNV clone assignments derived from datum2node.tsv by
    # grouping identical genotype strings into clusters.
    
    d2n_path = os.path.join(tree0_dir, "datum2node.tsv")
    df = pd.read_csv(d2n_path, sep="\t", header=None,
                     names=["sample", "geno"])

    # Factorize genotype strings into cluster IDs 0..K-1
    df["cluster"], _ = pd.factorize(df["geno"])
    df["cluster"] = df["cluster"] + 1  # make them 1..K

    # Order matches sample order (s0, s1, ...) which matches genotype_ssm

    #print ("OLD Way", df["cluster"].values.astype(int))
    #print ("New way", df_temp["Clone_Number"].values.astype(int))

    return df_temp["Clone_Number"].values.astype(int) 
 



def load_true_phi(tree0_dir):
    true_phi_path = os.path.join(tree0_dir, "cellular_prev.csv")
    df_temp = pd.read_csv(true_phi_path, header = None, names = ["ID", "cell_prev"], sep = "\\s+")

    snvs_to_clone_number = load_true_snvs(tree0_dir)
    set_of_clones = set(snvs_to_clone_number.copy())
    
    map_clones_to_prev = {i: -math.inf for i in set_of_clones}

    snv_phis = df_temp["cell_prev"].values.astype(float)

    for i in range(len(snv_phis)):
        map_clones_to_prev[snvs_to_clone_number[i]] = snv_phis[i]
                
    phi = [0 for _ in range(len(set_of_clones))]
    for i in map_clones_to_prev:
        phi[i-1] = float(map_clones_to_prev[i])


    sum = 0
    for i in phi:
        sum+= i
    if sum > 0:
        for i in range(len(phi)):
            phi[i] = phi[i]/float(sum)
        
    print("Phi: ", phi)

    return phi 



    """  
    # True clone prevalences from cellular_prev.csv + datum2node.tsv:
    # phi_k = mean cellular_prev over all SNVs assigned to clone k, normalized.
    # 
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

    return phi """


def load_phi(tree0_dir):
    """Wrapper in case we need ground-truth phi elsewhere."""
    return load_true_phi(tree0_dir)


def load_scrna_data(file_path):
    return pd.read_csv(file_path, sep="\t", header=0)



def process_scrna_data(scrna_df, bulk_snvs):
    scrna_df = scrna_df.copy()

    #set column names, strip any white space from them
    scrna_df.columns = [c.strip() for c in scrna_df.columns]
 
    #strip any white space from each columns contents; also replace any "NAs" with 0 
    scrna_df["ID"] = scrna_df["ID"].astype(str).str.strip()
    scrna_df["Cell"] = scrna_df["Cell"].astype(str).str.strip()
    scrna_df["a"] = scrna_df["a"].replace("NA", 0).fillna(0).astype(int)
    scrna_df["d"] = scrna_df["d"].replace("NA", 0).fillna(0).astype(int)

    #get the ids for each bulk snv
    bulk_ids = [snv.id for snv in bulk_snvs]

    #convert each snv_id to an index 
    id_to_index = {snv_id: i for i, snv_id in enumerate(bulk_ids)}
    M = len(bulk_ids)

    #holds all the unique cell names from the data file 
    cells = sorted(scrna_df["Cell"].unique())
    C = len(cells)
    cell_to_index = {cell: i for i, cell in enumerate(cells)}

    #setup our S matrix
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

    for c in range(C):
        for m in range(M):
            entry = S[c][m]
            if not (isinstance(entry, tuple) and len(entry) == 2):
                print("BAD ENTRY:", c, m, entry)
                raise SystemExit("STOP: S contains malformed (b, d) values")

    print("S matrix successfully built.")
    print(f"S shape = {len(S)} cells x {len(S[0])} SNVs")

    return S

def run_mcmc_single(tree0_dir, seed):
    print("--------------------------------------------------------------------")
    print(f"Running MCMC on {tree0_dir} ...")

    np.random.seed(seed)

    bulk_snvs = load_bulk_snvs(tree0_dir)

    scrna_file = f"{tree0_dir}/simul_sc.txt"
    scrna_df = load_scrna_data(scrna_file)

    # Build S ONLY — do NOT return phi, clone_has_snv, params
    S = process_scrna_data(scrna_df, bulk_snvs)

    # Fixed tree has K=3 clones by construction
    K = 3
    #we use "np.ones" because our alpha is set to 1.0 
    phi_init = np.random.dirichlet(np.ones(K))

    # Debug check
    print("Checking S for invalid entries...")
    for c, cell in enumerate(S):
        for n, entry in enumerate(cell):
            if not isinstance(entry, tuple) or len(entry) != 2:
                print("INVALID ENTRY:", c, n, entry)
                raise SystemExit("S IS MALFORMED")

    print("Seed:", seed, "phi_init:", phi_init[:3])

    map_result, one_chain = mcmc(
        bulk_snvs=bulk_snvs,
        scrna_data=S,             
        lamb_0=1.0,
        lamb=0.1,
        gamma=5.0,
        epsilon=0.001,
        num_iterations=100000,
        burnin=250,
        use_fixed_tree=True,
        phi_init=phi_init
    )


    return map_result, one_chain

def run_parallel_chains(tree0_dir, num_chains = 4, save_to_disk = True):
    print("Reached parallel chains")

    chain_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    seed = 1000 + chain_id
    
    print(f"Running chain {chain_id}")
    map_result, chain = run_mcmc_single(tree0_dir, seed)
    

    # args = [tree0_dir] * num_chains
    # with mp.Pool(processes=num_chains) as pool:
    #     chains = pool.map(run_mcmc_single, args)

    if save_to_disk:
        filename = f"chain{chain_id}.pkl"
        print(f"Saving chain {chain_id} to {filename}")
        with open(filename, "wb") as f:
            pickle.dump(chain, f)
        
        with open(f"chain{chain_id}_map.pkl", "wb") as f:
            pickle.dump(map_result, f)

    return chain

def run_mcmc_fixed_snvs(tree0_dir):
    print("Reached Run MCMC Fixed SNVs")
    chain_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    seed = 1008 + chain_id
    print("--------------------------------------------------------------------")
    print(f"Running chain {chain_id}")
    print("--------------------------------------------------------------------")
    print(f"Running MCMC on {tree0_dir} ...")

    np.random.seed(seed)

    bulk_snvs = load_bulk_snvs(tree0_dir)

    scrna_file = f"{tree0_dir}/simul_sc.txt"
    scrna_df = load_scrna_data(scrna_file)

    # Build S ONLY 
    S = process_scrna_data(scrna_df, bulk_snvs)

    # Fixed tree has K=3 clones by construction
    K = 3
    #we use "np.ones" because our alpha is set to 1.0 
    phi_init = np.random.dirichlet(np.ones(K))
    z = load_true_snvs(tree0_dir)

    # Debug check
    print("Checking S for invalid entries...")
    for c, cell in enumerate(S):
        for n, entry in enumerate(cell):
            if not isinstance(entry, tuple) or len(entry) != 2:
                print("INVALID ENTRY:", c, n, entry)
                raise SystemExit("S IS MALFORMED")

    print("Seed:", seed, "phi_init:", phi_init[:3])

    map_tree, chain = mcmc_fixed_snv(bulk_snvs=bulk_snvs, scrna_data=S, epsilon=0.001, 
                                              num_iterations=1000000, z=z, scrna_config=None, phi_init=phi_init)
    
    filename = f"chain{chain_id}.pkl"
    print(f"Saving chain {chain_id} to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(chain, f)
        
    with open(f"chain{chain_id}_map.pkl", "wb") as f:
        pickle.dump(map_tree, f)

    return chain




def analyze_convergence(chains):
    logp = np.array([[s["log_posterior"] for s in chain] for chain in chains])
    phi = np.array([[s["phi"] for s in chain] for chain in chains])
    #idata = az.from_dict(posterior={"log_posterior": logp, "phi": phi})

    #print("ESS:", az.ess(idata))
    #print("R-hat:", az.rhat(idata))
    #az.plot_trace(idata, var_names=["log_posterior"])
    #az.plot_trace(idata, var_names=["phi"]) 


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
    print("Inferred Phi: ", inferred_phi)

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
    #true_tree = Tree(os.path.join(tree0_dir, "tree.newick"), format=1)
    inferred_tree_newick = map_result["tree"].to_newick()
    #inferred_tree = Tree(inferred_tree_newick, format=1)

    #rf, max_rf, *_ = true_tree.robinson_foulds(inferred_tree)
    #if max_rf <= 0:
    #    print(f"Robinson-Foulds distance: {rf} (max_rf undefined: {max_rf})")
    #else:
    #    print(f"Robinson-Foulds distance: {rf} / {max_rf}")
    print("--------------------------------------------------------------------")


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
    print ("Reached Main")
    chain = run_mcmc_fixed_snvs(TREE0_DIR)
    #chain = run_parallel_chains(TREE0_DIR, num_chains=4, save_to_disk=True)
    

    # map_result = run_mcmc_single(TREE0_DIR)
    # print(map_result["tree"].to_newick())
    # analyze_performance(TREE0_DIR, map_result) 