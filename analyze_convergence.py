import pickle
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import os


plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


def analyze_conv():
    pickle_dir = "pickle_files_2"
    chains = []
    log_post_chains = []
    for i in range(0, 8):
        fname = os.path.join(pickle_dir, f"chain{i}.pkl")
        with open(fname, "rb") as f:
            print("loaded: " +  fname)
            chains.append(pickle.load(f))

    num_chains = len(chains)
    num_iters = len(chains[0])

    posterior = {}

    for key in ["log_posterior", "log_bulk", "log_scrna", "log_prior"]:
        posterior[key] = np.array([
            [draw[key] for draw in chain]
            for chain in chains
        ])  # shape: (chains, draws)

    # Vector parameter: phi
    phi_dim = len(chains[0][0]["phi"])

    posterior["phi"] = np.array([
        [draw["phi"] for draw in chain]
        for chain in chains
    ])  

    coords = {"phi_dim": np.arange(phi_dim)}

    dims = {"phi": ["phi_dim"]}

    idata = az.from_dict(posterior=posterior, coords=coords, dims=dims)

    idata = idata.sel(draw=slice(200000, None))

    rhat = az.rhat(idata)
    ess = az.ess(idata)
    print(az.summary(idata))
    print("\nR-hat:\n", rhat)
    print("\nESS:\n", ess)

    az.summary(idata).to_csv(os.path.join(pickle_dir, "summary.csv"))

    axes = az.plot_trace(
        idata,
        var_names=["log_posterior", "phi"],
        figsize=(16, 10),
        compact=False,
        divergences=None,
        backend_kwargs={
            'constrained_layout': True
        }
    )
    
    fig = plt.gcf()  # Get current figure
    
    # Improve x-axis formatting for trace plots
    for ax in fig.get_axes():
        if ax.get_xlabel():  # Only for plots with x-axis labels
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    fig.suptitle('MCMC Trace and Posterior Distributions', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(os.path.join(pickle_dir, "trace.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


    axes = az.plot_rank(
        idata,
        var_names=["phi"],
        kind='vlines',  # Can be 'bars' or 'vlines'
    )
    
    fig = plt.gcf()
    fig.set_size_inches(14, 6)
    
    plt.suptitle('MCMC Rank Plots (Phi Parameters)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(os.path.join(pickle_dir, "rank.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    
    # 1. Autocorrelation plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    az.plot_autocorr(idata, var_names=["log_posterior"], 
                     combined=True, ax=axes[0])
    axes[0].set_title('Autocorrelation: Log Posterior', fontweight='bold')
    
    # Plot autocorr for first few phi dimensions
    az.plot_autocorr(idata, var_names=["phi"], 
                     coords={"phi_dim": [0, 1, 2]}, 
                     combined=True, ax=axes[1])
    axes[1].set_title('Autocorrelation: Phi (first 3 dimensions)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(pickle_dir, "autocorr.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 2. ESS plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    az.plot_ess(idata, var_names=["phi"], kind="local", ax=axes[0])
    axes[0].set_title('ESS Local', fontweight='bold')
    
    az.plot_ess(idata, var_names=["phi"], kind="quantile", ax=axes[1])
    axes[1].set_title('ESS Quantile', fontweight='bold')
    
    fig.suptitle('Effective Sample Size Diagnostics', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(os.path.join(pickle_dir, "ess_diagnostic.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 3. Forest plot for phi parameters
    if phi_dim <= 20:  # Only if not too many dimensions
        fig, ax = plt.subplots(figsize=(10, max(6, phi_dim * 0.4)))
        az.plot_forest(idata, var_names=["phi"], combined=True, ax=ax)
        ax.set_title('Posterior Distributions: Phi Parameters', 
                     fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(pickle_dir, "forest.png"), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # 4. Summary plot showing R-hat and ESS
    summary_df = az.summary(idata)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # R-hat plot
    rhat_vals = summary_df['r_hat'].values
    axes[0].hist(rhat_vals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(1.01, color='red', linestyle='--', linewidth=2, label='Threshold (1.01)')
    axes[0].set_xlabel('R-hat Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of R-hat Values', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ESS bulk plot
    ess_bulk_vals = summary_df['ess_bulk'].values
    axes[1].hist(ess_bulk_vals, bins=30, edgecolor='black', alpha=0.7, color='forestgreen')
    axes[1].axvline(400, color='red', linestyle='--', linewidth=2, label='Minimum (400)')
    axes[1].set_xlabel('ESS Bulk')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of ESS Bulk', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle('Convergence Diagnostics Summary', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(os.path.join(pickle_dir, "diagnostics_summary.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n✓ All plots saved to {pickle_dir}/")
    print(f"  - trace.png (trace plots)")
    print(f"  - rank.png (rank plots)")
    print(f"  - autocorr.png (autocorrelation)")
    print(f"  - ess_diagnostic.png (ESS diagnostics)")
    print(f"  - forest.png (forest plot, if phi_dim <= 20)")
    print(f"  - diagnostics_summary.png (R-hat and ESS distributions)")


    phi = idata.posterior["phi"]
    phi_pooled = phi.stack(sample=("chain", "draw")).values

    logp = idata.posterior["log_posterior"].stack(
        sample=("chain", "draw")).values  

    map_idx = np.argmax(logp)
    phi_map = [phi_pooled[0][map_idx], phi_pooled[1][map_idx], phi_pooled[2][map_idx]]

    def aitchison_distance(phi_true, phi_est, eps=1e-10):
        def clr(x):
            return np.log(x) - np.mean(np.log(x))

        return np.linalg.norm(clr(phi_true) - clr(phi_est))

    phi_true = np.array([0.5, 0.25, 0.125])
    phi_true /= phi_true.sum()

    print ("phi map:", phi_map)

    d_aitch = aitchison_distance(phi_true, phi_map)
    print ("Aitchison Distance: ", d_aitch)
    
    return rhat, ess




if __name__ == "__main__":
    analyze_conv()
