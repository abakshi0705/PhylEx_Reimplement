import pickle
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_conv():
    pickle_dir = "pickle_files"
    chains = []
    for i in range(0, 5):
        fname = os.path.join(pickle_dir, f"chain{i}.pkl")
        with open(fname, "rb") as f:
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

    idata = idata.sel(draw=slice(2000, None))

    rhat = az.rhat(idata)
    ess = az.ess(idata)
    print(az.summary(idata))
    print("\nR-hat:\n", rhat)
    print("\nESS:\n", ess)

    az.summary(idata).to_csv(os.path.join(pickle_dir, "summary.csv"))

    az.plot_trace(idata, var_names=["log_posterior", "phi"])
    plt.savefig(os.path.join(pickle_dir, "trace.png"))
    plt.close()

    az.plot_rank(idata, var_names=["phi"])
    plt.savefig(os.path.join(pickle_dir, "rank.png"))
    plt.close()
    
    return rhat, ess


if __name__ == "__main__":
    analyze_conv()
