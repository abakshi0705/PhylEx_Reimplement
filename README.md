## Running PhylEx Reimplementations
To run the first version of the reimplementation (with default parameters):
1. Go to run_simulation.py
2. ensure line #453 is not commented; Run the simulation to generate 1 MCMC chain
3. In order to generate multiple chains, use run_mcmc.sh on BioHPC; to adjust the number of iterations and burnin, adjust num_iterations and burnin on line 329
Chains are saved as pickle files

To run the second version of the reimplementation (with default parameters):
1. Go to run_simulation.py
2. Ensure line #453 is commented and line #454 is uncommented; Run the simulation to generate 1 MCMC chain
3. In order to generate multiple chains, use run_mcmc.sh on BioHPC; to adjust number of iterations and burnin, adjust num_iterations and burnin on lines 259, 260
Chains are saved as pickle files

To analyze chain convergence:
1. Add pickle files to one directory
2. In analyze_convergence.py, adjust pickle_dir so that it reflects this directory
3. Run analyze_convergence.py

##Project Description

PhylEx is a Bayesian phylogenetic inference framework for reconstructing tumor evolution from bulk DNA sequencing and single-cell RNA sequencing (scRNA-seq) data. The method infers clonal tree structure (the phylogenetic tree representing tumor evolution), SNV assignments (which mutations belong to which clones), and clone prevalences (the cellular prevalence or frequency of each clone in the tumor).

The implementation uses Markov Chain Monte Carlo (MCMC) sampling with slice sampling for tree exploration and Metropolis-Hastings for clone prevalence updates.

## Project Structure

```
PhylEx_Reimplement-main/
├── PhylEx.py                    # Main MCMC inference engine
├── bulk_dna_likelihood.py       # Bulk DNA sequencing likelihood computation
├── scrna.py                     # Single-cell RNA-seq likelihood computation
├── clone_prevalences.py         # Clone prevalence sampling (Dirichlet prior)
├── TSSB.py                      # Tree-Structured Stick-Breaking prior
├── run_simulation.py            # Example simulation and evaluation script
├── tests/                       # Test suites
│   ├── test_scrna.py
│   ├── test_phi_sample.py
│   └── test_tssb.py
└── cherry1/                     
```

## Core Components

### 1. bulk_dna_likelihood.py - Bulk DNA Sequencing Model

This module computes the likelihood of bulk DNA sequencing data given clone assignments and prevalences.

#### Classes

SNV: Single SNV in bulk data. Contains variant_reads (number of reads supporting the variant allele), total_reads (total read depth), major_cn (the major copy numbers), minor_cn (the minor copy numbers), and clone_index (which clone this SNV is assigned to z_n).

Genotype: Genotype with total copies c and variant copies v. So for instance if c = 2, v = 1 then we have one reference and one variant.

#### Key Functions

enumerate_genotypes(major_cn, minor_cn): Return all genotypes for given major/minor copy numbers. Assumes that total copy c is the sum of major and minor. Then for this c, considers all possible variant copy counts v which puts a uniform prior over v given c.

theta(genotype, phi_clone, epsilon): Per-read variant probability theta(g, phi, epsilon) for one genotype. From the paper: if nu(gn) = 0, returns epsilon. If nu(gn) = c(gn), returns phi(1-epsilon) + (1-phi)epsilon. Otherwise returns phi(nu(gn)/c(gn)) + (1-phi)epsilon. If v == c, the SNV is present in all copies and is a mixture of tumor clone (1 - epsilon) and epsilon weighted by phi_clone. For the intermediate case 0 < v < c, within the clone the expected variant fraction is v/c.

log_binomial_pmf(k, n, p): Log Binomial(n, p) at k computed stably. Implements C(n, k) + k log p + (n - k) log(1 - p) using lgamma for the log factorial terms. If no trials (n == 0), returns 0.0 if k == 0 else -inf. Out of range k gives zero prob in Bin. Makes sure not log(0) by clipping p.

logsumexp(values): Stable logsumexp over a list of log values.

snv_log_likelihood(snv, phi, epsilon): Computes log P(b_n | d_n, M_n, m_n, phi_{z_n}, epsilon) for one SNV. Computes log P(b_n | d_n, M_n, m_n, phi_{z_n}, epsilon) = log Bin(d_n, b_n; theta). Assumes a uniform prior over genotypes.

bulk_log_likelihood(snvs, phi, epsilon): Total bulk log-likelihood log p(B | T, z, phi) = sum over n P(b_n | d_n, M_n, m_n, phi_{z_n}, epsilon). Sums per SNV log likelihoods in the bulk dataset.

### 2. scrna.py - Single-Cell RNA Sequencing Model

This file computes the scRNA-seq likelihood term log p(S | T, z, phi). It models allelic imbalance using a Beta-Binomial mixture, dropout (zero reads), background noise for SNVs not present in a clone, and cell-to-clone marginalization using clone prevalences phi.

#### Classes

ScRNALikelihoodParams: Holds the hyperparameters for the Beta-Binomial mixture model. These are the hyperparameters that were provided in the supplementary file. These define mono-allelic expression, bi-allelic expression, and background/error distribution.

#### Key Functions

logsumexp(values): Computes the sum of log probabilities in a list (function is same as one in bulk_dna_likelihood).

log_beta_binomial(b, d, alpha, beta): Compute the log of the Beta-Binomial PMF: C(d, b) * Beta(b + alpha, d - b + beta) / Beta(alpha, beta). Log of the binomial coefficient C(d, b) - equivalent to d choose b in log space. Computes log Beta(b+alpha, d-b+beta) - log Beta(alpha, beta).

log_mutated_mixture(b, d, params): Likelihood for mutated and non-mutated SNVs. Compute the log likelihood for a mutated SNV in a clone. Uses a mixture of two Beta-Binomial distributions: mono-allelic expression and bi-allelic expression.

log_background(b, d, params): Compute the log likelihood for an SNV that is not mutated in this clone. This acts as a noise/error model.

log_likelihood_cell_given_clone(cell_index, S, clone_k, clone_has_snv, params): Likelihood of a single cell assuming it belongs to 1 clone. Compute log P(S_cell | clone = k). Loops over all SNVs and uses the mixture model if mutation is present in clone k, or uses background noise if mutation is absent.

log_scrna_likelihood(S, phi, clone_has_snv, params): Full scRNA-seq likelihood with clone marginalization. Compute log p(S | T, z, phi) by marginalizing over which clone each cell may have come from. For each cell c: P(S_c | T, z, phi) = sum_k phi[k] * P(S_c | clone k). Returns the sum of log P(S_c | ...) over all cells. Collects log(phi[k] * likelihood of cell c in clone k) and marginalizes over all clone options via log-sum-exp.

### 3. clone_prevalences.py - Clone Prevalence Sampling

This module handles sampling clone prevalences phi using a Dirichlet prior and Metropolis-Hastings updates.

#### Classes

PhiSample: This class handles sampling clone prevalences in the PhylEx model. Tree contains the clone structure and number of clones. Alpha is the Dirichlet prior concentration parameter. Dirichlet(1) is the standard non-informative prior so we set alpha=1.0 for each clone. Keeps the original alpha parameter and constructs alpha vectors on demand. Validates alpha (must be > 0).

#### Key Functions

dirichlet_log(phi, alpha): Computing the log density of a dirichlet distribution, using the formula: log p(phi | alpha) = lgamma(sum(alpha)) - sum(lgamma(alpha_i)) + sum((alpha_i -1) * log(phi_1)). This is the prior over clone prevalences. Prevents log(0) by clipping phi.

PhiSample.prior_sample(): Initial sample from dirichlet prior. Samples using the current tree size and ensures concentrations are strictly positive.

PhiSample.prior_log(phi): Computing the Dirichlet log prior for a given phi. Ensures alpha vector matches phi length and is strictly positive.

PhiSample.sample_prior(): Drawing fresh phi from Dirichlet prior.

PhiSample.propose(phi, step=50): Propose a new clone prevalence vector by sampling from a Dirichlet distribution centered around the current phi. Scales phi by a step factor so that most proposals stay close to the current value, which helps the MCMC explore the space smoothly without making huge jumps. Forms proposal concentration parameters centered on current phi. Ensures all entries > 0 for Dirichlet sampling.

PhiSample.update(phi, snvs, epsilon, S, clone_has_snv): Run one update step of the Metropolis Hastings sampler for phi. Generate a new phi and compute its posterior probability using the bulk DNA likelihood, scRNA likelihood, and the Dirichlet prior over phi. The proposal is accepted with the standard MH acceptance probability. Parameters: phi is current prevalence, snvs is list of snv objects providing bulk read information, epsilon is bulk sequencing noise level, S is per-cell scRNA seq read counts, clone_has_snv is boolean matrix including mutation inheritance.

PhiSample.maptophi(z): Given an assignment vector z[n] = clone index of SNV n, return an array snv_phi[n]= phi[z[n]].

### 4. TSSB.py - Tree-Structured Stick-Breaking Prior

Reimplementation of the TSSB algorithm to generate a prior for the distribution of clonal trees and assign SNVs to specific clones in the tree.

#### Classes

Node: Represents a node in the phylogenetic tree. Parent is the parent of this node. upsilon_u is the portion of the stick available to be broken up by the node. remaining_stick is the portion of the stick that remains after this node breaks off its piece. is_root indicates if the node is the root or not. height is the height of this node in the tree. snvs are snvs assigned to this node; snvs assigned after node has been created. pi_u = upsilon_u - remaining_stick.

Simplified_Node: Simplified node structure for fixed tree initialization.

#### Key Functions

get_node_list(lamb_0, lamb, gamma): Gets list of nodes for a specific tree, based on parameters lamb_0, lamb, gamma. Requires 0 < lamb_0, 0 < lamb <= 1, and gamma > 0. Initializes the root node as the base node. The root node is the healthy non-cancerous cell and has exactly one child, which is the first cancer (progenitor) cell. Initializes the progenitor cell. upsilon represents the proportion of the stick able to be broken by a node; this should equal 1 for the progenitor. The progenitor is a cancer clone (not the healthy root), so is_root should be False.

tssb(node, lamb_0, lamb, gamma): Recursive helper function that generates all the children of a given node, and those childrens' nodes until remaining stick is smaller than a threshold value (0.005). Requires 0 < lamb_0, 0 < lamb <= 1, and gamma > 0.

assign_snvs(n, node_list): Assign SNVs to a clone. n is the number of snvs. The probability of an snv being assigned to a clone is equal to that clones pi value, which was found in the tssb algorithm. Randomly assigns a node based on the pi_value for that node to assign an SNV to. The index of z represents SNV number.

get_snvs_for_node(z, node_index): Gets the snvs for a specific node index.

get_node_genotypes(node_list, z, node_index): Gets a list of all SNV indices present at a specific node, including the ancestral SNVs.

initialize_fixed_tree_snv_assignment(node_list, lamb_0, lamb, gamma): Initializes pi values for a fixed tree structure.

### 5. PhylEx.py - Main Inference Engine

This is the core module that orchestrates the MCMC inference.

#### Classes

Tree: Represents the phylogenetic tree. node_list is the full list of all nodes in the tree, including the non-cancerous root node. snvs is the SNV-to-node assignment vector (z). nodes_except_root are all the nodes in the tree excluding the non-cancerous root node. k is the number of cancer clones. node_index_full_tree(clone_index) returns the index of a node in the full tree, given a clone index. A clone index would be an index from self.nodes_except_root. to_newick() converts tree to Newick format string.

TreeSampler: Class for slice sampling a tree; we use this to explore the tree space. slice_sample_tree() samples new SNV assignments using slice sampling. This list will hold all the new snv assignments that we make. sample_snv_assignment() samples assignment for a single SNV. Finds the node that this SNV is currently assigned to. Calculates the current log probability of this snv. We slice sample u ~ uniform(0, current_prob). Mathematically, this is some value x * current_prob where x is between 0 and 1. Taking the log of this, we get log(x) + current_log_prob. Will hold all nodes where new_log_prob > log_u. If there are no valid nodes, ie the currently assigned node is the only possible node for this SNV, returns the currently assigned node. log_prob_snv() computes log probability of assigning SNV to a node. Creates an adjusted snv assignment list where the snv of interest is assigned to the node of interest. The node of interest is the node which is currently being looped over in sample_snv_assignment. Gets the probability for the snv being assigned to this node. Adjusts the snv assignment list so that it represents clone indices (excludes root indices). prune_empty_nodes() removes any nodes with no SNVs assigned to it and whose descendants have no snvs assigned to them.

#### Key Functions

adjust_z(z, tree): Adjust z so that it represents clone indices in the tree (basically the indices in tree.nodes_except_root). For example, if a node has an index of x in the real tree, the index of the clone should be x-1.

make_clone_has_snv_matrix(z, tree, num_snvs): Returns a matrix containing information on if a clone has an SNV or not. clone_has_snv is a k x n matrix where n = number of snvs and k is the number of clones. If a clone has an snv, or any of its ancestor nodes have an snv, then clone_has_snv[k][n] = True. Otherwise clone_has_snv[k][n] = False.

check_if_descendant(current_node, ancestor_node): Helper method that checks whether a specific node is the descendant of another node. Descendants may not be the direct child of the node, but can be the grandchild or great grandchild etc.

update_bulk_snvs_indices(bulk_snvs, z_updated): Updates SNV assignment for bulk DNA data.

initialize_prior_tree(lamb_0, lamb, gamma, num_snvs, alpha, scrna_params): Initializes tree from TSSB prior.

initialize_prior_fixed_tree(...): Initializes a fixed 3-node tree (root, progenitor, two children) for testing.

compute_log_posterior(phi, bulk_snvs, scrna_data, clone_has_snv, epsilon, scrna_config, phi_sampler): Computes full log posterior = log_bulk + log_scrna + log_prior.

mcmc(bulk_snvs, scrna_data, lamb_0, lamb, gamma, epsilon, num_iterations, burnin, use_fixed_tree, scrna_config): Main MCMC inference function. Initializes tree and clone prevalences. For each iteration: updates clone prevalences phi using Metropolis-Hastings, updates SNV assignments z using slice sampling, updates tree structure (if not using fixed tree), tracks maximum a posteriori (MAP) estimate. Returns MAP tree, phi, z, and log posterior.

### 6. run_simulation.py - Example Usage and Evaluation

This script demonstrates how to load bulk SNV data from files, run MCMC inference, evaluate performance against ground truth, and compare inferred vs. true trees using Robinson-Foulds distance.

#### Key Functions

load_bulk_snvs(tree0_dir): Load bulk SNVs from genotype_ssm.txt, which has columns: ID, CHR, POS, REF, ALT, b, d, major_cn, minor_cn. Reads variant_reads from column b and total_reads from column d. Copy numbers fall back to defaults if not present.

load_true_snvs(tree0_dir): True SNV→clone assignments derived from datum2node.tsv by grouping identical genotype strings into clusters. Factorizes genotype strings into cluster IDs 0..K-1, then makes them 1..K. Order matches sample order (s0, s1, ...) which matches genotype_ssm.

load_true_phi(tree0_dir): True clone prevalences from cellular_prev.csv + datum2node.tsv. phi_k = mean cellular_prev over all SNVs assigned to clone k, normalized.

load_phi(tree0_dir): Wrapper in case we need ground-truth phi elsewhere.

run_mcmc_single(tree0_dir): Runs MCMC on a dataset.

analyze_performance(tree0_dir, map_result): Evaluates SNV assignment accuracy and Adjusted Rand Index (ARI), clone prevalence RMSE, and tree topology using Robinson-Foulds distance. ARI is label-invariant, so fine even if label IDs differ. Pads shorter vector with zeros so RMSE is always defined.

build_cluster_labels_from_datum2node(tree0_dir, out_name): Read datum2node.tsv (sample, genotype string) and turn it into a cluster_labels-style file: (sample, cluster_id). Each unique genotype string gets its own cluster_id.

## Dependencies

numpy
pandas
scikit-learn (for adjusted_rand_score)
ete3 (for tree comparison - Robinson-Foulds)
matplotlib (for plotting, optional)

## Data Format

### Bulk DNA Data (genotype_ssm.txt)

Tab-separated file with columns: ID (SNV identifier), CHR (chromosome), POS (position), REF (reference allele), ALT (alternate allele), b (variant reads/variant_reads), d (total reads/total_reads), major_cn (major copy number), minor_cn (minor copy number).

### Ground Truth Files

datum2node.tsv: Maps samples to genotype strings (for true SNV assignments)

cellular_prev.csv: True cellular prevalences per sample

tree.newick: True phylogenetic tree in Newick format

## Algorithm Overview

1. Initialization: Generate initial tree from TSSB prior and assign SNVs randomly.

2. MCMC Iterations:
   - Clone Prevalence Update: Metropolis-Hastings step proposing new phi from Dirichlet centered on current phi
   - SNV Assignment Update: Slice sampling to reassign each SNV to a node
   - Tree Structure Update: (If not fixed) Tree can grow/shrink based on SNV assignments

3. Likelihood Computation:
   - Bulk DNA: Binomial likelihood with genotype enumeration and copy number modeling
   - scRNA-seq: Beta-Binomial mixture with allelic imbalance modeling
   - Prior: Dirichlet prior on clone prevalences

4. MAP Tracking: Throughout MCMC, track the configuration with highest log posterior.

## Key Parameters

lamb_0: TSSB base rate (controls tree branching). Requires 0 < lamb_0.

lamb: TSSB decay parameter (controls depth). Requires 0 < lamb <= 1.

gamma: TSSB stick-breaking parameter (controls node probabilities). Requires gamma > 0.

epsilon: Sequencing error rate (typically 0.001).

alpha: Dirichlet concentration for clone prevalences (default: 1.0). Dirichlet(1) is the standard non-informative prior.

num_iterations: Number of MCMC iterations.

burnin: Number of burn-in iterations (currently not used, but tracked).

## Unit testing for individual components

python tests/test_scrna.py
python tests/test_phi_sample.py
python tests/test_tssb.py

## other

The implementation assumes bulk DNA and scRNA-seq data are aligned to the same SNV positions. Copy number information (major_cn, minor_cn) is required for accurate bulk DNA likelihood computation. The scRNA-seq model handles dropout (zero reads) gracefully. Tree structure can be fixed (for testing) or allowed to vary during inference.


