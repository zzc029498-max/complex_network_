import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import defaultdict

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def calculate_community_membership(graph, algorithm, weights=None):
    """
    Calculate community membership using a specified community detection algorithm.
    
    Args:
        graph: igraph Graph object, the target network for analysis
        algorithm: Name of the algorithm, supported options: 'infomap', 'louvain', 'leiden', 'girvan_newman'
        weights: Edge weight array (for weighted graph analysis, optional)
    
    Returns:
        VertexClustering object with community assignments for each node
    """
    if algorithm == 'infomap':
        comm = graph.community_infomap(edge_weights=weights)
    elif algorithm == 'louvain':
        comm = graph.community_multilevel(weights=weights)  # Louvain modularity maximization
    elif algorithm == 'leiden':
        comm = graph.community_leiden(objective_function='modularity', weights=weights)  # Optimized Louvain
    elif algorithm == 'girvan_newman':
        # Girvan-Newman: hierarchical edge betweenness clustering, select optimal partition
        dendrogram = graph.community_edge_betweenness(weights=weights)
        comm = dendrogram.as_clustering()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return comm

def compute_similarity_metrics(comm_true, comm_pred):
    """
    Compute standard partition similarity metrics between ground-truth and detected communities.
    Metrics: Jaccard Index, Arithmetic Normalized Mutual Information (NMI), Normalized Variation of Information (NVI)
    
    Args:
        comm_true: Ground-truth VertexClustering object
        comm_pred: Algorithm-detected VertexClustering object
    
    Returns:
        Tuple (jaccard, nmi, nvi) of computed metrics
    """
    # Calculate NMI and Variation of Information with igraph's built-in functions
    nmi = ig.compare_communities(comm_true, comm_pred, method='nmi')
    vi = ig.compare_communities(comm_true, comm_pred, method='vi')
    # Normalize VI by log2(N) to fit [0,1] range
    nvi = vi / np.log2(len(comm_true.membership))

    # Calculate Pairwise Jaccard Index (community detection standard implementation)
    membership_true = np.array(comm_true.membership)
    membership_pred = np.array(comm_pred.membership)
    n_nodes = len(membership_true)
    
    # Create boolean matrices for same-community node pairs
    same_community_true = (membership_true[:, None] == membership_true[None, :])
    same_community_pred = (membership_pred[:, None] == membership_pred[None, :])
    
    # Exclude self-pairs (diagonal values)
    np.fill_diagonal(same_community_true, False)
    np.fill_diagonal(same_community_pred, False)
    
    # Jaccard = Intersection / Union of same-community pairs
    intersection = np.sum(same_community_true & same_community_pred)
    union = np.sum(same_community_true | same_community_pred)
    jaccard = intersection / union if union != 0 else 0.0

    return jaccard, nmi, nvi

def save_clu_file(membership, output_path):
    """
    Save community membership results to standard Pajek .clu format (1-based indexing).
    This matches the assignment's required output format.
    
    Args:
        membership: List of community labels for each node (0-based)
        output_path: Full path to save the .clu file
    """
    with open(output_path, 'w') as f:
        f.write(f"*Vertices {len(membership)}\n")
        for label in membership:
            f.write(f"{label + 1}\n")  # Convert to 1-based indexing for Pajek standard

# ------------------------------------------------------------------------------
# Task 1: Community Structure Analysis for SBM Synthetic Networks
# ------------------------------------------------------------------------------
def task1_analyze_synthetic_networks(synthetic_dir, output_dir):
    """
    Complete Task 1 workflow: analyze synthetic SBM networks, run community detection,
    compute performance metrics, generate trend plots, and save all results.
    
    Args:
        synthetic_dir: Path to folder containing synthetic .net network files
        output_dir: Root path to save all Task 1 outputs
    """
    # Create output directories for organized results
    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, "task1_figures")
    result_dir = os.path.join(output_dir, "task1_results")
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Fixed SBM parameters from the assignment
    TOTAL_NODES = 300
    N_BLOCKS = 5
    NODES_PER_BLOCK = TOTAL_NODES // N_BLOCKS
    # Build ground-truth community membership (5 blocks of 60 nodes each)
    ground_truth_membership = []
    for block_id in range(N_BLOCKS):
        ground_truth_membership.extend([block_id] * NODES_PER_BLOCK)

    # Load and sort all synthetic network files by prr value
    network_files = [f for f in os.listdir(synthetic_dir) if f.endswith(".net")]
    prr_values = []
    for filename in network_files:
        # Extract prr value from filename (format: prr_0.04)
        filename_parts = filename.split("_")
        prr_index = filename_parts.index("prr") + 1
        prr_values.append(float(filename_parts[prr_index]))
    
    # Sort files and prr values in ascending order for consistent analysis
    sorted_indices = np.argsort(prr_values)
    network_files = [network_files[i] for i in sorted_indices]
    prr_values = [prr_values[i] for i in sorted_indices]

    # Load prr=1.00 network first: for fixed layout and ground-truth graph binding
    prr1_file = next((f for f, prr in zip(network_files, prr_values) if np.isclose(prr, 1.0)), None)
    if not prr1_file:
        raise ValueError("Required prr=1.00 network file not found in the input directory")
    g_prr1 = ig.Graph.Read_Pajek(os.path.join(synthetic_dir, prr1_file))
    fixed_layout = g_prr1.layout_kamada_kawai()  # Fixed layout for all visualizations

    # Create ground-truth VertexClustering object (bound to the graph for metric calculation)
    comm_ground_truth = ig.VertexClustering(g_prr1, membership=ground_truth_membership)

    # Define algorithms per assignment requirements: 1 Infomap + 2 modularity maximization algorithms
    algorithms = ["infomap", "louvain", "leiden"]
    result_storage = defaultdict(list)

    # Process each synthetic network across all prr values
    for net_file, prr in zip(network_files, prr_values):
        print(f"[Task 1] Processing prr = {prr:.2f} ({net_file})")
        # Load the target network
        graph = ig.Graph.Read_Pajek(os.path.join(synthetic_dir, net_file))

        # Run each selected community detection algorithm
        for algo in algorithms:
            # Detect communities
            detected_comm = calculate_community_membership(graph, algo)
            # Save community results to Pajek .clu format
            save_clu_file(
                detected_comm.membership,
                os.path.join(result_dir, f"comm_{algo}_prr_{prr:.2f}.clu")
            )
            # Calculate similarity metrics against ground truth
            jaccard, nmi, nvi = compute_similarity_metrics(comm_ground_truth, detected_comm)
            # Store all results for trend analysis
            result_storage["prr"].append(prr)
            result_storage["algorithm"].append(algo)
            result_storage["n_communities"].append(len(detected_comm))
            result_storage["modularity"].append(detected_comm.modularity)
            result_storage["jaccard"].append(jaccard)
            result_storage["nmi"].append(nmi)
            result_storage["nvi"].append(nvi)

    # Save full results to CSV for report writing
    results_df = pd.DataFrame(result_storage)
    results_df.to_csv(os.path.join(result_dir, "synthetic_network_results.csv"), index=False)

    # Generate trend plots for all required metrics
    target_metrics = ["n_communities", "modularity", "jaccard", "nmi", "nvi"]
    for metric in target_metrics:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=results_df,
            x="prr",
            y=metric,
            hue="algorithm",
            marker="o",
            linewidth=2
        )
        plt.xlabel("Intra-block connection probability (prr)", fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.title(f"Evolution of {metric.replace('_', ' ')} across prr values", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f"{metric}_trend.png"), dpi=300)
        plt.close()

# ------------------------------------------------------------------------------
# Task 2: Community Structure Analysis for Real Primary School Contact Network
# ------------------------------------------------------------------------------
def task2_analyze_real_network(real_data_dir, output_dir):
    """
    Complete Task 2 workflow: analyze weighted/unweighted primary school contact networks,
    run modularity-based community detection, analyze community composition,
    and save all results for the assignment report.
    
    Args:
        real_data_dir: Path to folder containing primary school network and metadata files
        output_dir: Root path to save all Task 2 outputs
    """
    # Create output directories for organized results
    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, "task2_figures")
    result_dir = os.path.join(output_dir, "task2_results")
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Load unweighted and weighted versions of the primary school network
    graph_unweighted = ig.Graph.Read_Pajek(os.path.join(real_data_dir, "primaryschool_u.net"))
    graph_weighted = ig.Graph.Read_Pajek(os.path.join(real_data_dir, "primaryschool_w.net"))
    edge_weights = np.array(graph_weighted.es["weight"])  # Extract contact duration weights

    # Robust metadata loading and cleaning (fixes length mismatch and type errors)
    # Load metadata: node_id (1-based) + school group assignment
    metadata = pd.read_csv(
        os.path.join(real_data_dir, "metadata_primary_school.txt"),
        sep=r"\s+",
        header=None,
        names=["node_id", "school_group"]
    )
    # Convert node_id to numeric, set invalid values to NaN
    metadata["node_id"] = pd.to_numeric(metadata["node_id"], errors='coerce')
    # Drop rows with invalid/missing node_id (clean bad data/headers)
    metadata = metadata.dropna(subset=["node_id"])
    # Convert node_id to integer type
    metadata["node_id"] = metadata["node_id"].astype(int)
    # Convert to 0-based vertex_id to match igraph's node indexing
    metadata["vertex_id"] = metadata["node_id"] - 1
    # Filter to only include valid vertex IDs present in the network
    total_nodes = graph_weighted.vcount()
    metadata = metadata[metadata["vertex_id"].between(0, total_nodes - 1)]
    # Sort by vertex_id to match igraph's node order exactly
    metadata = metadata.sort_values("vertex_id").reset_index(drop=True)

    # Run community detection per assignment requirements (modularity-based Louvain algorithm)
    selected_algo = "louvain"
    comm_unweighted = calculate_community_membership(graph_unweighted, selected_algo)
    comm_weighted = calculate_community_membership(graph_weighted, selected_algo, weights=edge_weights)

    # Save community results to Pajek .clu format
    save_clu_file(
        comm_unweighted.membership,
        os.path.join(result_dir, f"comm_real_unweighted_{selected_algo}.clu")
    )
    save_clu_file(
        comm_weighted.membership,
        os.path.join(result_dir, f"comm_real_weighted_{selected_algo}.clu")
    )

    # --------------------------
    # NEW: Generate Fixed Layout (Kamada-Kawai with inverse weights)
    # --------------------------
    inv_weights = 1 / (edge_weights + 1e-6)  # Avoid division by zero
    fixed_layout = graph_weighted.layout_kamada_kawai(weights=inv_weights)
    layout_coords = np.array(fixed_layout.coords)  # Convert to numpy array for matplotlib

    # --------------------------
    # NEW: Task 2 Visualization (Unweighted vs Weighted)
    # --------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # --- Plot 1: Unweighted Network ---
    membership_u = np.array(comm_unweighted.membership)
    unique_comms_u = np.unique(membership_u)
    palette_u = plt.cm.tab20(np.linspace(0, 1, len(unique_comms_u)))
    color_map_u = {c: tuple(palette_u[i]) for i, c in enumerate(unique_comms_u)}
    colors_u = [color_map_u[m] for m in membership_u]
    
    ax1.scatter(layout_coords[:, 0], layout_coords[:, 1], s=100, c=colors_u, edgecolors='k', alpha=0.8)
    ax1.set_title("Unweighted Network", fontsize=16)
    ax1.axis('off')

    # --- Plot 2: Weighted Network ---
    membership_w = np.array(comm_weighted.membership)
    unique_comms_w = np.unique(membership_w)
    palette_w = plt.cm.tab20(np.linspace(0, 1, len(unique_comms_w)))
    color_map_w = {c: tuple(palette_w[i]) for i, c in enumerate(unique_comms_w)}
    colors_w = [color_map_w[m] for m in membership_w]
    
    ax2.scatter(layout_coords[:, 0], layout_coords[:, 1], s=100, c=colors_w, edgecolors='k', alpha=0.8)
    ax2.set_title("Weighted Network", fontsize=16)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "real_network_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Add detected community labels to metadata (for composition analysis)
    valid_vertex_ids = metadata["vertex_id"].values
    metadata["detected_community"] = [comm_weighted.membership[v] for v in valid_vertex_ids]

    # Generate and save community composition contingency table
    composition_table = pd.crosstab(metadata["detected_community"], metadata["school_group"])
    composition_table.to_csv(os.path.join(result_dir, "community_school_group_composition.csv"))

    # Generate stacked bar plot for community composition (required for the report)
    composition_table.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        colormap="viridis"
    )
    plt.xlabel("Detected Community", fontsize=12)
    plt.ylabel("Number of Individuals", fontsize=12)
    plt.title("Community Composition by School Group (Weighted Network)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "community_composition_stacked_bar.png"), dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# Main Execution Workflow
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------
    # USER CONFIGURATION - PATHS
    # --------------------------
    SYNTHETIC_DATA_FOLDER = "./A3_synthetic_networks"
    REAL_DATA_FOLDER = "./A3_primary_school_network"
    OUTPUT_ROOT_FOLDER = "./community_detection_assignment_results"

    # Run full assignment workflow
    print("=" * 60)
    print("Community Detection Assignment - Full Analysis")
    print("=" * 60)
    
    task1_analyze_synthetic_networks(SYNTHETIC_DATA_FOLDER, OUTPUT_ROOT_FOLDER)
    print("\n" + "-" * 60 + "\n")
    task2_analyze_real_network(REAL_DATA_FOLDER, OUTPUT_ROOT_FOLDER)
    
    print("\n✅ ALL TASKS COMPLETED SUCCESSFULLY!")
    print("All results saved to:", OUTPUT_ROOT_FOLDER)
    print("=" * 60)