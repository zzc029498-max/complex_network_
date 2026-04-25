# ===================== 1. Import Libraries =====================
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
import os

# ===================== 2. Core Parameter Settings =====================
# Network parameters (1000 nodes as required)
N = 1000

# Four target networks: ER <k>=4, ER <k>=6, BA <k>=4, BA <k>=6
network_configs = {
    "ER <k>=4": {"type": "ER", "k": 4},
    "ER <k>=6": {"type": "ER", "k": 6},
    "BA <k>=4": {"type": "BA", "k": 4},
    "BA <k>=6": {"type": "BA", "k": 6},
}

# Epidemic parameters
mu_list = [0.2, 0.4]  # Recovery probability
beta_range = np.arange(0, 0.31, 0.01)  # Infection probability: 0 to 0.3 step 0.01

# Monte Carlo simulation parameters (follows the attached PDF)
Nrep = 50                # Number of simulation repetitions
T_max = 1000             # Total time steps per run
T_trans = 900            # Transient steps (only last 100 steps used for stationary average)
rho0 = 0.05              # Initial infected fraction (50 infected nodes for N=1000)

# Output directory
save_path = "./SIS_simulation_results"
os.makedirs(save_path, exist_ok=True)

# ===================== 3. Network Generation =====================
def generate_network(N, config):
    """
    Generate ER or BA network and return adjacency matrix.
    Args:
        N: number of nodes
        config: dict with network type and average degree <k>
    Returns:
        adj_matrix: numpy adjacency matrix
        G: networkx graph object
    """
    net_type = config["type"]
    k = config["k"]

    if net_type == "ER":
        # Erdős–Rényi random graph: p = <k>/(N-1)
        p = k / (N - 1)
        G = nx.erdos_renyi_graph(N, p, seed=42)
    elif net_type == "BA":
        # Barabási–Albert scale-free network: <k> = 2*m
        m = k // 2
        G = nx.barabasi_albert_graph(N, m, seed=42)
    else:
        raise ValueError("Only ER and BA networks are supported")

    # Convert to dense adjacency matrix for Numba speed
    adj_matrix = nx.to_numpy_array(G, dtype=np.int32)
    # Save network structure
    np.save(f"{save_path}/{net_type}_k{k}_adj.npy", adj_matrix)
    return adj_matrix, G

# ===================== 4. SIS Monte Carlo Simulation (Numba Accelerated) =====================
@jit(nopython=True)
def sis_monte_carlo(adj_matrix, beta, mu, T_max, T_trans, rho0, Nrep):
    """
    SIS model Monte Carlo simulation (SYNCHRONOUS update as in PDF).
    Rules:
      1. Infected node recovers with probability mu
      2. Susceptible node is infected by each infected neighbor with probability beta
      3. Stationary density <rho> averaged over time and repetitions
    Args:
        adj_matrix: network adjacency matrix
        beta: infection probability
        mu: recovery probability
        T_max: total time steps
        T_trans: transient steps to discard
        rho0: initial fraction of infected nodes
        Nrep: number of independent runs
    Returns:
        rho_avg: stationary average infected fraction
    """
    N = adj_matrix.shape[0]
    rho_total = 0.0

    for rep in range(Nrep):
        # Initialize state: 0 = Susceptible, 1 = Infected
        state = np.zeros(N, dtype=np.int32)
        init_infected = np.random.choice(N, size=int(N * rho0), replace=False)
        state[init_infected] = 1
        rho_time_sum = 0.0

        for t in range(T_max):
            next_state = state.copy()  # Synchronous update

            # Step 1: Recovery process
            for i in range(N):
                if state[i] == 1:
                    if np.random.random() < mu:
                        next_state[i] = 0

            # Step 2: Infection process
            for i in range(N):
                if state[i] == 0:
                    # Check all neighbors; break if infected
                    for j in range(N):
                        if adj_matrix[i, j] == 1 and state[j] == 1:
                            if np.random.random() < beta:
                                next_state[i] = 1
                                break

            state = next_state

            # Accumulate stationary data
            if t >= T_trans:
                rho_t = np.sum(state) / N
                rho_time_sum += rho_t

        # Time average for one run
        rho_rep_avg = rho_time_sum / (T_max - T_trans)
        rho_total += rho_rep_avg

    # Average over all repetitions
    rho_avg = rho_total / Nrep
    return rho_avg

# ===================== 5. MMCA Theoretical Prediction (For Highest Grade) =====================
def sis_mmca(adj_matrix, beta, mu, max_iter=10000, tol=1e-8):
    """
    Microscopic Markov Chain Approach (MMCA) for SIS stationary state.
    Update equation:
    p_i(t+1) = (1-p_i) * [1 − product(1−β p_j)] + p_i*(1−μ)
    Args:
        adj_matrix: adjacency matrix
        beta: infection rate
        mu: recovery rate
    Returns:
        rho_mmca: average stationary infected density
    """
    N = adj_matrix.shape[0]
    p = np.ones(N) * 0.2

    for _ in range(max_iter):
        p_prev = p.copy()
        for i in range(N):
            prod = 1.0
            for j in range(N):
                if adj_matrix[i, j] == 1:
                    prod *= (1.0 - beta * p_prev[j])
            p[i] = (1.0 - p_prev[i]) * (1.0 - prod) + p_prev[i] * (1.0 - mu)

        if np.max(np.abs(p - p_prev)) < tol:
            break

    rho_mmca = np.mean(p)
    return rho_mmca

# ===================== 6. Main Simulation Loop =====================
if __name__ == "__main__":
    # Generate all networks
    networks = {}
    print("=== Generating Networks ===")
    for name, config in tqdm(network_configs.items()):
        adj, G = generate_network(N, config)
        networks[name] = adj
        avg_deg = np.mean(np.sum(adj, axis=1))
        print(f"{name} done | Average degree: {avg_deg:.2f}")

    # Run simulations and store results
    results = {}
    print("\n=== Starting Monte Carlo Simulations ===")

    for mu in mu_list:
        results[mu] = {}
        print(f"\n===== Recovery probability mu = {mu} =====")

        for net_name, adj in networks.items():
            print(f"Simulating network: {net_name}")
            rho_mc_list = []
            rho_mmca_list = []

            for beta in tqdm(beta_range):
                # Monte Carlo
                rho_mc = sis_monte_carlo(adj, beta, mu, T_max, T_trans, rho0, Nrep)
                rho_mc_list.append(rho_mc)
                # MMCA theory
                rho_mmca = sis_mmca(adj, beta, mu)
                rho_mmca_list.append(rho_mmca)

            # Save results
            results[mu][net_name] = {
                "beta": beta_range,
                "rho_mc": np.array(rho_mc_list),
                "rho_mmca": np.array(rho_mmca_list)
            }
            np.save(f"{save_path}/results_mu{mu}_{net_name}.npy", results[mu][net_name])

    # Plot results (four networks in one figure per mu)
    print("\n=== Generating Plots ===")
    plt.rcParams["figure.dpi"] = 300

    for mu in mu_list:
        plt.figure(figsize=(8, 6))
        for net_name in networks.keys():
            res = results[mu][net_name]
            # Monte Carlo (solid line with markers)
            plt.plot(res["beta"], res["rho_mc"],
                     marker="o", ms=3, lw=1.5, label=f"{net_name} (MC)")
            # MMCA (dashed line)
            plt.plot(res["beta"], res["rho_mmca"],
                     '--', lw=1, alpha=0.7, label=f"{net_name} (MMCA)")

        plt.xlabel("Infection probability β", fontsize=12)
        plt.ylabel("Steady-state infected fraction <ρ>", fontsize=12)
        plt.title(f"SIS Epidemic Spreading (N={N}, μ={mu})", fontsize=14)
        plt.legend(fontsize=9, loc="best")
        plt.grid(alpha=0.3, linestyle="--")
        plt.xlim(0, 0.3)
        plt.ylim(0, 1.0)
        # Annotate simulation parameters
        plt.text(0.02, 0.95,
                 f"Nrep={Nrep}, Tmax={T_max}, Ttrans={T_trans}",
                 fontsize=8, transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(f"{save_path}/SIS_results_mu{mu}.pdf")
        plt.savefig(f"{save_path}/SIS_results_mu{mu}.png")
        plt.close()

    print(f"\n=== All tasks completed. Results saved to: {save_path} ===")
