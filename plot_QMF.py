import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration (matches your original code, no changes needed)
save_path = "./SIS_simulation_results"
mu_list = [0.2, 0.4]
network_configs = {
    "ER <k>=4": {"type": "ER", "k": 4},
    "ER <k>=6": {"type": "ER", "k": 6},
    "BA <k>=4": {"type": "BA", "k": 4},
    "BA <k>=6": {"type": "BA", "k": 6},
}
beta_range = np.arange(0, 0.31, 0.01)

# Pre-stored average degrees (matches your generated networks)
network_avg_degrees = {
    "ER <k>=4": 3.98,
    "ER <k>=6": 6.04,
    "BA <k>=4": 3.99,
    "BA <k>=6": 5.98
}

# Plotting configuration
plt.rcParams["figure.dpi"] = 300

# Generate plots for both mu values
for mu in mu_list:
    plt.figure(figsize=(8, 6))
    
    # Load and plot results for each network
    for net_name in network_configs.keys():
        # Load your existing simulation results
        res_file = f"{save_path}/results_mu{mu}_{net_name}.npy"
        res = np.load(res_file, allow_pickle=True).item()
        
        # Plot original Monte Carlo and MMCA curves
        plt.plot(res["beta"], res["rho_mc"],
                 marker="o", ms=3, lw=1.5, label=f"{net_name} (MC)")
        plt.plot(res["beta"], res["rho_mmca"],
                 '--', lw=1, alpha=0.7, label=f"{net_name} (MMCA)")
        
        # Calculate and plot QMF curve (only for ER networks)
        net_type = network_configs[net_name]["type"]
        k_avg = network_avg_degrees[net_name]
        if net_type == "ER":
            beta_c = mu / k_avg
            rho_qmf_list = []
            for beta in beta_range:
                if beta < beta_c:
                    rho_qmf_list.append(0.0)
                else:
                    rho_qmf_list.append(1.0 - mu / (beta * k_avg))
            # Plot QMF as dotted line
            plt.plot(beta_range, rho_qmf_list,
                     ':', lw=2, alpha=0.9, label=f"{net_name} (QMF)")

    # Plot formatting (matches your original figure style)
    plt.xlabel("Infection probability β", fontsize=12)
    plt.ylabel("Steady-state infected fraction <ρ>", fontsize=12)
    plt.title(f"SIS Epidemic Spreading (N=1000, μ={mu})", fontsize=14)
    plt.legend(fontsize=8, loc="best")
    plt.grid(alpha=0.3, linestyle="--")
    plt.xlim(0, 0.3)
    plt.ylim(0, 1.0)
    plt.text(0.02, 0.95,
             f"Nrep=50, Tmax=1000, Ttrans=900",
             fontsize=8, transform=plt.gca().transAxes)
    plt.tight_layout()
    
    # Save new plots (with _QMF suffix to avoid overwriting original)
    plt.savefig(f"{save_path}/SIS_results_mu{mu}_QMF.pdf")
    plt.savefig(f"{save_path}/SIS_results_mu{mu}_QMF.png")
    plt.close()
    print(f"QMF-enhanced plot for μ={mu} generated successfully!")

print("\n✅ All QMF-enhanced plots saved to SIS_simulation_results folder")