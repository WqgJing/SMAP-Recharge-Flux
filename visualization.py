
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_comprehensive_results(model, q0_data, soil_params, n_t=200, n_z=100, device='cpu'):
    """
    Comprehensive plotting with 5 subplots - NORMALIZED VERSION
    """
    model.eval()

    # Get time range from q0_data
    t_min, t_max = min(q0_data[0]), max(q0_data[0])

    # Create time grid
    t_lin = torch.linspace(t_min, t_max, n_t).to(device)

    # Get water table depth over time - FIXED
    with torch.no_grad():
        zb_vals = model.predict_water_table(t_lin.view(-1, 1)).cpu().numpy().flatten()

    # Create adaptive z grid
    z_min = -np.max(zb_vals) - 0.5
    z_max = 0.0
    z_lin = torch.linspace(z_min, z_max, n_z).to(device)

    # Create meshgrid for h(z,t)
    T, Z = torch.meshgrid(t_lin, z_lin, indexing="ij")
    t_flat = T.reshape(-1, 1)
    z_flat = Z.reshape(-1, 1)

    # Compute h(z,t) over the grid - FIXED
    with torch.no_grad():
        h_flat, _ = model.predict_head(z_flat, t_flat)
    H = h_flat.reshape(n_t, n_z).cpu().numpy()

    # Mask out regions below water table
    T_np = T.cpu().numpy()
    Z_np = Z.cpu().numpy()

    H_masked = H.copy()
    for i, t_val in enumerate(t_lin.cpu().numpy()):
        zb_at_t = zb_vals[i]
        mask = Z_np[i, :] < -zb_at_t
        H_masked[i, mask] = np.nan

    # Get surface head h(0,t) - FIXED
    z_surface = torch.zeros(n_t, 1).to(device)
    t_surface = t_lin.view(-1, 1)
    with torch.no_grad():
        h_surface, _ = model.predict_head(z_surface, t_surface)
    h_surface = h_surface.cpu().numpy().flatten()

    # Initial conditions - FIXED
    t_ic = torch.tensor([t_min]).to(device).view(-1, 1)
    z_ic_range = torch.linspace(-1.0, 0.0, 50).to(device).view(-1, 1)

    # Prescribed IC
    with torch.no_grad():
        zb_ic = model.predict_water_table(t_ic).cpu().numpy().item()
    h_ic_prescribed = (-zb_ic - z_ic_range.cpu().numpy().flatten())

    # Modeled IC - FIXED
    t_ic_expanded = t_ic.expand(50, 1)
    with torch.no_grad():
        h_ic_modeled, _ = model.predict_head(z_ic_range, t_ic_expanded)
    h_ic_modeled = h_ic_modeled.cpu().numpy().flatten()

    # Calculate h at water table - FIXED
    z_wt = (-torch.tensor(zb_vals)).to(device).view(-1, 1)
    t_wt = t_lin.view(-1, 1)
    with torch.no_grad():
        h_at_wt, _ = model.predict_head(z_wt, t_wt)
    h_at_wt = h_at_wt.cpu().numpy().flatten()

    # Create the 5-subplot figure
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("PINN Results Analysis", fontsize=16, fontweight="bold")

    # Subplot 1: h(z,t) 2D colormap
    im1 = axs[0, 0].pcolormesh(
        T.cpu().numpy() / 86400, Z.cpu().numpy(), H_masked, shading="auto", cmap="viridis"
    )
    axs[0, 0].plot(
        t_lin.cpu().numpy() / 86400, -zb_vals, "r--", lw=2, label="Water table z_b(t)"
    )
    axs[0, 0].set_title("Pressure Head h(z,t)")
    axs[0, 0].set_xlabel("Time [days]")
    axs[0, 0].set_ylabel("Depth z [m]")
    axs[0, 0].set_ylim(-5, 0)
    axs[0, 0].legend(loc="lower right")
    plt.colorbar(im1, ax=axs[0, 0], label="h [m]")

    # Subplot 2: zb(t) time series
    axs[0, 1].plot(t_lin.cpu().numpy() / 86400, zb_vals, "b-", lw=2)
    axs[0, 1].set_title("Water Table Depth z_b(t)")
    axs[0, 1].set_xlabel("Time [days]")
    axs[0, 1].set_ylabel("z_b [m] (positive downward)")
    axs[0, 1].grid(True, alpha=0.3)

    # Subplot 3: h(zb(t),t)
    axs[0, 2].plot(t_lin.cpu().numpy() / 86400, h_at_wt, "m-", lw=2, label="h(-zb(t), t)")
    axs[0, 2].axhline(0, color="k", lw=1, linestyle="--", alpha=0.7, label="h=0")
    axs[0, 2].set_title("Pressure Head at Water Table")
    axs[0, 2].set_xlabel("Time [days]")
    axs[0, 2].set_ylabel("h(-zb(t), t) [m]")
    axs[0, 2].grid(True, alpha=0.3)
    axs[0, 2].legend()

    # Subplot 4: Surface flux comparison - FIXED
    z_surf_req = torch.zeros(n_t, 1, requires_grad=True).to(device)
    t_surf_req = t_lin.view(-1, 1).requires_grad_(True)

    # Normalize for computation
    z_surf_tilde = model.normalizer.normalize_z(z_surf_req)
    t_surf_tilde = model.normalizer.normalize_t(t_surf_req)
    
    h_surf_grad_tilde, _ = model(z_surf_tilde, t_surf_tilde)
    K_surf_tilde = model.normalizer.K_tilde(h_surf_grad_tilde)
    dh_dz_surf_tilde = torch.autograd.grad(h_surf_grad_tilde.sum(), z_surf_tilde, create_graph=True)[0]
    q_surf_tilde = -K_surf_tilde * (dh_dz_surf_tilde + 1.0)
    
    # Denormalize flux
    q_surf_simulated = model.normalizer.denormalize_q(q_surf_tilde).detach().cpu().numpy().flatten()

    axs[1, 0].plot(
        np.array(q0_data[0]) / 86400, np.array(q0_data[1]) * 1e6, "g-", lw=2, label="q0 (prescribed)"
    )
    axs[1, 0].plot(
        t_lin.cpu().numpy() / 86400, q_surf_simulated * 1e6, "r--", lw=2, label="q0 (simulated)"
    )
    axs[1, 0].set_title("Surface Flux Comparison")
    axs[1, 0].set_xlabel("Time [days]")
    axs[1, 0].set_ylabel("q0 [μm/s]")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Subplot 5: Initial conditions comparison
    axs[1, 1].plot(
        h_ic_prescribed,
        z_ic_range.cpu().numpy().flatten(),
        "g-",
        lw=2,
        label="Prescribed IC: h(z,t0)",
    )
    axs[1, 1].plot(
        h_ic_modeled,
        z_ic_range.cpu().numpy().flatten(),
        "b-",
        lw=2,
        label="Modeled IC: h(z,t0)",
    )
    axs[1, 1].set_title("Initial Conditions Comparison")
    axs[1, 1].set_xlabel("Pressure head h [m]")
    axs[1, 1].set_ylabel("Depth z [m]")
    axs[1, 1].set_ylim(-1.0, 0.0)
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].invert_yaxis()

    # Hide unused subplot
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Water table depth range: {zb_vals.min():.3f} to {zb_vals.max():.3f} m")
    print(f"Surface head range: {h_surface.min():.3f} to {h_surface.max():.3f} m")
    print(f"h(-zb(t), t) range: {h_at_wt.min():.6f} to {h_at_wt.max():.6f} m")
    print(f"Max |h(-zb(t), t)|: {np.max(np.abs(h_at_wt)):.6f} m (should be close to 0)")
    print(f"Max IC error: {np.max(np.abs(h_ic_modeled - h_ic_prescribed)):.6f} m")


def plot_training_losses(losses_pool, comps_pool):
    """
    Plot training loss evolution with 6 subplots showing different loss components.
    
    Args:
        losses_pool: list or array of total weighted losses over epochs
        comps_pool: dictionary containing loss components with keys:
                   'pde', 'surf', 'wt_head', 'ic_h'
    """

    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Pool+Batch Training - Weighted Loss Evolution', fontsize=16, fontweight='bold')

    # Plot 1: Total Weighted Loss
    ax1 = axes[0, 0]
    ax1.semilogy(losses_pool, 'r-', label='Total Weighted Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Weighted Loss')
    ax1.set_title('Total Weighted Loss Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: PDE Weighted Loss
    ax2 = axes[0, 1]
    ax2.semilogy(comps_pool['pde'], 'r-', label='PDE Weighted Loss', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PDE Weighted Loss')
    ax2.set_title('PDE Weighted Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Surface BC Weighted Loss
    ax3 = axes[0, 2]
    ax3.semilogy(comps_pool['surf'], 'r-', label='Surface BC Weighted Loss', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Surface BC Weighted Loss')
    ax3.set_title('Surface BC Weighted Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Water Table Head Weighted Loss
    ax4 = axes[1, 0]
    ax4.semilogy(comps_pool['wt_head'], 'r-', label='WT Head Weighted Loss', alpha=0.7)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('WT Head Weighted Loss')
    ax4.set_title('Water Table Head Weighted Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Initial Condition Weighted Loss
    ax5 = axes[1, 1]
    ax5.semilogy(comps_pool['ic_h'], 'r-', label='IC Weighted Loss', alpha=0.7)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('IC Weighted Loss')
    ax5.set_title('Initial Condition Weighted Loss')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Final Weighted Loss Summary
    ax6 = axes[1, 2]
    # Create bar plot showing final weighted losses
    loss_names = ['Total', 'PDE', 'Surf BC', 'WT Head', 'IC']
    pool_final = [losses_pool[-1], comps_pool['pde'][-1], comps_pool['surf'][-1], 
                 comps_pool['wt_head'][-1], comps_pool['ic_h'][-1]]

    x = np.arange(len(loss_names))
    width = 0.7

    bars = ax6.bar(x, pool_final, width, label='Final Weighted Loss', alpha=0.7, color='red')

    ax6.set_ylabel('Final Weighted Loss Value')
    ax6.set_title('Final Weighted Loss Summary')
    ax6.set_xticks(x)
    ax6.set_xticklabels(loss_names, rotation=45)
    ax6.legend()
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
