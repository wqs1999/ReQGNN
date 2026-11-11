#!/usr/bin/env python3
"""
================================================================================
Qubit-Reuse PQC vs Normal PQC: Gradient Variance Comparison
================================================================================

Compare gradient variance between two PQC designs for graph convolution:

  1. Qubit-Reused PQC (6 qubits):
     - Center: 3 qubits, Neighbor: 3 qubits
     - Neighbor qubits are measured+reset after each neighbor → reused N times
     - Total qubits: ALWAYS 6 (regardless of N)

  2. Normal PQC (3+3N qubits):
     - Center: 3 qubits
     - Each neighbor gets DEDICATED 3 qubits (no reuse)
     - Total qubits: 3 + 3N (grows with N)

  Both circuits:
    - Use SAME kernel weight shape (n_layers, 6, 3) — always entangle 6 qubits
    - Use SAME pooling weight shape (3, 3)
    - Use SAME linear weight shape (n_layers, 3, 3)
    - Process neighbors SEQUENTIALLY (same structure, different qubit allocation)

  The ONLY variable: qubit reuse vs fresh qubit allocation.

Key hypothesis:
  Reused PQC maintains stable gradient variance while normal PQC suffers from
  Barren Plateau (gradient variance decay proportional to 1/2^n_qubits).

Author: BPAnalysis Project
================================================================================
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import warnings
import json

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (18, 10),
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIM = 8           # => 3 center qubits (2^3 = 8)
N_KERNEL_LAYERS = 2       # StronglyEntanglingLayers depth for kernel
N_LINEAR_LAYERS = 2       # StronglyEntanglingLayers depth for readout
N_NEIGHBORS_LIST = [1, 2, 3]  # Number of neighbors (backprop feasible up to 12q)
N_TRIALS = 100            # Random initializations per configuration
LOSS_TYPE = 'mse'         # 'mse' or 'nll'

print(f"Output directory: {OUTPUT_DIR}")


# ============================================================================
# Circuit Builders
# ============================================================================

def build_reuse_circuit(n_neighbors: int):
    """
    Build the qubit-reused PQC.

    Features:
      - 6 qubits total: wires 0,1,2 = center; wires 3,4,5 = neighbor (reused)
      - Neighbor qubits are measured with reset=True after each neighbor
      - Same qkernel_w applied each iteration on the same 6 qubits
      - Same qpooling_w for conditional RY rotations

    Returns: (qnode, n_qubits_total)
    """
    n_center = int(np.log2(FEATURE_DIM))  # 3
    n_total = n_center * 2                # 6
    center_w = list(range(n_center))      # [0, 1, 2]
    neigh_w = list(range(n_center, n_total))  # [3, 4, 5]

    dev = qml.device('default.qubit', wires=n_total)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(center_feat, neigh_feat, qkernel_w, qpooling_w, qlinear_w):
        # Process each neighbor sequentially with qubit REUSE
        for i in range(n_neighbors):
            qml.AmplitudeEmbedding(center_feat, wires=center_w, normalize=True)
            qml.AmplitudeEmbedding(neigh_feat[i], wires=neigh_w, normalize=True)
            qml.StronglyEntanglingLayers(qkernel_w, wires=range(n_total))

            # Measure + RESET neighbor qubits (this is the REUSE mechanism)
            measures = [qml.measure(w, reset=True) for w in neigh_w]
            for mi, m in enumerate(measures):
                for j in center_w:
                    qml.cond(m, qml.RY)(qpooling_w[mi][j], wires=j)

        qml.StronglyEntanglingLayers(qlinear_w, wires=center_w)
        return qml.probs(wires=center_w)

    return circuit, n_total


def build_normal_circuit(n_neighbors: int):
    """
    Build the normal (no-reuse) PQC.

    Features:
      - 3+3N qubits total: wires 0,1,2 = center; each neighbor gets 3 dedicated qubits
      - Each neighbor processed sequentially (same structure as reuse circuit)
      - SAME qkernel_w shape (always entangles 6 qubits at a time)
      - SAME qpooling_w shape (3, 3)
      - Key difference: neighbor qubits are FRESH each time, not reused

    Returns: (qnode, n_qubits_total)
    """
    n_center = int(np.log2(FEATURE_DIM))  # 3
    n_per_neighbor = n_center             # 3 qubits per neighbor
    n_total = n_center + n_per_neighbor * n_neighbors  # 3 + 3N
    center_w = list(range(n_center))      # [0, 1, 2]

    dev = qml.device('default.qubit', wires=n_total)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(center_feat, neigh_feat, qkernel_w, qpooling_w, qlinear_w):
        # Process each neighbor sequentially with DEDICATED fresh qubits
        for i in range(n_neighbors):
            start = n_center + i * n_per_neighbor
            neigh_w_i = list(range(start, start + n_per_neighbor))

            qml.AmplitudeEmbedding(center_feat, wires=center_w, normalize=True)
            qml.AmplitudeEmbedding(neigh_feat[i], wires=neigh_w_i, normalize=True)

            # Entangle center + current neighbor (always 6 qubits)
            active_wires = center_w + neigh_w_i
            qml.StronglyEntanglingLayers(qkernel_w, wires=active_wires)

            # Measure + reset neighbor qubits (reset doesn't matter here
            # since we never touch them again, kept for structural parity)
            measures = [qml.measure(w, reset=True) for w in neigh_w_i]
            for mi, m in enumerate(measures):
                for j in center_w:
                    qml.cond(m, qml.RY)(qpooling_w[mi][j], wires=j)

        qml.StronglyEntanglingLayers(qlinear_w, wires=center_w)
        return qml.probs(wires=center_w)

    return circuit, n_total


# ============================================================================
# Gradient Variance Computation
# ============================================================================

def compute_gradient_variance(circuit_type: str, n_neighbors: int,
                               n_trials: int = N_TRIALS,
                               seed: int = 42) -> dict:
    """
    Compute gradient variance at random initialization for a given circuit.

    Parameters
    ----------
    circuit_type : 'reuse' or 'normal'
    n_neighbors : int
        Number of neighbors to process.
    n_trials : int
        Number of random initializations.

    Returns
    -------
    dict with keys:
        var_qkernel, var_qpooling, var_qlinear, var_total,
        mean_abs_qkernel, mean_abs_qpooling, mean_abs_qlinear,
        n_qubits, n_neighbors, circuit_type, n_params_total
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build circuit
    if circuit_type == 'reuse':
        qnode, n_qubits = build_reuse_circuit(n_neighbors)
    else:
        qnode, n_qubits = build_normal_circuit(n_neighbors)

    # Parameter shapes (same for both circuits!)
    n_center = int(np.log2(FEATURE_DIM))
    kernel_shape = qml.StronglyEntanglingLayers.shape(
        n_layers=N_KERNEL_LAYERS, n_wires=n_center * 2  # always 6 wires
    )
    linear_shape = qml.StronglyEntanglingLayers.shape(
        n_layers=N_LINEAR_LAYERS, n_wires=n_center  # always 3 wires
    )
    pooling_shape = (n_center, n_center)  # (3, 3)

    n_params = (kernel_shape[0] * kernel_shape[1] * kernel_shape[2] +
                pooling_shape[0] * pooling_shape[1] +
                linear_shape[0] * linear_shape[1] * linear_shape[2])

    all_grads_qkernel = []
    all_grads_qpooling = []
    all_grads_qlinear = []

    for trial in range(n_trials):
        # Random initialization
        qkernel_w = torch.rand(*kernel_shape, dtype=torch.float32,
                               requires_grad=True)
        qpooling_w = torch.rand(*pooling_shape, dtype=torch.float32,
                                requires_grad=True)
        qlinear_w = torch.rand(*linear_shape, dtype=torch.float32,
                               requires_grad=True)

        # Random normalized input features
        center_feat = torch.randn(FEATURE_DIM, dtype=torch.float32)
        center_feat = center_feat / center_feat.norm()

        neigh_feat = torch.randn(n_neighbors, FEATURE_DIM, dtype=torch.float32)
        neigh_feat = neigh_feat / neigh_feat.norm(dim=1, keepdim=True)

        # Forward pass
        probs = qnode(center_feat, neigh_feat, qkernel_w, qpooling_w, qlinear_w)

        # Loss
        if LOSS_TYPE == 'mse':
            target = torch.rand_like(probs)
            loss = torch.sum((probs - target) ** 2)
        else:
            target_class = torch.randint(0, len(probs), (1,)).item()
            loss = -torch.log(probs[target_class] + 1e-10)

        # Backward
        loss.backward()

        all_grads_qkernel.append(
            qkernel_w.grad.detach().flatten().cpu().numpy())
        all_grads_qpooling.append(
            qpooling_w.grad.detach().flatten().cpu().numpy())
        all_grads_qlinear.append(
            qlinear_w.grad.detach().flatten().cpu().numpy())

    # Aggregate
    gk = np.array(all_grads_qkernel)
    gp = np.array(all_grads_qpooling)
    gl = np.array(all_grads_qlinear)
    gall = np.concatenate([gk.flatten(), gp.flatten(), gl.flatten()])

    return {
        'circuit_type': circuit_type,
        'n_neighbors': n_neighbors,
        'n_qubits': n_qubits,
        'n_params': n_params,
        'var_qkernel': float(np.var(gk.flatten())),
        'var_qpooling': float(np.var(gp.flatten())),
        'var_qlinear': float(np.var(gl.flatten())),
        'var_total': float(np.var(gall)),
        'mean_abs_qkernel': float(np.mean(np.abs(gk.flatten()))),
        'mean_abs_qpooling': float(np.mean(np.abs(gp.flatten()))),
        'mean_abs_qlinear': float(np.mean(np.abs(gl.flatten()))),
        'std_qkernel': float(np.std(gk.flatten())),
        'std_qpooling': float(np.std(gp.flatten())),
        'std_qlinear': float(np.std(gl.flatten())),
    }


# ============================================================================
# Main Analysis
# ============================================================================

def run_analysis():
    """Run gradient variance comparison for all neighbor counts."""
    results = []

    for n_neighbors in N_NEIGHBORS_LIST:
        # Reuse circuit (always 6 qubits)
        print(f"\nN={n_neighbors}: Reuse circuit (6 qubits, {n_neighbors} neighbors)...")
        t0 = time.time()
        r_reuse = compute_gradient_variance('reuse', n_neighbors, N_TRIALS)
        t_reuse = time.time() - t0
        results.append(r_reuse)
        print(f"  qubits={r_reuse['n_qubits']}, "
              f"Var(total)={r_reuse['var_total']:.2e}, "
              f"Var(kernel)={r_reuse['var_qkernel']:.2e}, "
              f"time={t_reuse:.1f}s")

        # Normal circuit (3+3N qubits)
        print(f"N={n_neighbors}: Normal circuit ({3+3*n_neighbors} qubits, {n_neighbors} neighbors)...")
        t0 = time.time()
        r_normal = compute_gradient_variance('normal', n_neighbors, N_TRIALS)
        t_normal = time.time() - t0
        results.append(r_normal)
        print(f"  qubits={r_normal['n_qubits']}, "
              f"Var(total)={r_normal['var_total']:.2e}, "
              f"Var(kernel)={r_normal['var_qkernel']:.2e}, "
              f"time={t_normal:.1f}s")

    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_results(results: list):
    """Generate comprehensive comparison charts."""

    # Separate reuse and normal results
    reuse = [r for r in results if r['circuit_type'] == 'reuse']
    normal = [r for r in results if r['circuit_type'] == 'normal']

    nn_list = [r['n_neighbors'] for r in reuse]  # [1, 2, 3, 4, 5]
    normal_qubits = [r['n_qubits'] for r in normal]

    # ========================================================================
    # Figure 1: Main comparison (2x2 subplots)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    colors = {'reuse': '#2196F3', 'normal': '#F44336'}
    markers = {'reuse': 'o', 'normal': 's'}
    labels = {'reuse': 'Reuse PQC (6 qubits)', 'normal': 'Normal PQC (3+3N qubits)'}

    # --- Subplot 1: Total gradient variance ---
    ax = axes[0, 0]
    ax.semilogy(nn_list, [r['var_total'] for r in reuse],
                color=colors['reuse'], marker=markers['reuse'],
                linewidth=2.5, markersize=12, label=labels['reuse'])
    ax.semilogy(nn_list, [r['var_total'] for r in normal],
                color=colors['normal'], marker=markers['normal'],
                linewidth=2.5, markersize=12, label=labels['normal'])

    # Add theoretical BP reference: Var ~ 1/2^n (normalized to first point)
    v0_normal = normal[0]['var_total']
    q0_normal = normal[0]['n_qubits']
    bp_ref = [v0_normal * (2 ** q0_normal) / (2 ** q) for q in normal_qubits]
    ax.semilogy(nn_list, bp_ref, '--', color='gray', linewidth=2, alpha=0.7,
                label='Theoretical BP: Var proportional to 1/2^n')

    # Annotate qubit counts on normal line
    for i, (nn, nq) in enumerate(zip(nn_list, normal_qubits)):
        ax.annotate(f'{nq}q', (nn, normal[i]['var_total']),
                    textcoords="offset points", xytext=(0, -18),
                    ha='center', fontsize=9, color=colors['normal'],
                    fontweight='bold')

    ax.set_xlabel('Number of Neighbors (N)')
    ax.set_ylabel('Total Gradient Variance (log scale)')
    ax.set_title('Total Gradient Variance Comparison\n'
                 '(Lower values = higher Barren Plateau risk)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(nn_list)

    # --- Subplot 2: Variance ratio (Reuse / Normal) ---
    ax = axes[0, 1]
    ratios_total = [reuse[i]['var_total'] / normal[i]['var_total']
                    for i in range(len(nn_list))]
    ratios_kernel = [reuse[i]['var_qkernel'] / normal[i]['var_qkernel']
                     for i in range(len(nn_list))]
    ratios_pooling = [reuse[i]['var_qpooling'] / normal[i]['var_qpooling']
                      for i in range(len(nn_list))]
    ratios_linear = [reuse[i]['var_qlinear'] / normal[i]['var_qlinear']
                     for i in range(len(nn_list))]

    x = np.arange(len(nn_list))
    width = 0.2
    ax.bar(x - 1.5 * width, ratios_total, width, color='#E91E63',
           label='Total', edgecolor='black', linewidth=0.5)
    ax.bar(x - 0.5 * width, ratios_kernel, width, color='#2196F3',
           label='Kernel Weights', edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.5 * width, ratios_pooling, width, color='#FF9800',
           label='Pooling Weights', edgecolor='black', linewidth=0.5)
    ax.bar(x + 1.5 * width, ratios_linear, width, color='#4CAF50',
           label='Linear Weights', edgecolor='black', linewidth=0.5)

    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={nn}\n(6q vs {nq}q)'
                        for nn, nq in zip(nn_list, normal_qubits)])
    ax.set_ylabel('Variance Ratio: Var(Reuse) / Var(Normal)')
    ax.set_title('Gradient Variance Advantage of Qubit Reuse\n'
                 '(Ratio > 1 = Reuse has HIGHER variance = better trainability)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Subplot 3: Per-parameter-group variance comparison ---
    ax = axes[1, 0]

    # Reuse: dashed lines
    ax.semilogy(nn_list, [r['var_qkernel'] for r in reuse],
                'o--', color='#2196F3', linewidth=2, markersize=10,
                label='Reuse: Kernel', alpha=0.8)
    ax.semilogy(nn_list, [r['var_qpooling'] for r in reuse],
                's--', color='#FF9800', linewidth=2, markersize=10,
                label='Reuse: Pooling', alpha=0.8)
    ax.semilogy(nn_list, [r['var_qlinear'] for r in reuse],
                '^--', color='#4CAF50', linewidth=2, markersize=10,
                label='Reuse: Linear', alpha=0.8)

    # Normal: solid lines
    ax.semilogy(nn_list, [r['var_qkernel'] for r in normal],
                'o-', color='#2196F3', linewidth=2, markersize=10,
                label='Normal: Kernel', alpha=0.8)
    ax.semilogy(nn_list, [r['var_qpooling'] for r in normal],
                's-', color='#FF9800', linewidth=2, markersize=10,
                label='Normal: Pooling', alpha=0.8)
    ax.semilogy(nn_list, [r['var_qlinear'] for r in normal],
                '^-', color='#4CAF50', linewidth=2, markersize=10,
                label='Normal: Linear', alpha=0.8)

    ax.set_xlabel('Number of Neighbors (N)')
    ax.set_ylabel('Gradient Variance (log scale)')
    ax.set_title('Per-Parameter-Group Gradient Variance\n'
                 '(Dashed = Reuse, Solid = Normal)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(nn_list)

    # --- Subplot 4: BP Mitigation Factor ---
    ax = axes[1, 1]

    # Compute BP mitigation: how many times better than theoretical BP scaling
    # Mitigation factor = Var(actual) / Var(theoretical BP)
    mitigation_reuse = []
    mitigation_normal = []

    for i in range(len(nn_list)):
        # For reuse: compare to theoretical BP from 6 qubits
        # (constant, should be flat)
        bp_theory_r = reuse[0]['var_total']  # same qubit count, no decay
        mitigation_reuse.append(reuse[i]['var_total'] / bp_theory_r)

        # For normal: compare to theoretical BP from 3+3N qubits
        bp_theory_n = normal[0]['var_total'] * (2 ** normal[0]['n_qubits']) / \
                      (2 ** normal[i]['n_qubits'])
        mitigation_normal.append(normal[i]['var_total'] / bp_theory_n)

    x = np.arange(len(nn_list))
    width = 0.35
    bars1 = ax.bar(x - width / 2, mitigation_reuse, width,
                   color=colors['reuse'], label='Reuse PQC (6 qubits)',
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, mitigation_normal, width,
                   color=colors['normal'], label='Normal PQC (3+3N qubits)',
                   edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.02,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.02,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Baseline (no BP mitigation)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={nn}\n({nq}q)' for nn, nq in zip(nn_list, normal_qubits)])
    ax.set_ylabel('Var(actual) / Var(theoretical BP)')
    ax.set_title('Barren Plateau Mitigation Factor\n'
                 '(>1 = better than theoretical BP, <1 = worse)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Qubit-Reuse PQC vs Normal PQC: Gradient Variance Comparison\n'
                 f'(feature_dim={FEATURE_DIM}, kernel_layers={N_KERNEL_LAYERS}, '
                 f'linear_layers={N_LINEAR_LAYERS}, trials={N_TRIALS})',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'reuse_vs_normal_main.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")

    # ========================================================================
    # Figure 2: Variance vs Qubit Count (direct BP visualization)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot reuse points (all at 6 qubits)
    reuse_qubits = [r['n_qubits'] for r in reuse]
    reuse_vars = [r['var_total'] for r in reuse]
    ax.scatter(reuse_qubits, reuse_vars, c=colors['reuse'], s=200, marker='o',
               zorder=5, label='Reuse PQC (N=1..5, all 6 qubits)', edgecolors='black')
    # Connect reuse points to show stability
    ax.plot(reuse_qubits, reuse_vars, '-', color=colors['reuse'], linewidth=1.5, alpha=0.5)

    # Plot normal points (3+3N qubits)
    normal_qubits_list = [r['n_qubits'] for r in normal]
    normal_vars = [r['var_total'] for r in normal]
    ax.scatter(normal_qubits_list, normal_vars, c=colors['normal'], s=200, marker='s',
               zorder=5, label='Normal PQC (N=1..5, 3+3N qubits)', edgecolors='black')
    ax.plot(normal_qubits_list, normal_vars, '-', color=colors['normal'], linewidth=1.5, alpha=0.5)

    # Theoretical BP curve
    q_range = np.linspace(6, 18, 100)
    bp_curve = normal[0]['var_total'] * (2 ** normal[0]['n_qubits']) / (2 ** q_range)
    ax.semilogy(q_range, bp_curve, '--', color='gray', linewidth=2, alpha=0.7,
                label='Theoretical BP: Var proportional to 1/2^n')

    # Annotations
    for i, (nn, nq, rv, nv) in enumerate(zip(nn_list, normal_qubits_list,
                                              reuse_vars, normal_vars)):
        ax.annotate(f'N={nn}', (6, rv), textcoords="offset points",
                    xytext=(-50, 10), ha='center', fontsize=9, color=colors['reuse'],
                    fontweight='bold')
        ax.annotate(f'N={nn}', (nq, nv), textcoords="offset points",
                    xytext=(30, 10), ha='center', fontsize=9, color=colors['normal'],
                    fontweight='bold')

    ax.set_xlabel('Total Number of Qubits')
    ax.set_ylabel('Total Gradient Variance (log scale)')
    ax.set_title('Direct Barren Plateau Visualization\n'
                 'Reuse (6q, flat) vs Normal (3+3N qubits, exponential decay)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(set(reuse_qubits + normal_qubits_list)))

    save_path = os.path.join(OUTPUT_DIR, 'reuse_vs_normal_bp.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ========================================================================
    # Figure 3: Summary text report
    # ========================================================================
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    lines = []
    lines.append("=" * 70)
    lines.append("QUBIT-REUSE PQC vs NORMAL PQC: GRADIENT VARIANCE ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Configuration:")
    lines.append(f"  Feature dimension: {FEATURE_DIM} (center qubits: {int(np.log2(FEATURE_DIM))})")
    lines.append(f"  Kernel layers: {N_KERNEL_LAYERS}")
    lines.append(f"  Linear layers: {N_LINEAR_LAYERS}")
    lines.append(f"  Trials per config: {N_TRIALS}")
    lines.append(f"  Loss type: {LOSS_TYPE}")
    lines.append("")
    lines.append(f"Number of parameters (identical for both circuits):")
    n_params = results[0]['n_params']
    lines.append(f"  qkernel_weight:  {qml.StronglyEntanglingLayers.shape(N_KERNEL_LAYERS, 6)} = {N_KERNEL_LAYERS * 6 * 3}")
    lines.append(f"  qpooling_weight: (3, 3) = 9")
    lines.append(f"  qlinear_weight:  {qml.StronglyEntanglingLayers.shape(N_LINEAR_LAYERS, 3)} = {N_LINEAR_LAYERS * 3 * 3}")
    lines.append(f"  TOTAL: {n_params}")
    lines.append("")
    lines.append("-" * 70)
    lines.append(f"{'N':>4s}  {'Reuse Qubits':>12s}  {'Reuse Var':>14s}  "
                 f"{'Normal Qubits':>14s}  {'Normal Var':>14s}  {'Ratio':>10s}")
    lines.append("-" * 70)

    for i in range(len(nn_list)):
        r = reuse[i]
        n = normal[i]
        ratio = r['var_total'] / n['var_total']
        lines.append(f"{nn_list[i]:4d}  {r['n_qubits']:12d}  {r['var_total']:14.2e}  "
                     f"{n['n_qubits']:14d}  {n['var_total']:14.2e}  {ratio:10.1f}x")

    lines.append("-" * 70)
    lines.append("")
    lines.append("Key Observations:")
    lines.append("")

    # Analyze trends
    reuse_vars_list = [r['var_total'] for r in reuse]
    normal_vars_list = [r['var_total'] for r in normal]

    reuse_stability = max(reuse_vars_list) / min(reuse_vars_list) if min(reuse_vars_list) > 0 else float('inf')
    normal_decay = normal_vars_list[0] / normal_vars_list[-1] if normal_vars_list[-1] > 0 else float('inf')

    lines.append(f"  1. Reuse circuit variance stability (max/min): {reuse_stability:.2f}x")
    lines.append(f"     -> Reuse PQC maintains nearly constant gradient variance")
    lines.append(f"        regardless of neighbor count (always 6 qubits).")
    lines.append("")

    lines.append(f"  2. Normal circuit variance decay (N=1 -> N=5): {normal_decay:.1f}x")
    # Compare to theoretical
    bp_theory_decay = (2 ** normal[0]['n_qubits']) / (2 ** normal[-1]['n_qubits'])
    lines.append(f"     -> Theoretical BP decay (2^{normal[0]['n_qubits']} / 2^{normal[-1]['n_qubits']}): {bp_theory_decay:.1f}x")
    lines.append(f"     -> Actual decay is {normal_decay/bp_theory_decay:.1f}x {'worse' if normal_decay > bp_theory_decay else 'better'} than theoretical BP")
    lines.append("")

    ratio_n1 = reuse[0]['var_total'] / normal[0]['var_total']
    ratio_n5 = reuse[-1]['var_total'] / normal[-1]['var_total']
    lines.append(f"  3. Variance advantage of reuse over normal:")
    lines.append(f"     At N=1 (both 6q): {ratio_n1:.1f}x")
    lines.append(f"     At N=5 (6q vs 18q): {ratio_n5:.1f}x")
    lines.append(f"     -> Reuse provides {ratio_n5/ratio_n1:.1f}x relative improvement from N=1 to N=5")
    lines.append("")

    lines.append("-" * 70)
    lines.append("CONCLUSION:")
    if ratio_n5 > 100:
        lines.append("  The qubit-reuse PQC demonstrates DRAMATIC gradient variance advantage")
        lines.append("  over the normal PQC. By keeping the qubit count fixed at 6 through")
        lines.append("  measurement-based qubit reuse, the circuit completely avoids the")
        lines.append("  exponential decay of gradient variance that plagues normal PQCs")
        lines.append("  as the number of neighbors (and thus qubits) grows.")
    elif ratio_n5 > 10:
        lines.append("  The qubit-reuse PQC demonstrates SIGNIFICANT gradient variance advantage")
        lines.append("  over the normal PQC. The reuse mechanism effectively mitigates barren")
        lines.append("  plateaus by maintaining constant Hilbert space dimension.")
    elif ratio_n5 > 2:
        lines.append("  The qubit-reuse PQC shows a MODERATE gradient variance advantage over")
        lines.append("  the normal PQC, providing some barren plateau mitigation.")
    else:
        lines.append("  The qubit-reuse PQC shows comparable gradient variance to the normal")
        lines.append("  PQC. While it doesn't provide dramatic BP mitigation in this setup,")
        lines.append("  it achieves parity with far fewer qubits (6 vs 18).")
    lines.append("=" * 70)

    ax.text(0.05, 0.98, '\n'.join(lines), transform=ax.transAxes,
            fontfamily='monospace', fontsize=10.5, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    save_path = os.path.join(OUTPUT_DIR, 'reuse_vs_normal_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# JSON Export
# ============================================================================

def export_json(results: list):
    """Export numerical results as JSON for further analysis."""
    save_path = os.path.join(OUTPUT_DIR, 'reuse_vs_normal_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("QUBIT-REUSE PQC vs NORMAL PQC: GRADIENT VARIANCE COMPARISON")
    print("=" * 70)
    print(f"Neighbor counts: {N_NEIGHBORS_LIST}")
    print(f"Trials per config: {N_TRIALS}")
    print(f"Reuse circuit: always 6 qubits")
    print(f"Normal circuit: 3+3N qubits = {[3+3*n for n in N_NEIGHBORS_LIST]}")
    print()

    t_total_start = time.time()

    results = run_analysis()

    t_total = time.time() - t_total_start
    print(f"\nTotal computation time: {t_total:.1f}s")

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    plot_results(results)
    export_json(results)

    # Quick summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    reuse = [r for r in results if r['circuit_type'] == 'reuse']
    normal = [r for r in results if r['circuit_type'] == 'normal']

    print(f"\n{'N':>4s}  {'Reuse Var':>14s}  {'Normal Var':>14s}  "
          f"{'Ratio (R/N)':>14s}  {'Normal Qubits':>14s}")
    print("-" * 65)
    for i in range(len(reuse)):
        ratio = reuse[i]['var_total'] / normal[i]['var_total']
        print(f"{reuse[i]['n_neighbors']:4d}  {reuse[i]['var_total']:>14.2e}  "
              f"{normal[i]['var_total']:>14.2e}  {ratio:>14.1f}x  "
              f"{normal[i]['n_qubits']:>14d}")

    ratio_n5 = reuse[-1]['var_total'] / normal[-1]['var_total']
    print(f"\nAt N=5 (6 qubits vs 18 qubits): Reuse has {ratio_n5:.1f}x higher gradient variance")

    return results


if __name__ == '__main__':
    results = main()
