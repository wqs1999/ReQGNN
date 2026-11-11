#!/usr/bin/env python3
"""
Focused test: Reuse vs Normal PQC gradient variance at N=4 (6q vs 15q).
Uses lightning.qubit + adjoint for 15-qubit circuits to avoid OOM.
"""

import torch
import pennylane as qml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.figsize': (16, 12), 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIM = 8
N_KERNEL_LAYERS = 2
N_LINEAR_LAYERS = 2
N_NEIGHBORS_LIST = [1, 2, 3, 4]
N_TRIALS = 80  # fewer trials for speed

# Use lightning.qubit when qubits >= 15 to avoid OOM
# For 12 qubits, default.qubit+backprop works fine
LIGHTNING_THRESHOLD = 13  # qubit count


def build_reuse_circuit(n_neighbors):
    n_center = int(np.log2(FEATURE_DIM))  # 3
    n_total = n_center * 2                # 6
    center_w = list(range(n_center))
    neigh_w = list(range(n_center, n_total))

    dev = qml.device('default.qubit', wires=n_total)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(center_feat, neigh_feat, qkernel_w, qpooling_w, qlinear_w):
        for i in range(n_neighbors):
            qml.AmplitudeEmbedding(center_feat, wires=center_w, normalize=True)
            qml.AmplitudeEmbedding(neigh_feat[i], wires=neigh_w, normalize=True)
            qml.StronglyEntanglingLayers(qkernel_w, wires=range(n_total))
            measures = [qml.measure(w, reset=True) for w in neigh_w]
            for mi, m in enumerate(measures):
                for j in center_w:
                    qml.cond(m, qml.RY)(qpooling_w[mi][j], wires=j)
        qml.StronglyEntanglingLayers(qlinear_w, wires=center_w)
        return qml.probs(wires=center_w)

    return circuit, n_total


def build_normal_circuit(n_neighbors):
    n_center = int(np.log2(FEATURE_DIM))  # 3
    n_per_neighbor = n_center             # 3
    n_total = n_center + n_per_neighbor * n_neighbors  # 3+3N
    center_w = list(range(n_center))

    # Choose device/diff_method based on qubit count
    if n_total >= LIGHTNING_THRESHOLD:
        dev = qml.device('lightning.qubit', wires=n_total)
        diff_method = 'adjoint'
    else:
        dev = qml.device('default.qubit', wires=n_total)
        diff_method = 'backprop'

    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def circuit(center_feat, neigh_feat, qkernel_w, qpooling_w, qlinear_w):
        for i in range(n_neighbors):
            start = n_center + i * n_per_neighbor
            neigh_w_i = list(range(start, start + n_per_neighbor))

            qml.AmplitudeEmbedding(center_feat, wires=center_w, normalize=True)
            qml.AmplitudeEmbedding(neigh_feat[i], wires=neigh_w_i, normalize=True)

            active_wires = center_w + neigh_w_i
            qml.StronglyEntanglingLayers(qkernel_w, wires=active_wires)

            measures = [qml.measure(w, reset=True) for w in neigh_w_i]
            for mi, m in enumerate(measures):
                for j in center_w:
                    qml.cond(m, qml.RY)(qpooling_w[mi][j], wires=j)

        qml.StronglyEntanglingLayers(qlinear_w, wires=center_w)
        return qml.probs(wires=center_w)

    return circuit, n_total


def compute_variance(circuit_type, n_neighbors, n_trials=N_TRIALS, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if circuit_type == 'reuse':
        qnode, n_qubits = build_reuse_circuit(n_neighbors)
    else:
        qnode, n_qubits = build_normal_circuit(n_neighbors)

    n_center = int(np.log2(FEATURE_DIM))
    kern_shape = qml.StronglyEntanglingLayers.shape(N_KERNEL_LAYERS, n_center * 2)
    lin_shape = qml.StronglyEntanglingLayers.shape(N_LINEAR_LAYERS, n_center)
    pool_shape = (n_center, n_center)

    all_grads_kern = []
    all_grads_pool = []
    all_grads_lin = []

    for trial in range(n_trials):
        kw = torch.rand(*kern_shape, dtype=torch.float32, requires_grad=True)
        pw = torch.rand(*pool_shape, dtype=torch.float32, requires_grad=True)
        lw = torch.rand(*lin_shape, dtype=torch.float32, requires_grad=True)

        c = torch.randn(FEATURE_DIM, dtype=torch.float32)
        c = c / c.norm()
        n = torch.randn(n_neighbors, FEATURE_DIM, dtype=torch.float32)
        n = n / n.norm(dim=1, keepdim=True)

        probs = qnode(c, n, kw, pw, lw)
        loss = torch.sum((probs - torch.rand_like(probs)) ** 2)
        loss.backward()

        all_grads_kern.append(kw.grad.detach().flatten().cpu().numpy())
        all_grads_pool.append(pw.grad.detach().flatten().cpu().numpy())
        all_grads_lin.append(lw.grad.detach().flatten().cpu().numpy())

    gk = np.array(all_grads_kern)
    gp = np.array(all_grads_pool)
    gl = np.array(all_grads_lin)
    gall = np.concatenate([gk.flatten(), gp.flatten(), gl.flatten()])

    return {
        'type': circuit_type, 'n_neighbors': n_neighbors, 'n_qubits': n_qubits,
        'var_total': float(np.var(gall)),
        'var_kernel': float(np.var(gk.flatten())),
        'var_pooling': float(np.var(gp.flatten())),
        'var_linear': float(np.var(gl.flatten())),
        'mean_abs_kernel': float(np.mean(np.abs(gk.flatten()))),
        'mean_abs_pooling': float(np.mean(np.abs(gp.flatten()))),
        'mean_abs_linear': float(np.mean(np.abs(gl.flatten()))),
    }


def main():
    print("=" * 70)
    print("REUSE vs NORMAL: GRADIENT VARIANCE (N=1..4, with lightning for 15q)")
    print("=" * 70)

    results = []
    for nn in N_NEIGHBORS_LIST:
        normal_qubits = 3 + 3 * nn
        print(f"\n--- N={nn} (Reuse: 6q, Normal: {normal_qubits}q) ---")

        t0 = time.time()
        r = compute_variance('reuse', nn, N_TRIALS)
        results.append(r)
        print(f"  Reuse ({r['n_qubits']}q): var_total={r['var_total']:.2e}, "
              f"var_kernel={r['var_kernel']:.2e}, time={time.time()-t0:.1f}s")

        t0 = time.time()
        r = compute_variance('normal', nn, N_TRIALS)
        results.append(r)
        print(f"  Normal ({r['n_qubits']}q): var_total={r['var_total']:.2e}, "
              f"var_kernel={r['var_kernel']:.2e}, time={time.time()-t0:.1f}s")

    # Print table
    reuse = [r for r in results if r['type'] == 'reuse']
    normal = [r for r in results if r['type'] == 'normal']
    print("\n" + "=" * 70)
    print(f"{'N':>4s}  {'Reuse 6q':>14s}  {'Normal q':>8s}  "
          f"{'Normal Var':>14s}  {'Ratio R/N':>12s}  {'Device':>20s}")
    print("-" * 80)
    for i in range(len(reuse)):
        r, n = reuse[i], normal[i]
        ratio = r['var_total'] / n['var_total']
        dev = 'lightning+adjoint' if n['n_qubits'] >= LIGHTNING_THRESHOLD else 'default+backprop'
        print(f"{r['n_neighbors']:4d}  {r['var_total']:>14.2e}  "
              f"{n['n_qubits']:8d}  {n['var_total']:>14.2e}  "
              f"{ratio:>12.2f}x  {dev:>20s}")

    # Generate charts
    nn_list = [r['n_neighbors'] for r in reuse]
    normal_qubits_list = [r['n_qubits'] for r in normal]

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    c_reuse, c_normal = '#2196F3', '#F44336'

    # 1. Total variance
    ax = axes[0, 0]
    ax.semilogy(nn_list, [r['var_total'] for r in reuse], 'o-',
                color=c_reuse, linewidth=2.5, markersize=12, label='Reuse PQC (6 qubits)')
    ax.semilogy(nn_list, [r['var_total'] for r in normal], 's-',
                color=c_normal, linewidth=2.5, markersize=12, label='Normal PQC (3+3N qubits)')

    # BP reference: Var ~ 1/2^n (anchored at N=1)
    v0 = normal[0]['var_total']
    q0 = normal[0]['n_qubits']
    bp_ref = [v0 * (2**q0) / (2**q) for q in normal_qubits_list]
    ax.semilogy(nn_list, bp_ref, '--', color='gray', linewidth=2, alpha=0.7,
                label='Theoretical BP: Var ~ 1/2^n')

    for i, (nn, nq) in enumerate(zip(nn_list, normal_qubits_list)):
        ax.annotate(f'{nq}q', (nn, normal[i]['var_total']),
                    textcoords="offset points", xytext=(0, -18),
                    ha='center', fontsize=9, color=c_normal, fontweight='bold')

    ax.set_xlabel('Number of Neighbors (N)')
    ax.set_ylabel('Total Gradient Variance (log scale)')
    ax.set_title('Total Gradient Variance: Reuse vs Normal\n'
                 '(Only difference: qubit reuse vs fresh qubit allocation)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(nn_list)

    # 2. Ratio bar chart
    ax = axes[0, 1]
    ratios_total = [reuse[i]['var_total'] / normal[i]['var_total'] for i in range(len(nn_list))]
    ratios_kernel = [reuse[i]['var_kernel'] / normal[i]['var_kernel'] for i in range(len(nn_list))]
    ratios_pooling = [reuse[i]['var_pooling'] / normal[i]['var_pooling'] for i in range(len(nn_list))]
    ratios_linear = [reuse[i]['var_linear'] / normal[i]['var_linear'] for i in range(len(nn_list))]

    x = np.arange(len(nn_list))
    width = 0.2
    ax.bar(x - 1.5*width, ratios_total, width, color='#E91E63', label='Total', edgecolor='black')
    ax.bar(x - 0.5*width, ratios_kernel, width, color='#2196F3', label='Kernel', edgecolor='black')
    ax.bar(x + 0.5*width, ratios_pooling, width, color='#FF9800', label='Pooling', edgecolor='black')
    ax.bar(x + 1.5*width, ratios_linear, width, color='#4CAF50', label='Linear', edgecolor='black')
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={nn}\n(6q vs {nq}q)' for nn, nq in zip(nn_list, normal_qubits_list)])
    ax.set_ylabel('Var(Reuse) / Var(Normal)')
    ax.set_title('Gradient Variance Ratio\n(>1 = Reuse has HIGHER variance)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Per-parameter-group comparison
    ax = axes[1, 0]
    ax.semilogy(nn_list, [r['var_kernel'] for r in reuse], 'o--', color='#2196F3',
                linewidth=2, markersize=10, label='Reuse: Kernel', alpha=0.7)
    ax.semilogy(nn_list, [r['var_pooling'] for r in reuse], 's--', color='#FF9800',
                linewidth=2, markersize=10, label='Reuse: Pooling', alpha=0.7)
    ax.semilogy(nn_list, [r['var_linear'] for r in reuse], '^--', color='#4CAF50',
                linewidth=2, markersize=10, label='Reuse: Linear', alpha=0.7)
    ax.semilogy(nn_list, [r['var_kernel'] for r in normal], 'o-', color='#2196F3',
                linewidth=2, markersize=10, label='Normal: Kernel', alpha=0.7)
    ax.semilogy(nn_list, [r['var_pooling'] for r in normal], 's-', color='#FF9800',
                linewidth=2, markersize=10, label='Normal: Pooling', alpha=0.7)
    ax.semilogy(nn_list, [r['var_linear'] for r in normal], '^-', color='#4CAF50',
                linewidth=2, markersize=10, label='Normal: Linear', alpha=0.7)
    ax.set_xlabel('Number of Neighbors (N)')
    ax.set_ylabel('Gradient Variance (log scale)')
    ax.set_title('Per-Parameter-Group Variance\n(Dashed=Reuse 6q, Solid=Normal 3+3Nq)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(nn_list)

    # 4. BP mitigation factor
    ax = axes[1, 1]
    mit_reuse, mit_normal = [], []
    for i in range(len(nn_list)):
        bp_theory_n = normal[0]['var_total'] * (2**normal[0]['n_qubits']) / (2**normal[i]['n_qubits'])
        mit_reuse.append(reuse[i]['var_total'] / bp_theory_n)
        mit_normal.append(normal[i]['var_total'] / bp_theory_n)

    x = np.arange(len(nn_list))
    bw = 0.35
    b1 = ax.bar(x - bw/2, mit_reuse, bw, color=c_reuse, label='Reuse PQC', edgecolor='black')
    b2 = ax.bar(x + bw/2, mit_normal, bw, color=c_normal, label='Normal PQC', edgecolor='black')
    for b in b1:
        ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.02, f'{b.get_height():.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    for b in b2:
        ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.02, f'{b.get_height():.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='No BP mitigation')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={nn}\n({nq}q)' for nn, nq in zip(nn_list, normal_qubits_list)])
    ax.set_ylabel('Var(actual) / Var(theoretical BP)')
    ax.set_title('Barren Plateau Mitigation Factor\n(>1 = better than random PQC)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Qubit-Reuse PQC vs Normal PQC: Gradient Variance\n'
                 f'(FEATURE_DIM={FEATURE_DIM}, trials={N_TRIALS})',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'reuse_vs_normal_N4.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")

    return results


if __name__ == '__main__':
    results = main()
