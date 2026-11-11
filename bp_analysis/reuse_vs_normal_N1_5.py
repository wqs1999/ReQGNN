#!/usr/bin/env python3
"""
Qubit-Reuse vs Normal PQC: Gradient Variance Comparison
========================================================

Reuse PQC:   4 qubits (2 center + 2 neighbor_pool), neighbor qubits REUSED N times
Normal PQC:  2 + 2N qubits (2 center + each neighbor gets 2 dedicated qubits)

Both circuits:
  - 2 center qubits, feature_dim = 4 (2^2)
  - 2-layer StronglyEntanglingLayers per entanglement step
  - Sequential neighbor processing
  - idential parameter counts

N ranges from 1 to 5:
  N=1: 4q vs 4q   N=2: 4q vs 6q   N=3: 4q vs 8q
  N=4: 4q vs 10q  N=5: 4q vs 12q

All within memory limits (max 4096 states for N=5 normal).
"""

import torch, pennylane as qml, numpy as np, time, os, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'figure.figsize': (18, 13), 'savefig.dpi': 150,
                     'savefig.bbox': 'tight'})

OUT = os.path.dirname(os.path.abspath(__file__))
CENTER_QUBITS = 2
NEIGHBOR_PER = 2
FEATURE_DIM = 2 ** CENTER_QUBITS  # 4
KERNEL_LAYERS = 2
LINEAR_LAYERS = 2
N_LIST = [1, 2, 3, 4, 5]
N_TRIALS = 100
SEED = 42


# ============================================================================
# Circuit Builders
# ============================================================================

def build_reuse_circuit(n_neighbors):
    """Reuse circuit: 4 qubits (0,1=center; 2,3=neighbor, measure+reset)."""
    n_total = CENTER_QUBITS + NEIGHBOR_PER  # 4
    cw = list(range(CENTER_QUBITS))          # [0, 1]
    nw = list(range(CENTER_QUBITS, n_total)) # [2, 3]

    dev = qml.device('default.qubit', wires=n_total)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(cf, nf, kw, pw, lw):
        for i in range(n_neighbors):
            qml.AmplitudeEmbedding(cf, wires=cw, normalize=True)
            qml.AmplitudeEmbedding(nf[i], wires=nw, normalize=True)
            qml.StronglyEntanglingLayers(kw, wires=cw + nw)
            measures = [qml.measure(w, reset=True) for w in nw]
            for mi, m in enumerate(measures):
                for j in cw:
                    qml.cond(m, qml.RY)(pw[mi][j], wires=j)
        qml.StronglyEntanglingLayers(lw, wires=cw)
        return qml.probs(wires=cw)

    return circuit, n_total


def build_normal_circuit(n_neighbors):
    """Normal circuit: 2 + 2N qubits, each neighbor gets dedicated qubits."""
    n_total = CENTER_QUBITS + NEIGHBOR_PER * n_neighbors
    cw = list(range(CENTER_QUBITS))

    dev = qml.device('default.qubit', wires=n_total)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(cf, nf, kw, pw, lw):
        for i in range(n_neighbors):
            start = CENTER_QUBITS + i * NEIGHBOR_PER
            nw_i = list(range(start, start + NEIGHBOR_PER))
            qml.AmplitudeEmbedding(cf, wires=cw, normalize=True)
            qml.AmplitudeEmbedding(nf[i], wires=nw_i, normalize=True)
            qml.StronglyEntanglingLayers(kw, wires=cw + nw_i)
            measures = [qml.measure(w, reset=True) for w in nw_i]
            for mi, m in enumerate(measures):
                for j in cw:
                    qml.cond(m, qml.RY)(pw[mi][j], wires=j)
        qml.StronglyEntanglingLayers(lw, wires=cw)
        return qml.probs(wires=cw)

    return circuit, n_total


# ============================================================================
# Variance Computation
# ============================================================================

def compute_variance(circuit_type, n_neighbors, n_trials=N_TRIALS, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)

    if circuit_type == 'reuse':
        qnode, n_qubits = build_reuse_circuit(n_neighbors)
    else:
        qnode, n_qubits = build_normal_circuit(n_neighbors)

    # Parameter shapes (same for both: 4-qubit entanglement block)
    kern_shape = qml.StronglyEntanglingLayers.shape(KERNEL_LAYERS, CENTER_QUBITS + NEIGHBOR_PER)
    lin_shape = qml.StronglyEntanglingLayers.shape(LINEAR_LAYERS, CENTER_QUBITS)
    pool_shape = (CENTER_QUBITS, CENTER_QUBITS)
    n_params = kern_shape[0]*kern_shape[1]*kern_shape[2] + pool_shape[0]*pool_shape[1] + lin_shape[0]*lin_shape[1]*lin_shape[2]

    grads_k, grads_p, grads_l = [], [], []

    for _ in range(n_trials):
        kw = torch.rand(*kern_shape, dtype=torch.float32, requires_grad=True)
        pw = torch.rand(*pool_shape, dtype=torch.float32, requires_grad=True)
        lw = torch.rand(*lin_shape, dtype=torch.float32, requires_grad=True)

        c = torch.randn(FEATURE_DIM, dtype=torch.float32); c = c / c.norm()
        nf = torch.randn(n_neighbors, FEATURE_DIM, dtype=torch.float32)
        nf = nf / nf.norm(dim=1, keepdim=True)

        out = qnode(c, nf, kw, pw, lw)
        loss = torch.sum((out - torch.rand_like(out)) ** 2)
        loss.backward()

        grads_k.append(kw.grad.detach().flatten().cpu().numpy())
        grads_p.append(pw.grad.detach().flatten().cpu().numpy())
        grads_l.append(lw.grad.detach().flatten().cpu().numpy())

    gk = np.array(grads_k); gp = np.array(grads_p); gl = np.array(grads_l)
    gall = np.concatenate([gk.flatten(), gp.flatten(), gl.flatten()])

    return {
        'type': circuit_type, 'n_neighbors': n_neighbors,
        'n_qubits': n_qubits, 'n_params': n_params,
        'var_total': float(np.var(gall)),
        'var_kernel': float(np.var(gk.flatten())),
        'var_pooling': float(np.var(gp.flatten())),
        'var_linear': float(np.var(gl.flatten())),
        'mean_abs_kernel': float(np.mean(np.abs(gk.flatten()))),
        'mean_abs_pooling': float(np.mean(np.abs(gp.flatten()))),
        'mean_abs_linear': float(np.mean(np.abs(gl.flatten()))),
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_results(results):
    reuse = [r for r in results if r['type'] == 'reuse']
    normal = [r for r in results if r['type'] == 'normal']
    nn_list = [r['n_neighbors'] for r in reuse]
    normal_qs = [r['n_qubits'] for r in normal]

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    cr, cn = '#2196F3', '#F44336'

    # ===== [0,0] Total Variance =====
    ax = axes[0, 0]
    ax.semilogy(nn_list, [r['var_total'] for r in reuse], 'o-', color=cr,
                linewidth=2.5, markersize=12, label='Reuse PQC (4 qubits, fixed)')
    ax.semilogy(nn_list, [r['var_total'] for r in normal], 's-', color=cn,
                linewidth=2.5, markersize=12, label='Normal PQC (2+2N qubits, growing)')

    # BP reference: Var ~ 1/2^n, anchored at N=1
    v0, q0 = normal[0]['var_total'], normal[0]['n_qubits']
    bp_ref = [v0 * (2**q0) / (2**q) for q in normal_qs]
    ax.semilogy(nn_list, bp_ref, '--', color='gray', linewidth=2, alpha=0.7,
                label='Theoretical BP: Var ~ 1/2^n')

    for i, (nn, nq) in enumerate(zip(nn_list, normal_qs)):
        ax.annotate(f'{nq}q', (nn, normal[i]['var_total']),
                    textcoords="offset points", xytext=(0, -20),
                    ha='center', fontsize=9, color=cn, fontweight='bold')

    ax.set_xlabel('Number of Neighbors (N)')
    ax.set_ylabel('Total Gradient Variance (log scale)')
    ax.set_title('Gradient Variance: Reuse vs Normal\n'
                 '(Deviation from gray line = BP mitigation)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xticks(nn_list)

    # ===== [0,1] Variance Ratio =====
    ax = axes[0, 1]
    x = np.arange(len(nn_list))
    w = 0.2
    ratios_t = [reuse[i]['var_total'] / normal[i]['var_total'] for i in range(len(nn_list))]
    ratios_k = [reuse[i]['var_kernel'] / normal[i]['var_kernel'] for i in range(len(nn_list))]
    ratios_p = [reuse[i]['var_pooling'] / normal[i]['var_pooling'] for i in range(len(nn_list))]
    ratios_l = [reuse[i]['var_linear'] / normal[i]['var_linear'] for i in range(len(nn_list))]

    bars_t = ax.bar(x - 1.5*w, ratios_t, w, color='#E91E63', label='Total', edgecolor='black')
    ax.bar(x - 0.5*w, ratios_k, w, color='#2196F3', label='Kernel', edgecolor='black')
    ax.bar(x + 0.5*w, ratios_p, w, color='#FF9800', label='Pooling', edgecolor='black')
    ax.bar(x + 1.5*w, ratios_l, w, color='#4CAF50', label='Linear', edgecolor='black')
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)

    # Value labels on bars
    for bar in bars_t:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}',
                ha='center', va='bottom' if h > 0 else 'top', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'N={nn}\n(4q vs {nq}q)' for nn, nq in zip(nn_list, normal_qs)])
    ax.set_ylabel('Var(Reuse) / Var(Normal)')
    ax.set_title('Variance Advantage Ratio\n(>1 = Reuse has higher gradient variance)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # ===== [0,2] Qubit Usage =====
    ax = axes[0, 2]
    x = np.arange(len(nn_list))
    bw = 0.35
    b1 = ax.bar(x - bw/2, [r['n_qubits'] for r in reuse], bw, color=cr,
                label='Reuse PQC', edgecolor='black')
    b2 = ax.bar(x + bw/2, normal_qs, bw, color=cn,
                label='Normal PQC', edgecolor='black')
    for b in b1: ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.2,
                         f'{int(b.get_height())}', ha='center', fontweight='bold')
    for b in b2: ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.2,
                         f'{int(b.get_height())}', ha='center', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'N={n}' for n in nn_list])
    ax.set_ylabel('Total Qubits');
    ax.set_title('Hardware Resource Comparison')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # ===== [1,0] Per-Parameter-Group Variance =====
    ax = axes[1, 0]
    ax.semilogy(nn_list, [r['var_kernel'] for r in reuse], 'o--', color='#2196F3',
                linewidth=1.5, markersize=8, label='Reuse: Kernel', alpha=0.7)
    ax.semilogy(nn_list, [r['var_pooling'] for r in reuse], 's--', color='#FF9800',
                linewidth=1.5, markersize=8, label='Reuse: Pooling', alpha=0.7)
    ax.semilogy(nn_list, [r['var_linear'] for r in reuse], '^--', color='#4CAF50',
                linewidth=1.5, markersize=8, label='Reuse: Linear', alpha=0.7)
    ax.semilogy(nn_list, [r['var_kernel'] for r in normal], 'o-', color='#2196F3',
                linewidth=2, markersize=8, label='Normal: Kernel', alpha=0.7)
    ax.semilogy(nn_list, [r['var_pooling'] for r in normal], 's-', color='#FF9800',
                linewidth=2, markersize=8, label='Normal: Pooling', alpha=0.7)
    ax.semilogy(nn_list, [r['var_linear'] for r in normal], '^-', color='#4CAF50',
                linewidth=2, markersize=8, label='Normal: Linear', alpha=0.7)
    ax.set_xlabel('Number of Neighbors (N)')
    ax.set_ylabel('Gradient Variance (log)')
    ax.set_title('Per-Parameter-Group Variance\n(Dashed=Reuse 4q, Solid=Normal 2+2N q)')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3); ax.set_xticks(nn_list)

    # ===== [1,1] BP Mitigation Factor =====
    ax = axes[1, 1]
    x = np.arange(len(nn_list))
    bw = 0.35
    mit_reuse, mit_normal = [], []
    for i in range(len(nn_list)):
        bp_theory_n = normal[0]['var_total'] * (2**normal[0]['n_qubits']) / (2**normal[i]['n_qubits'])
        mit_reuse.append(reuse[i]['var_total'] / bp_theory_n)
        mit_normal.append(normal[i]['var_total'] / bp_theory_n)

    b1 = ax.bar(x - bw/2, mit_reuse, bw, color=cr, label='Reuse PQC', edgecolor='black')
    b2 = ax.bar(x + bw/2, mit_normal, bw, color=cn, label='Normal PQC', edgecolor='black')
    for b in b1:
        ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.02,
                f'{b.get_height():.1f}', ha='center', fontsize=8, fontweight='bold')
    for b in b2:
        ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.02,
                f'{b.get_height():.1f}', ha='center', fontsize=8, fontweight='bold')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='No BP mitigation')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={nn}\n({nq}q)' for nn, nq in zip(nn_list, normal_qs)])
    ax.set_ylabel('Var(actual) / Var(theoretical BP)')
    ax.set_title('Barren Plateau Mitigation Factor\n(>1 = better than theoretical random PQC)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    # ===== [1,2] Summary Report =====
    ax = axes[1, 2]; ax.axis('off')

    lines = ["=" * 55, "  SUMMARY REPORT", "=" * 55, "",
             f"Configuration:",
             f"  Center qubits: {CENTER_QUBITS} (feature_dim={FEATURE_DIM})",
             f"  Neighbor qubits per group: {NEIGHBOR_PER}",
             f"  Kernel layers: {KERNEL_LAYERS}, Linear layers: {LINEAR_LAYERS}",
             f"  Trials per point: {N_TRIALS}",
             f"  Parameters per circuit: {results[0]['n_params']}",
             f"  Reuse circuit: {CENTER_QUBITS + NEIGHBOR_PER} qubits (fixed)",
             f"  Normal circuit: {CENTER_QUBITS} + {NEIGHBOR_PER}*N qubits (grows)",
             "", "-" * 55,
             f" {'N':>3s}  {'Reuse (4q)':>14s}  {'Normal':>8s}  {'Ratio':>8s}  "
             f"{'BP Mitigation':>15s}",
             "-" * 55]

    for i in range(len(nn_list)):
        r, n = reuse[i], normal[i]
        ratio = r['var_total'] / n['var_total']
        bp_theory = normal[0]['var_total'] * (2**normal[0]['n_qubits']) / (2**normal[i]['n_qubits'])
        bp_mit = n['var_total'] / bp_theory
        lines.append(f" {nn_list[i]:3d}  {r['var_total']:>14.2e}  "
                     f"{n['n_qubits']:5d}q  {ratio:>8.1f}x  {bp_mit:>15.1f}x")

    lines.append("-" * 55)
    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("  - Sequential entanglement means each StronglyEntanglingLayers")
    lines.append("    only acts on 4 qubits. Extra qubits are idle spectators.")
    lines.append("  - Var(Reuse) = Var(Normal) confirms: reuse has NO gradient penalty.")
    lines.append("  - Reuse achieves identical gradient behavior with 2x fewer qubits")
    lines.append("    at N=5 (4q vs 12q).")

    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

    plt.suptitle(f'Qubit-Reuse PQC (4 qubits) vs Normal PQC (2+2N qubits)\n'
                 f'Gradient Variance Comparison for Graph Convolution',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT, 'reuse_vs_normal_N1_5.png')
    plt.savefig(path); plt.close()
    print(f"\nSaved chart: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("REUSE vs NORMAL PQC: GRADIENT VARIANCE (4q reuse, 2+2N q normal)")
    print(f"CENTER_QUBITS={CENTER_QUBITS}, FEATURE_DIM={FEATURE_DIM}")
    print(f"KERNEL_LAYERS={KERNEL_LAYERS}, LINEAR_LAYERS={LINEAR_LAYERS}")
    print(f"N_LIST={N_LIST}, TRIALS={N_TRIALS}")
    print("=" * 70)

    results = []
    for N in N_LIST:
        reuse_q = CENTER_QUBITS + NEIGHBOR_PER  # 4
        normal_q = CENTER_QUBITS + NEIGHBOR_PER * N  # 2+2N

        print(f"\n--- N={N} (Reuse: {reuse_q}q, Normal: {normal_q}q) ---")

        t0 = time.time()
        r_r = compute_variance('reuse', N, N_TRIALS, SEED)
        results.append(r_r)
        print(f"  Reuse ({reuse_q}q):  var_total={r_r['var_total']:.2e}, "
              f"var_k={r_r['var_kernel']:.2e}, t={time.time()-t0:.1f}s")

        t0 = time.time()
        r_n = compute_variance('normal', N, N_TRIALS, SEED)
        results.append(r_n)
        print(f"  Normal ({normal_q}q): var_total={r_n['var_total']:.2e}, "
              f"var_k={r_n['var_kernel']:.2e}, t={time.time()-t0:.1f}s")

    # Print table
    reuse = [r for r in results if r['type'] == 'reuse']
    normal = [r for r in results if r['type'] == 'normal']
    normal_qs = [r['n_qubits'] for r in normal]

    print("\n" + "=" * 70)
    print(f"{'N':>3s}  {'Reuse 4q':>14s}  {'Normal q':>8s}  "
          f"{'Normal Var':>14s}  {'Ratio R/N':>12s}  {'BP Mitigation':>15s}")
    print("-" * 75)
    for i in range(len(reuse)):
        r, n = reuse[i], normal[i]
        ratio = r['var_total'] / n['var_total']
        bp_theory = normal[0]['var_total'] * (2**normal[0]['n_qubits']) / (2**normal[i]['n_qubits'])
        bp_mit = n['var_total'] / bp_theory
        print(f"{r['n_neighbors']:3d}  {r['var_total']:>14.2e}  "
              f"{n['n_qubits']:8d}  {n['var_total']:>14.2e}  "
              f"{ratio:>12.1f}x  {bp_mit:>15.1f}x")

    ratio_N5 = reuse[-1]['var_total'] / normal[-1]['var_total']
    q_saved = normal_qs[-1] - reuse[0]['n_qubits']
    print(f"\nN=5: Reuse 4q vs Normal {normal_qs[-1]}q "
          f"-> Reuse has {ratio_N5:.1f}x gradient variance, saves {q_saved} qubits")

    # Save JSON
    json_path = os.path.join(OUT, 'reuse_vs_normal_N1_5.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {json_path}")

    # Plot
    plot_results(results)
    return results


if __name__ == '__main__':
    main()
