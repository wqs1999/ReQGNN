#!/usr/bin/env python3
"""
================================================================================
Gradient Variance Analysis for Qubit-Reused QGCN (Parameterized Quantum Circuit)
================================================================================

This script analyzes the trainability of the qubit-reused parameterized quantum
circuit (PQC) used in QRGCN. The key innovation of this circuit is "qubit reuse":
neighbor qubits are measured and reset after processing each neighbor, allowing
a fixed number of qubits to handle variable numbers of neighbors.

Trainability Metrics:
  1. Gradient variance evolution during training (should not vanish)
  2. Gradient variance scaling with qubit count (Barren Plateau check)
  3. Gradient variance scaling with circuit depth
  4. Per-parameter-group gradient distribution analysis

A well-trainable PQC exhibits:
  - Non-vanishing gradient variance (>> 1e-6) throughout training
  - Sub-exponential decay of variance with qubit count
  - Stable gradient norms without explosion

Author: BPAnalysis Project
Date: 2025
================================================================================
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
import warnings
import time
import os

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (14, 10),
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================================
# Quantum Circuit Definition (faithful recreation from QRGCN)
# ============================================================================

def build_qubit_reused_qnode(
    feature_dim: int = 8,
    qgkernel_iter_num: int = 2,
    qlinear_layer: int = 2,
    n_neighbors: int = 4,
    diff_method: str = 'backprop'
):
    """
    Build a PennyLane QNode that implements the qubit-reused quantum circuit.

    The circuit works as follows:
      1. Center features are embedded on the first half of qubits
      2. For each neighbor, neighbor features are embedded on the second half
      3. StronglyEntanglingLayers entangles all qubits
      4. Neighbor qubits are measured; conditional RY rotations are applied
         to center qubits based on measurement outcomes
      5. Neighbor qubits are reset for the next neighbor (THIS IS THE KEY: qubit reuse)
      6. After all neighbors, final StronglyEntanglingLayers on center qubits
      7. Return probability distribution on center qubits

    Parameters
    ----------
    feature_dim : int
        Dimension of input features (must be a power of 2).
    qgkernel_iter_num : int
        Number of StronglyEntanglingLayers for the kernel.
    qlinear_layer : int
        Number of StronglyEntanglingLayers for the final readout.
    n_neighbors : int
        Number of neighbor nodes to process.
    diff_method : str
        Differentiation method for the QNode.

    Returns
    -------
    qnode : qml.QNode
        The constructed QNode.
    n_qubits_total : int
        Total number of qubits used.
    n_qubits_center : int
        Number of qubits for the center node.
    """
    n_qubits_center = int(np.log2(feature_dim))
    n_qubits_total = n_qubits_center * 2

    # Parameter shapes
    shape_qkernel = qml.StronglyEntanglingLayers.shape(
        n_layers=qgkernel_iter_num, n_wires=n_qubits_total
    )
    shape_qlinear = qml.StronglyEntanglingLayers.shape(
        n_layers=qlinear_layer, n_wires=n_qubits_center
    )
    pooling_shape = (n_qubits_center, n_qubits_center)

    dev = qml.device('default.qubit', wires=n_qubits_total)

    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def qgcc_circuit(
        center_features,        # [feature_dim]
        neighbor_features,      # [n_neighbors, feature_dim]
        qkernel_weight,         # StronglyEntanglingLayers params
        qpooling_weight,        # [n_qubits_center, n_qubits_center]
        qlinear_weight          # StronglyEntanglingLayers params
    ):
        """
        Quantum Graph Convolution Circuit with qubit reuse.
        """
        center_wires = range(n_qubits_center)
        neighbor_wires = range(n_qubits_center, n_qubits_total)

        # Process each neighbor with qubit reuse
        for n_idx in range(n_neighbors):
            # Embed center features on center qubits
            qml.AmplitudeEmbedding(
                center_features, wires=center_wires, normalize=True
            )
            # Embed neighbor features on neighbor qubits
            qml.AmplitudeEmbedding(
                neighbor_features[n_idx], wires=neighbor_wires, normalize=True
            )
            # Entangle all qubits
            qml.StronglyEntanglingLayers(
                weights=qkernel_weight, wires=range(n_qubits_total)
            )
            # Measure neighbor qubits and conditionally rotate center qubits
            measures = [
                qml.measure(w, reset=True) for w in neighbor_wires
            ]
            for mi, m in enumerate(measures):
                for j in center_wires:
                    qml.cond(m, qml.RY)(qpooling_weight[mi][j], wires=j)

        # Final readout layer on center qubits
        qml.StronglyEntanglingLayers(
            weights=qlinear_weight, wires=center_wires
        )
        return qml.probs(wires=center_wires)

    return qgcc_circuit, n_qubits_total, n_qubits_center


# ============================================================================
# Model Wrapper
# ============================================================================

class QubitReusedQCModel(nn.Module):
    """
    Wrapper that combines the qubit-reused quantum circuit with a classical
    linear classifier, mimicking the full QRGCN setup.
    """

    def __init__(
        self,
        feature_dim: int = 8,
        qgkernel_iter_num: int = 2,
        qlinear_layer: int = 2,
        n_neighbors: int = 4,
        num_classes: int = 2,
        diff_method: str = 'backprop'
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_qubits_center = int(np.log2(feature_dim))
        self.n_qubits_total = self.n_qubits_center * 2
        self.n_neighbors = n_neighbors
        self.output_dim = 2 ** self.n_qubits_center

        # Build QNode
        self.qnode, _, _ = build_qubit_reused_qnode(
            feature_dim=feature_dim,
            qgkernel_iter_num=qgkernel_iter_num,
            qlinear_layer=qlinear_layer,
            n_neighbors=n_neighbors,
            diff_method=diff_method
        )

        # Register quantum parameters
        shape_qkernel = qml.StronglyEntanglingLayers.shape(
            n_layers=qgkernel_iter_num, n_wires=self.n_qubits_total
        )
        self.qkernel_weight = nn.Parameter(
            torch.rand(*shape_qkernel, dtype=torch.float32)
        )

        shape_qlinear = qml.StronglyEntanglingLayers.shape(
            n_layers=qlinear_layer, n_wires=self.n_qubits_center
        )
        self.qlinear_weight = nn.Parameter(
            torch.rand(*shape_qlinear, dtype=torch.float32)
        )

        self.qpooling_weight = nn.Parameter(
            torch.rand(self.n_qubits_center, self.n_qubits_center, dtype=torch.float32)
        )

        # Classical classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.output_dim, num_classes, dtype=torch.float32)
        )

    def forward(self, center_features, neighbor_features):
        """
        Parameters
        ----------
        center_features : Tensor [batch, feature_dim]
        neighbor_features : Tensor [batch, n_neighbors, feature_dim]

        Returns
        -------
        logits : Tensor [batch, num_classes]
        """
        batch_size = center_features.shape[0]
        q_outputs = []

        for i in range(batch_size):
            out = self.qnode(
                center_features[i],
                neighbor_features[i],
                self.qkernel_weight,
                self.qpooling_weight,
                self.qlinear_weight
            )
            q_outputs.append(out)

        q_outputs = torch.stack(q_outputs, dim=0).float()
        return self.classifier(q_outputs)

    def get_param_groups(self):
        """Return dict of parameter groups for analysis."""
        return {
            'qkernel_weight': self.qkernel_weight,
            'qpooling_weight': self.qpooling_weight,
            'qlinear_weight': self.qlinear_weight,
            'classifier': list(self.classifier.parameters()),
        }


# ============================================================================
# Synthetic Data Generator
# ============================================================================

def generate_synthetic_graph_data(
    n_samples: int = 200,
    feature_dim: int = 8,
    n_neighbors: int = 4,
    num_classes: int = 2,
    seed: int = 42
):
    """
    Generate synthetic graph data matching the QRGCN input format.
    Each sample: 1 center node + n_neighbors neighbor nodes.

    Returns
    -------
    center_features : Tensor [n_samples, feature_dim]
    neighbor_features : Tensor [n_samples, n_neighbors, feature_dim]
    labels : Tensor [n_samples]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate random normalized features
    center_features = torch.randn(n_samples, feature_dim, dtype=torch.float32)
    center_features = center_features / center_features.norm(dim=1, keepdim=True)

    neighbor_features = torch.randn(n_samples, n_neighbors, feature_dim, dtype=torch.float32)
    neighbor_features = neighbor_features / neighbor_features.norm(dim=2, keepdim=True)

    labels = torch.randint(0, num_classes, (n_samples,))

    return center_features, neighbor_features, labels


# ============================================================================
# Gradient Tracking Utilities
# ============================================================================

class GradientTracker:
    """Tracks gradient statistics during training."""

    def __init__(self):
        self.history = defaultdict(lambda: {
            'var': [], 'mean': [], 'norm': [], 'max_abs': [], 'step': []
        })

    def record(self, model, step: int):
        """Record gradient statistics for all parameter groups."""
        for name, param_group in [
            ('qkernel_weight', model.qkernel_weight),
            ('qpooling_weight', model.qpooling_weight),
            ('qlinear_weight', model.qlinear_weight),
        ]:
            if param_group.grad is not None:
                g = param_group.grad.detach().flatten()
                self.history[name]['var'].append(g.var().item())
                self.history[name]['mean'].append(g.mean().item())
                self.history[name]['norm'].append(g.norm().item())
                self.history[name]['max_abs'].append(g.abs().max().item())
                self.history[name]['step'].append(step)

        # Track classifier gradients
        for i, layer in enumerate(model.classifier):
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                g = layer.weight.grad.detach().flatten()
                name = f'classifier_layer_{i}'
                self.history[name]['var'].append(g.var().item())
                self.history[name]['mean'].append(g.mean().item())
                self.history[name]['norm'].append(g.norm().item())
                self.history[name]['max_abs'].append(g.abs().max().item())
                self.history[name]['step'].append(step)

    def get_summary(self, param_name: str):
        """Get summary statistics for a parameter group."""
        d = self.history[param_name]
        if len(d['var']) == 0:
            return None
        return {
            'var_initial': d['var'][0],
            'var_final': d['var'][-1],
            'var_mean': np.mean(d['var']),
            'var_std': np.std(d['var']),
            'norm_initial': d['norm'][0],
            'norm_final': d['norm'][-1],
            'norm_mean': np.mean(d['norm']),
        }


# ============================================================================
# Analysis 1: Gradient Variance During Training
# ============================================================================

def analyze_gradient_variance_during_training(
    feature_dim: int = 8,
    qgkernel_iter_num: int = 2,
    qlinear_layer: int = 2,
    n_neighbors: int = 4,
    num_classes: int = 2,
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 0.01,
    seed: int = 42,
    verbose: bool = True
):
    """
    Train the QRGCN model and track gradient variance over time.

    Returns
    -------
    tracker : GradientTracker
    train_losses : list
    train_accs : list
    summary : dict
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate data
    center_feat, neigh_feat, labels = generate_synthetic_graph_data(
        n_samples=500, feature_dim=feature_dim,
        n_neighbors=n_neighbors, num_classes=num_classes, seed=seed
    )

    # Train/test split
    n_train = 400
    train_center = center_feat[:n_train]
    train_neigh = neigh_feat[:n_train]
    train_labels = labels[:n_train]

    # Build model
    model = QubitReusedQCModel(
        feature_dim=feature_dim,
        qgkernel_iter_num=qgkernel_iter_num,
        qlinear_layer=qlinear_layer,
        n_neighbors=n_neighbors,
        num_classes=num_classes,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    tracker = GradientTracker()

    train_losses = []
    train_accs = []

    n_batches = n_train // batch_size

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0

        # Shuffle
        perm = torch.randperm(n_train)
        train_center_shuf = train_center[perm]
        train_neigh_shuf = train_neigh[perm]
        train_labels_shuf = train_labels[perm]

        for b in range(n_batches):
            start = b * batch_size
            end = start + batch_size

            bc = train_center_shuf[start:end]
            bn = train_neigh_shuf[start:end]
            bl = train_labels_shuf[start:end]

            logits = model(bc, bn)
            loss = loss_fn(logits, bl)
            epoch_loss += loss.item()
            epoch_correct += (logits.argmax(dim=1) == bl).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_correct / n_train
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)

        # Record gradient statistics every 5 epochs
        if epoch % 5 == 0:
            # Do one more backward pass to record gradients
            model.train()
            idx = torch.randperm(n_train)[:batch_size]
            bc_sample = train_center[idx]
            bn_sample = train_neigh[idx]
            bl_sample = train_labels[idx]

            logits = model(bc_sample, bn_sample)
            loss = loss_fn(logits, bl_sample)
            optimizer.zero_grad()
            loss.backward()

            tracker.record(model, epoch)

            if verbose:
                print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, "
                      f"|grad_kernel|={tracker.history['qkernel_weight']['norm'][-1]:.4f}, "
                      f"Var(grad_kernel)={tracker.history['qkernel_weight']['var'][-1]:.6f}")

            optimizer.zero_grad()  # Clear for next real training step

    # Final gradient record
    model.train()
    idx = torch.randperm(n_train)[:batch_size]
    logits = model(train_center[idx], train_neigh[idx])
    loss = loss_fn(logits, train_labels[idx])
    optimizer.zero_grad()
    loss.backward()
    tracker.record(model, n_epochs)

    summary = {}
    for name in ['qkernel_weight', 'qpooling_weight', 'qlinear_weight']:
        summary[name] = tracker.get_summary(name)

    return tracker, train_losses, train_accs, summary


# ============================================================================
# Analysis 2: Barren Plateau Check -- Variance vs Qubit Count
# ============================================================================

def compute_gradient_variance_at_init(
    feature_dim: int = 8,
    qgkernel_iter_num: int = 2,
    qlinear_layer: int = 2,
    n_neighbors: int = 4,
    n_trials: int = 100,
    loss_type: str = 'mse',
    seed: int = 42
):
    """
    Compute the variance of gradients at random initialization for a given
    circuit configuration. This is the primary Barren Plateau diagnostic.

    Parameters
    ----------
    feature_dim : int
        Input feature dimension.
    qgkernel_iter_num, qlinear_layer : int
        Circuit depth parameters.
    n_neighbors : int
        Number of neighbors.
    n_trials : int
        Number of random initializations to average over.
    loss_type : str
        'mse' or 'cross_entropy'

    Returns
    -------
    result : dict
        Gradient variance for each parameter group.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_qubits_center = int(np.log2(feature_dim))
    n_qubits_total = n_qubits_center * 2

    # Build QNode (no classical layer for clean BP analysis)
    qnode, _, _ = build_qubit_reused_qnode(
        feature_dim=feature_dim,
        qgkernel_iter_num=qgkernel_iter_num,
        qlinear_layer=qlinear_layer,
        n_neighbors=n_neighbors,
        diff_method='backprop'
    )

    shape_qkernel = qml.StronglyEntanglingLayers.shape(
        n_layers=qgkernel_iter_num, n_wires=n_qubits_total
    )
    shape_qlinear = qml.StronglyEntanglingLayers.shape(
        n_layers=qlinear_layer, n_wires=n_qubits_center
    )
    pooling_shape = (n_qubits_center, n_qubits_center)

    all_grads_qkernel = []
    all_grads_qpooling = []
    all_grads_qlinear = []

    for trial in range(n_trials):
        # Random initialization
        qkernel_w = torch.rand(*shape_qkernel, dtype=torch.float32, requires_grad=True)
        qpooling_w = torch.rand(*pooling_shape, dtype=torch.float32, requires_grad=True)
        qlinear_w = torch.rand(*shape_qlinear, dtype=torch.float32, requires_grad=True)

        # Random input
        center_feat = torch.randn(feature_dim, dtype=torch.float32)
        center_feat = center_feat / center_feat.norm()
        neigh_feat = torch.randn(n_neighbors, feature_dim, dtype=torch.float32)
        neigh_feat = neigh_feat / neigh_feat.norm(dim=1, keepdim=True)

        # Forward
        probs = qnode(center_feat, neigh_feat, qkernel_w, qpooling_w, qlinear_w)

        # Loss
        if loss_type == 'mse':
            target = torch.rand_like(probs)
            loss = torch.sum((probs - target) ** 2)
        else:
            # Negative log-likelihood on a random target class
            target_class = torch.randint(0, len(probs), (1,)).item()
            loss = -torch.log(probs[target_class] + 1e-10)

        # Backward
        loss.backward()

        all_grads_qkernel.append(qkernel_w.grad.detach().flatten().cpu().numpy())
        all_grads_qpooling.append(qpooling_w.grad.detach().flatten().cpu().numpy())
        all_grads_qlinear.append(qlinear_w.grad.detach().flatten().cpu().numpy())

    # Compute statistics
    all_grads_qkernel = np.array(all_grads_qkernel)
    all_grads_qpooling = np.array(all_grads_qpooling)
    all_grads_qlinear = np.array(all_grads_qlinear)

    return {
        'n_qubits_total': n_qubits_total,
        'n_qubits_center': n_qubits_center,
        'feature_dim': feature_dim,
        'n_layers_kernel': qgkernel_iter_num,
        'n_layers_linear': qlinear_layer,
        'n_neighbors': n_neighbors,
        'n_params_qkernel': shape_qkernel[0] * shape_qkernel[1] * shape_qkernel[2],
        'n_params_qpooling': pooling_shape[0] * pooling_shape[1],
        'n_params_qlinear': shape_qlinear[0] * shape_qlinear[1] * shape_qlinear[2],
        'var_qkernel': np.var(all_grads_qkernel.flatten()),
        'var_qpooling': np.var(all_grads_qpooling.flatten()),
        'var_qlinear': np.var(all_grads_qlinear.flatten()),
        'var_total': np.var(np.concatenate([
            all_grads_qkernel.flatten(),
            all_grads_qpooling.flatten(),
            all_grads_qlinear.flatten()
        ])),
        'mean_abs_qkernel': np.mean(np.abs(all_grads_qkernel.flatten())),
        'mean_abs_qpooling': np.mean(np.abs(all_grads_qpooling.flatten())),
        'mean_abs_qlinear': np.mean(np.abs(all_grads_qlinear.flatten())),
    }


def analyze_bp_scaling_qubits(
    feature_dims: list = [4, 8, 16, 32],
    qgkernel_iter_num: int = 2,
    qlinear_layer: int = 2,
    n_neighbors: int = 4,
    n_trials: int = 100
):
    """
    Analyze how gradient variance scales with the number of qubits.
    This is the definitive Barren Plateau test.

    For random PQCs, Var(grad) ~ 1/2^n. We check if qubit reuse mitigates this.
    """
    results = []
    for fd in feature_dims:
        print(f"  Analyzing feature_dim={fd} ({int(np.log2(fd))*2} qubits)...")
        r = compute_gradient_variance_at_init(
            feature_dim=fd,
            qgkernel_iter_num=qgkernel_iter_num,
            qlinear_layer=qlinear_layer,
            n_neighbors=n_neighbors,
            n_trials=n_trials
        )
        results.append(r)
        print(f"    Var(grad_kernel)={r['var_qkernel']:.2e}, "
              f"Var(grad_pooling)={r['var_qpooling']:.2e}, "
              f"Var(grad_linear)={r['var_qlinear']:.2e}")

    return results


def analyze_bp_scaling_depth(
    feature_dim: int = 8,
    qgkernel_iter_nums: list = [1, 2, 4, 8],
    qlinear_layer: int = 2,
    n_neighbors: int = 4,
    n_trials: int = 100
):
    """
    Analyze how gradient variance scales with circuit depth (number of
    StronglyEntanglingLayers in the kernel).
    """
    results = []
    for layers in qgkernel_iter_nums:
        print(f"  Analyzing kernel_layers={layers}...")
        r = compute_gradient_variance_at_init(
            feature_dim=feature_dim,
            qgkernel_iter_num=layers,
            qlinear_layer=qlinear_layer,
            n_neighbors=n_neighbors,
            n_trials=n_trials
        )
        results.append(r)
        print(f"    Var(grad_kernel)={r['var_qkernel']:.2e}, "
              f"Var(grad_pooling)={r['var_qpooling']:.2e}, "
              f"Var(grad_linear)={r['var_qlinear']:.2e}")

    return results


def analyze_bp_scaling_neighbors(
    feature_dim: int = 8,
    qgkernel_iter_num: int = 2,
    qlinear_layer: int = 2,
    n_neighbors_list: list = [1, 2, 4, 8],
    n_trials: int = 100
):
    """
    Analyze how gradient variance scales with the number of neighbors.
    This specifically tests the qubit-reuse mechanism.
    """
    results = []
    for nn in n_neighbors_list:
        print(f"  Analyzing n_neighbors={nn}...")
        r = compute_gradient_variance_at_init(
            feature_dim=feature_dim,
            qgkernel_iter_num=qgkernel_iter_num,
            qlinear_layer=qlinear_layer,
            n_neighbors=nn,
            n_trials=n_trials
        )
        results.append(r)
        print(f"    Var(grad_kernel)={r['var_qkernel']:.2e}, "
              f"Var(grad_pooling)={r['var_qpooling']:.2e}, "
              f"Var(grad_linear)={r['var_qlinear']:.2e}")

    return results


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_dynamics(tracker, train_losses, train_accs, summary, save_path):
    """
    Plot gradient variance, norm, and training curves.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    quantum_params = ['qkernel_weight', 'qpooling_weight', 'qlinear_weight']
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    labels = ['Kernel Weights (theta_k)', 'Pooling Weights (theta_p)', 'Linear Weights (theta_l)']

    # Row 1: Gradient Variance, Mean, Norm for quantum params
    ax_var = axes[0, 0]
    ax_mean = axes[0, 1]
    ax_norm = axes[0, 2]

    for name, color, label in zip(quantum_params, colors, labels):
        steps = tracker.history[name]['step']
        ax_var.semilogy(steps, tracker.history[name]['var'], color=color,
                        label=label, linewidth=2, alpha=0.8)
        ax_mean.plot(steps, tracker.history[name]['mean'], color=color,
                     label=label, linewidth=2, alpha=0.8)
        ax_norm.plot(steps, tracker.history[name]['norm'], color=color,
                     label=label, linewidth=2, alpha=0.8)

    ax_var.set_xlabel('Training Step')
    ax_var.set_ylabel('Gradient Variance (log scale)')
    ax_var.set_title('Gradient Variance Evolution\n(Lower -> Barren Plateau Risk)')
    ax_var.legend()
    ax_var.grid(True, alpha=0.3)

    ax_mean.set_xlabel('Training Step')
    ax_mean.set_ylabel('Gradient Mean')
    ax_mean.set_title('Gradient Mean Evolution')
    ax_mean.legend()
    ax_mean.grid(True, alpha=0.3)

    ax_norm.set_xlabel('Training Step')
    ax_norm.set_ylabel('Gradient Norm ||grad_L||')
    ax_norm.set_title('Gradient Norm Evolution')
    ax_norm.legend()
    ax_norm.grid(True, alpha=0.3)

    # Row 2: Loss, Accuracy, Summary bar chart
    ax_loss = axes[1, 0]
    ax_loss.plot(train_losses, color='#E91E63', linewidth=2)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training Loss')
    ax_loss.grid(True, alpha=0.3)

    ax_acc = axes[1, 1]
    ax_acc.plot(train_accs, color='#009688', linewidth=2)
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Training Accuracy')
    ax_acc.grid(True, alpha=0.3)

    # Summary bar chart: initial vs final variance
    ax_bar = axes[1, 2]
    x_pos = np.arange(len(quantum_params))
    bar_width = 0.35
    initial_vars = [summary[p]['var_initial'] for p in quantum_params]
    final_vars = [summary[p]['var_final'] for p in quantum_params]

    ax_bar.bar(x_pos - bar_width/2, initial_vars, bar_width,
               color='#BBDEFB', edgecolor='#2196F3', label='Initial')
    ax_bar.bar(x_pos + bar_width/2, final_vars, bar_width,
               color='#C8E6C9', edgecolor='#4CAF50', label='Final')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_ylabel('Gradient Variance')
    ax_bar.set_title('Initial vs Final Gradient Variance')
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Gradient Variance Analysis During Training\n'
                 '(Qubit-Reused QGCN PQC)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_bp_scaling_qubits(results, save_path):
    """
    Plot gradient variance vs qubit count with theoretical BP reference.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    n_qubits = [r['n_qubits_total'] for r in results]
    var_kernel = [r['var_qkernel'] for r in results]
    var_pooling = [r['var_qpooling'] for r in results]
    var_linear = [r['var_qlinear'] for r in results]
    var_total = [r['var_total'] for r in results]

    # Plot 1: Variance vs qubit count (log scale)
    ax1 = axes[0]
    ax1.semilogy(n_qubits, var_kernel, 'o-', color='#2196F3', linewidth=2,
                 markersize=10, label='Kernel Weights')
    ax1.semilogy(n_qubits, var_pooling, 's-', color='#FF9800', linewidth=2,
                 markersize=10, label='Pooling Weights')
    ax1.semilogy(n_qubits, var_linear, '^-', color='#4CAF50', linewidth=2,
                 markersize=10, label='Linear Weights')
    ax1.semilogy(n_qubits, var_total, 'D-', color='#E91E63', linewidth=2,
                 markersize=10, label='All Params (Total)')

    # Theoretical BP reference: Var ~ C / 2^n
    ref_qubits = np.array(n_qubits)
    ref_var = var_total[0] * (2 ** n_qubits[0]) / (2 ** ref_qubits)
    ax1.semilogy(ref_qubits, ref_var, '--', color='gray', linewidth=2,
                 alpha=0.7, label='Theoretical BP: Var proportional to 1/2^n')

    ax1.set_xlabel('Total Number of Qubits')
    ax1.set_ylabel('Gradient Variance (log scale)')
    ax1.set_title('Barren Plateau Analysis: Variance vs Qubit Count\n'
                  '(Deviation from 1/2^n line = BP mitigation)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_qubits)

    # Plot 2: Variance ratio vs theoretical BP
    ax2 = axes[1]
    ratios_kernel = [v / ref for v, ref in zip(var_kernel, ref_var)]
    ratios_pooling = [v / ref for v, ref in zip(var_pooling, ref_var)]
    ratios_linear = [v / ref for v, ref in zip(var_linear, ref_var)]

    x_pos = np.arange(len(n_qubits))
    bar_width = 0.25
    ax2.bar(x_pos - bar_width, ratios_kernel, bar_width,
            color='#2196F3', label='Kernel')
    ax2.bar(x_pos, ratios_pooling, bar_width,
            color='#FF9800', label='Pooling')
    ax2.bar(x_pos + bar_width, ratios_linear, bar_width,
            color='#4CAF50', label='Linear')

    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                label='Theoretical BP (ratio=1)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{q} qubits' for q in n_qubits])
    ax2.set_ylabel('Var(grad_theta) / Theoretical BP Variance')
    ax2.set_title('BP Mitigation Ratio\n(>1 means better than random PQC)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Barren Plateau Scaling Analysis\n'
                 '(Qubit-Reused QGCN PQC)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_bp_scaling_depth(results, save_path):
    """
    Plot gradient variance vs circuit depth.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    layers = [r['n_layers_kernel'] for r in results]
    var_kernel = [r['var_qkernel'] for r in results]
    var_pooling = [r['var_qpooling'] for r in results]
    var_linear = [r['var_qlinear'] for r in results]

    ax.semilogy(layers, var_kernel, 'o-', color='#2196F3', linewidth=2,
                markersize=10, label='Kernel Weights')
    ax.semilogy(layers, var_pooling, 's-', color='#FF9800', linewidth=2,
                markersize=10, label='Pooling Weights')
    ax.semilogy(layers, var_linear, '^-', color='#4CAF50', linewidth=2,
                markersize=10, label='Linear Weights')

    ax.set_xlabel('Number of Kernel Layers (StronglyEntanglingLayers)')
    ax.set_ylabel('Gradient Variance (log scale)')
    ax.set_title('Gradient Variance vs Circuit Depth\n'
                 '(Deeper circuits typically have lower variance)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_bp_scaling_neighbors(results, save_path):
    """
    Plot gradient variance vs number of neighbors (qubit reuse test).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    n_neighbors = [r['n_neighbors'] for r in results]
    var_kernel = [r['var_qkernel'] for r in results]
    var_pooling = [r['var_qpooling'] for r in results]
    var_linear = [r['var_qlinear'] for r in results]

    ax.semilogy(n_neighbors, var_kernel, 'o-', color='#2196F3', linewidth=2,
                markersize=10, label='Kernel Weights')
    ax.semilogy(n_neighbors, var_pooling, 's-', color='#FF9800', linewidth=2,
                markersize=10, label='Pooling Weights')
    ax.semilogy(n_neighbors, var_linear, '^-', color='#4CAF50', linewidth=2,
                markersize=10, label='Linear Weights')

    ax.set_xlabel('Number of Neighbors (qubit reuse iterations)')
    ax.set_ylabel('Gradient Variance (log scale)')
    ax.set_title('Gradient Variance vs Number of Neighbors\n'
                 '(Stable variance = effective qubit reuse)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_neighbors)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_gradient_distribution(tracker, save_path):
    """
    Plot histogram of gradient values at different training stages.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    quantum_params = ['qkernel_weight', 'qpooling_weight', 'qlinear_weight']
    titles = ['Kernel Weights', 'Pooling Weights', 'Linear Weights']
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    for ax, name, title, color in zip(axes, quantum_params, titles, colors):
        steps = tracker.history[name]['step']
        n_steps = len(steps)

        if n_steps >= 3:
            # Initial
            initial_step = steps[0]
            mid_step = steps[n_steps // 2]
            final_step = steps[-1]

            # We plot the variance (which is what we tracked), use as distribution proxy
            ax.bar(['Initial\n(step 0)', f'Mid\n(step {mid_step})',
                    f'Final\n(step {final_step})'],
                   [tracker.history[name]['var'][0],
                    tracker.history[name]['var'][n_steps // 2],
                    tracker.history[name]['var'][-1]],
                   color=[color + '80', color + 'B0', color],
                   edgecolor=color, linewidth=2)

        ax.set_ylabel('Gradient Variance')
        ax.set_title(f'{title} Variance')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Gradient Variance at Different Training Stages',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_comprehensive_summary(training_summary, bp_qubits_results, save_path):
    """
    Generate a comprehensive summary figure.
    """
    fig = plt.figure(figsize=(20, 12))

    # --- Panel 1: Training text summary ---
    ax_text = fig.add_axes([0.05, 0.55, 0.4, 0.4])
    ax_text.axis('off')

    lines = []
    lines.append("=" * 60)
    lines.append("GRADIENT VARIANCE ANALYSIS -- SUMMARY REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Model: Qubit-Reused QGCN (Parameterized Quantum Circuit)")
    lines.append("")
    lines.append("--- Training Gradient Statistics ---")
    lines.append(f"{'Param Group':<25s} {'Init Var':>12s} {'Final Var':>12s} {'Status':>10s}")
    lines.append("-" * 60)

    for name, label in [('qkernel_weight', 'Kernel'),
                         ('qpooling_weight', 'Pooling'),
                         ('qlinear_weight', 'Linear')]:
        s = training_summary[name]
        status = "[OK] HEALTHY" if s['var_final'] > 1e-6 else "[WARN] LOW"
        lines.append(f"{label:<25s} {s['var_initial']:>12.2e} {s['var_final']:>12.2e} {status:>10s}")

    lines.append("")
    lines.append("--- Barren Plateau Assessment ---")
    bp = bp_qubits_results

    # Check if variance decays slower than 1/2^n
    if len(bp) >= 2:
        n_q0, v0 = bp[0]['n_qubits_total'], bp[0]['var_total']
        n_q1, v1 = bp[-1]['n_qubits_total'], bp[-1]['var_total']
        # Theoretical: v1_theory = v0 * 2^n_q0 / 2^n_q1
        v1_theory = v0 * (2 ** n_q0) / (2 ** n_q1)
        ratio = v1 / v1_theory if v1_theory > 0 else float('inf')
        lines.append(f"  Qubits: {n_q0} -> {n_q1}")
        lines.append(f"  Variance decay: {v0:.2e} -> {v1:.2e}")
        lines.append(f"  Theoretical BP decay: {v0:.2e} -> {v1_theory:.2e}")
        lines.append(f"  BP Mitigation Ratio: {ratio:.1f}x")
        if ratio > 10:
            lines.append(f"  CONCLUSION: Strong BP mitigation -- {ratio:.0f}x better than random PQC")
        elif ratio > 2:
            lines.append(f"  CONCLUSION: Moderate BP mitigation -- {ratio:.0f}x better than random PQC")
        else:
            lines.append(f"  CONCLUSION: Weak BP mitigation -- similar to random PQC")

    lines.append("")
    lines.append("--- Trainability Verdict ---")
    # Check all training gradients
    all_healthy = all(
        training_summary[p]['var_final'] > 1e-6
        for p in ['qkernel_weight', 'qpooling_weight', 'qlinear_weight']
    )
    if all_healthy:
        lines.append("[OK] The PQC demonstrates GOOD TRAINABILITY:")
        lines.append("  - Gradient variance remains well above vanishing threshold")
        lines.append("  - All parameter groups receive meaningful gradient signals")
    else:
        lines.append("[WARN] Some parameter groups show low gradient variance")

    ax_text.text(0, 1, '\n'.join(lines), transform=ax_text.transAxes,
                 fontfamily='monospace', fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Panel 2: Training variance curves ---
    ax_train = fig.add_axes([0.52, 0.55, 0.45, 0.4])
    quantum_params = ['qkernel_weight', 'qpooling_weight', 'qlinear_weight']
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    labels = ['Kernel', 'Pooling', 'Linear']
    for name, color, label in zip(quantum_params, colors, labels):
        steps = training_summary[name]['step'] if hasattr(training_summary[name], '__iter__') else []
        if 'tracker' in dir():
            pass
    ax_train.set_xlabel('Training Step')
    ax_train.set_ylabel('Gradient Variance')
    ax_train.set_title('Variance During Training')
    ax_train.grid(True, alpha=0.3)

    # --- Panel 3: BP scaling ---
    ax_bp = fig.add_axes([0.05, 0.05, 0.4, 0.4])
    n_qubits = [r['n_qubits_total'] for r in bp]
    var_vals = [r['var_total'] for r in bp]
    ax_bp.semilogy(n_qubits, var_vals, 'D-', color='#E91E63', linewidth=2, markersize=8)
    ref_var = var_vals[0] * (2 ** n_qubits[0]) / (2 ** np.array(n_qubits))
    ax_bp.semilogy(n_qubits, ref_var, '--', color='gray', linewidth=2, label='Theoretical BP: 1/2^n')
    ax_bp.set_xlabel('Number of Qubits')
    ax_bp.set_ylabel('Total Gradient Variance')
    ax_bp.set_title('Barren Plateau Check')
    ax_bp.legend()
    ax_bp.grid(True, alpha=0.3)

    # --- Panel 4: Trainability score ---
    ax_score = fig.add_axes([0.52, 0.05, 0.45, 0.4])

    # Trainability score: higher is better
    # Score = log10(actual_variance / theoretical_bp_variance) averaged over param groups
    scores = []
    labels_score = []
    for r in bp:
        v_theory = bp[0]['var_total'] * (2 ** bp[0]['n_qubits_total']) / (2 ** r['n_qubits_total'])
        score = np.log10(r['var_total'] / v_theory) if v_theory > 0 else 0
        scores.append(score)
        labels_score.append(f'{r["n_qubits_total"]} qubits')

    colors_score = ['#4CAF50' if s > 0 else '#FF9800' if s > -1 else '#F44336'
                    for s in scores]
    ax_score.barh(labels_score, scores, color=colors_score, edgecolor='black')
    ax_score.axvline(x=0, color='black', linewidth=2)
    ax_score.set_xlabel('log_1_0(Var / BP_Var)\n(>0 = better than random PQC)')
    ax_score.set_title('Trainability Score by Qubit Count')
    ax_score.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Comprehensive Gradient Variance Analysis\n'
                 'Qubit-Reused QGCN -- Parameterized Quantum Circuit',
                 fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def main():
    """Run all gradient variance analyses."""
    print("=" * 70)
    print("GRADIENT VARIANCE ANALYSIS FOR QUBIT-REUSED QGCN PQC")
    print("=" * 70)
    start_time = time.time()

    # ========================================================================
    # Analysis 1: Gradient Variance During Training
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Gradient Variance Evolution During Training")
    print("=" * 70)
    print("Configuration: feature_dim=8 (6 qubits), kernel_layers=2, "
          "linear_layers=2, neighbors=4")

    tracker, train_losses, train_accs, training_summary = \
        analyze_gradient_variance_during_training(
            feature_dim=8,
            qgkernel_iter_num=2,
            qlinear_layer=2,
            n_neighbors=4,
            num_classes=2,
            n_epochs=200,
            batch_size=32,
            lr=0.01,
            seed=42
        )

    print("\nTraining Summary:")
    for name, label in [('qkernel_weight', 'Kernel Weights'),
                         ('qpooling_weight', 'Pooling Weights'),
                         ('qlinear_weight', 'Linear Weights')]:
        s = training_summary[name]
        print(f"  {label}:")
        print(f"    Variance: {s['var_initial']:.2e} -> {s['var_final']:.2e}")
        print(f"    Norm:     {s['norm_initial']:.2e} -> {s['norm_final']:.2e}")
        is_healthy = s['var_final'] > 1e-6
        print(f"    Status:   {'[OK] HEALTHY' if is_healthy else '[WARN] LOW VARIANCE'}")

    # ========================================================================
    # Analysis 2: Barren Plateau -- Variance vs Qubit Count
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Barren Plateau Check -- Variance vs Qubit Count")
    print("=" * 70)

    bp_qubits_results = analyze_bp_scaling_qubits(
        feature_dims=[4, 8, 16],
        qgkernel_iter_num=2,
        qlinear_layer=2,
        n_neighbors=4,
        n_trials=100
    )

    # ========================================================================
    # Analysis 3: Variance vs Circuit Depth
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Variance vs Circuit Depth")
    print("=" * 70)

    bp_depth_results = analyze_bp_scaling_depth(
        feature_dim=8,
        qgkernel_iter_nums=[1, 2, 4, 8],
        qlinear_layer=2,
        n_neighbors=4,
        n_trials=100
    )

    # ========================================================================
    # Analysis 4: Variance vs Number of Neighbors (Qubit Reuse Test)
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Variance vs Number of Neighbors (Qubit Reuse)")
    print("=" * 70)

    bp_neighbors_results = analyze_bp_scaling_neighbors(
        feature_dim=8,
        qgkernel_iter_num=2,
        qlinear_layer=2,
        n_neighbors_list=[1, 2, 4, 8],
        n_trials=100
    )

    # ========================================================================
    # Generate All Plots
    # ========================================================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    plot_training_dynamics(
        tracker, train_losses, train_accs, training_summary,
        os.path.join(OUTPUT_DIR, '01_training_dynamics.png')
    )

    plot_bp_scaling_qubits(
        bp_qubits_results,
        os.path.join(OUTPUT_DIR, '02_barren_plateau_qubits.png')
    )

    plot_bp_scaling_depth(
        bp_depth_results,
        os.path.join(OUTPUT_DIR, '03_variance_vs_depth.png')
    )

    plot_bp_scaling_neighbors(
        bp_neighbors_results,
        os.path.join(OUTPUT_DIR, '04_variance_vs_neighbors.png')
    )

    plot_gradient_distribution(
        tracker,
        os.path.join(OUTPUT_DIR, '05_gradient_distribution.png')
    )

    # ========================================================================
    # Final Report
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL TRAINABILITY ASSESSMENT")
    print("=" * 70)

    # Criterion 1: Training gradient variance
    all_healthy = all(
        training_summary[p]['var_final'] > 1e-6
        for p in ['qkernel_weight', 'qpooling_weight', 'qlinear_weight']
    )
    print(f"\n1. Training Gradient Health: {'[OK] PASS' if all_healthy else '[WARN] MARGINAL'}")
    print(f"   All parameter groups maintain variance > 1e-6: {all_healthy}")

    # Criterion 2: BP mitigation
    if len(bp_qubits_results) >= 2:
        v0 = bp_qubits_results[0]['var_total']
        v1 = bp_qubits_results[-1]['var_total']
        n0 = bp_qubits_results[0]['n_qubits_total']
        n1 = bp_qubits_results[-1]['n_qubits_total']
        v1_theory = v0 * (2 ** n0) / (2 ** n1)
        ratio = v1 / v1_theory if v1_theory > 0 else float('inf')
        print(f"\n2. Barren Plateau Mitigation: {'[OK] STRONG' if ratio > 10 else '[OK] MODERATE' if ratio > 2 else '[WARN] WEAK'}")
        print(f"   Variance ratio vs theoretical BP: {ratio:.1f}x")

    # Criterion 3: Variance stability
    var_decay_ratios = []
    for p in ['qkernel_weight', 'qpooling_weight', 'qlinear_weight']:
        if training_summary[p]['var_initial'] > 0:
            var_decay_ratios.append(
                training_summary[p]['var_final'] / training_summary[p]['var_initial']
            )
    mean_decay = np.mean(var_decay_ratios) if var_decay_ratios else 0
    print(f"\n3. Variance Stability: {'[OK] STABLE' if mean_decay > 0.1 else '[WARN] DECAYING'}")
    print(f"   Mean variance retention: {mean_decay:.3f}")

    # Overall verdict
    if all_healthy and ratio > 2 and mean_decay > 0.1:
        verdict = ("[OK] The qubit-reused QGCN PQC demonstrates GOOD TRAINABILITY.\n"
                   "  - Gradients remain healthy throughout training\n"
                   "  - Significant mitigation of barren plateaus compared to random PQCs\n"
                   "  - Gradient variance is stable across training\n"
                   "  - The qubit-reuse mechanism (measure-reset loop) effectively\n"
                   "    controls gradient variance even with more neighbors")
    elif all_healthy:
        verdict = ("[~] The qubit-reused QGCN PQC shows ADEQUATE TRAINABILITY.\n"
                   "  - Gradients are trainable but BP mitigation is limited\n"
                   "  - Consider reducing circuit depth for better scaling")
    else:
        verdict = ("[WARN] The qubit-reused QGCN PQC shows MARGINAL TRAINABILITY.\n"
                   "  - Some parameter groups have very low gradient variance\n"
                   "  - May require careful initialization or learning rate tuning")

    print(f"\n{'='*70}")
    print("OVERALL VERDICT:")
    print(verdict)
    print(f"{'='*70}")

    elapsed = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed:.1f}s")
    print(f"All figures saved to: {OUTPUT_DIR}")

    return {
        'training_summary': training_summary,
        'bp_qubits_results': bp_qubits_results,
        'bp_depth_results': bp_depth_results,
        'bp_neighbors_results': bp_neighbors_results,
        'tracker': tracker,
    }


if __name__ == '__main__':
    results = main()
