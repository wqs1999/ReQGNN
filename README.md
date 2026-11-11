# QuReGNN: A Resource-Efficient Quantum Graph Neural Network with Qubit-Reuse Mechanism

Official code for QuReGNN (a.k.a. ReQGNN), a quantum graph neural network that reuses qubits
through mid-circuit measurement, reset, and classically-conditioned gates, built with
[PennyLane](https://pennylane.ai/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).

## Model

`QRGCN` (defined in `QubitReusedQGCN.py`):

1. **Amplitude embedding** — the center-node feature vector and one neighbor's feature vector are
   amplitude-encoded into two qubit registers (features are first PCA-reduced to a power-of-two
   dimension so each node uses a single-qubit-width register per part).
2. **Quantum graph kernel** — `StronglyEntanglingLayers` over the joint register.
3. **Quantum pooling with qubit reuse** — the neighbor register is measured mid-circuit and reset;
   conditional `RY` rotations driven by the measured bits write the neighbor information into the
   surviving register, so the register is reused for every neighbor instead of growing with graph
   size.
4. **Quantum linear layer + classical readout** — probabilities of the surviving register feed a
   dropout + linear classifier.

The model also supports **parameter-perturbation noise** for robustness studies:

```python
model.configure_noise(scale=0.05, noise_type="gaussian",
                      targets=("qkernel", "qpool", "qlinear"),
                      train_only=True, seed=42)
```

## Repository Layout

```
ptc/           experiments on PTC_FM / PTC_FR / PTC_MM / PTC_MR
dblp/          experiments on DBLP
bp_analysis/   trainability analysis (barren-plateau / gradient-variance comparison of
               qubit reuse vs. a conventional wide-register circuit)
data/          raw datasets (processed files are regenerated automatically on first run)
```

Each experiment folder is self-contained (`main*.py`, `QubitReusedQGCN.py`, `CustomDataLoader.py`,
`Tool.py`) because the two experiment lines use slightly different model variants and training
utilities. Run scripts from inside the folder so that the default dataset path `../data` resolves.

## Requirements

- Python >= 3.10
- torch >= 2.0
- torch_geometric >= 2.3
- pennylane >= 0.36
- numpy, scikit-learn, matplotlib, networkx

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd ptc
python main4PTCFM.py      # or main4PTCFR.py / main4PTCMM.py / main4PTCMR.py
```

```bash
cd dblp
python main4DBLP.py       # or main4DBLP2Class.py
```

Training curves are saved as PNG images in the working directory (e.g. `PTC_FM_noise.png`).

## Data

- `PTC_FM/FR/MM/MR` — raw TUDataset files; PyTorch Geometric builds the processed dataset
  automatically on first run.
- `Cora` — Planetoid raw files.
- `DBLP` — raw files from the [Heterogeneous Graph Benchmark](https://github.com/THUDM/HGB)
  (`adjM.npz`, `features_*.npz/npy`, `labels.npy`, `node_types.npy`, `train_val_test_idx.npz`).

## License

This project is released under the MIT License.
