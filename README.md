# Characterising cardiac tissue properties with graph neural networks (STACOM 2026)

Accepted to be published at STACOM 2026. [Preprint](https://arxiv.org/abs/2608.15843) accessible.

## Abstract 
Characterising electrophysiological properties of cardiac tissue efficiently and accurately from spatially sparse intracardiac measurements is clinically important for localising ablation targets and improving arrhythmia treatment. We developed a graph neural network-based framework trained on synthetic electrogram signals on 2D flat surfaces to identify areas of interest in the context of cardiac ablation for premature ventricular complexes (PVCs). Our method achieved an average precision of 0.96, 0.97, and 0.95 for the detection of single-patch fibrosis, rapid depolarisation and high excitability, respectively. The trained model can then be applied to 2D curved surfaces with few-shot fine-tuning, demonstrating its generalisation capability. Future work will develop this framework further for clinical use in PVC ablation.

## Methods
### Data generation ([solveAP_2D_jax.py](solveAP_2D_jax.py))
We simulate action potential propagation by solving the isotropic [Aliev-Panfilov](https://www.sciencedirect.com/science/article/pii/0960077995000895?via%3Dihub) monodomain model:
```math
\begin{aligned}
\frac{\partial V}{\partial t} &= \nabla \cdot (\mathbf{D} \nabla V) - kV (V - a)(V - 1) - VW \\
\frac{\partial W}{\partial t} &= \left( \epsilon + \frac{\mu_1 W}{V + \mu_2} \right) \left( -W - kV (V - b - 1) \right)
\end{aligned}
```
where $V(\vec{x},t)$ is the transmembrane potential and $W(\vec{x},t)$ a latent field  that controls the recovery of the action potential.

The pathologies we model with the corresponding qualitative changes in parameters are summarised below:

| Pathology | Parameter change |
|---|---|
| Fibrosis | Lower $D$, the diffusion coefficient |
| Rapid depolarisation | Increase $k$, which controls the steepness of the upstroke |
| Heightened excitability | Negative $a$, the threshold above which the cell depolarises |

From the action potential field, $V (\vec{x}, t)$, we can compute the (unscaled) unipolar extracellular potentials, $\phi_e(\vec{x_e}, t)$, measured by electrodes placed on the endocardium at locations $\vec{x_e}$ through an unbounded Poisson equation solver:
```math
    \phi_e(\vec{x_e}, t) \propto \iint \frac{\nabla \cdot (\mathbf{D}\,\nabla V (\vec{x}, t))}
{|\vec{x} - \vec{x_e}|}
\, \text{d}S.
```

### CNN + GNN module
We encode the electrograms with 1D-CNNs and propagate the information from electrodes to tissue nodes with GATv2 layers. 
<img width="1555" height="556" alt="image" src="https://github.com/user-attachments/assets/dea70497-ada0-4ce1-8d05-bfd7eb2e19da" />


## Running the code
Generating 2D flat geometry data: run [`solveAP_2D_jax.py`](solveAP_2D_jax.py) to generate electrograms ➡️ [`preprocess_graphs.py`](preprocess_graphs.py) to create graphs to be used in training \\
Generating 2D curved surfaces data: run [``]() \\

Training the model:
