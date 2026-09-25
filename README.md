# Characterising cardiac tissue properties with graph neural networks (STACOM 2026)

This paper is accepted to be published at STACOM 2026 (Strasbourg, France). Read our [preprint](https://arxiv.org/abs/2608.15843) for more details. This is the repository holding the code that generated the results reported in the paper. 

Feel free to play around with it, raise an issue for any questions, or contact ching-en.chiu18@imperial.ac.uk to discuss research ideas and collaborate! :)

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


## Code structure
### 2D flat geoemtries
Generating 2D flat geometry data: run [`solveAP_2D_jax.py`](solveAP_2D_jax.py) to generate electrograms ➡️ [`preprocess_graphs.py`](preprocess_graphs.py) to create graphs to be used in training. An example simulation with fibrosis patches can be seen below:

https://github.com/user-attachments/assets/54d2a404-ba9f-45e7-a687-a4f2ba997c33

Training the model:
Pretrain the CNN encoder module with [`train_cnn_electrode.py`](train_cnn_electrode.py) ➡️ train the GNN module with [`train_gnn_with_pretrained_cnn.py`](train_gnn_with_pretrained_cnn.py)

  
Evaluating the model:
- evaluate the performance of baseline CNN model by simply extrapolating the pretrained CNN's predictions to the tissue nodes: [`extrapolate_cnn_to_tm.py`](extrapolate_cnn_to_tm.py)
- evaluate the proposed GNN model with [`evaluate_gnn.py`](evaluate_gnn.py)

### 2D curved surfaces 
All related code is in [`2d_curved_surf`](2d_curved_surf), where you can
- generate 2D curved surfaces data with [`solveAP_surface_fenicsx_phie.py`](solveAP_surface_fenicsx_phie.py)
- create graphs to be used for training with [`surface_to_graph.py`](surface_to_graph.py)
- finetune pretrained models with [`surface_to_graph.py`](surface_to_graph.py)
- evaluate the performance with [`eval_surface_gnn.py`](eval_surface_gnn.py)

### Example data
An example of the simulation and graph data can be found in [`example_data/`](example_data).
