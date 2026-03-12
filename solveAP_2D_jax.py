# solveAP_2D_jax.py
# A python/JAX adaption of the MATLAB code: AlievPanfilov2D_RK_Istim_heter.m

# Key differences vs MATLAB:
# - Pure functions + JIT + lax.scan for the time loop
# - Stencil ops implemented with jnp.roll (central diffs / 5-pt Laplacian)
# - Outputs saved every gathert steps
#
# Notes:
# - Arrays include a 1-cell "ghost" border (size = ncells+2), matching your MATLAB code.
# - Neumann (no-flux) BC enforced by copying ghost cells from adjacent interior.
# - del2 scaling: MATLAB's del2 in 2D effectively returns ~ (Laplacian)/4, and you multiply by 4.
#   Here we compute Laplacian directly, so diffusion term is D * Laplacian + Dx*Vx + Dy*Vy.
#
# Run:
#   params = Params(...)
#   out = simulate(params)
#   Vsave, Wsave = out["Vsav"], out["Wsav"]
#%%
# =============================================================================
# IMPORTS
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.path import Path
import imageio
from scipy.signal import convolve2d
from PIL import Image, ImageDraw, ImageFont
import gc

Array = jnp.ndarray

# =============================================================================
# GPU CONFIGURATION & OPTIMIZATION GUIDE
# =============================================================================
# Check available devices and enable GPU if available
print("JAX devices:", jax.devices())
print("Default backend:", jax.default_backend())

# To force GPU usage (if available), uncomment:
# jax.config.update('jax_platform_name', 'gpu')

# For better GPU performance with large arrays:
# jax.config.update('jax_enable_x64', False)  # Use float32 (default, faster on GPU)

# =============================================================================
# GPU SETUP INSTRUCTIONS
# =============================================================================
# 1. INSTALL JAX WITH GPU SUPPORT (on server with CUDA):
#    pip install --upgrade "jax[cuda12]"  # For CUDA 12
#    pip install --upgrade "jax[cuda11]"  # For CUDA 11
#
# 2. VERIFY GPU IS DETECTED:
#    Run this file and check the output above shows "gpu" device
#
# 3. EXPECTED SPEEDUPS WITH GPU:
#    - Simulation (simulate()): 5-20x faster depending on grid size
#    - Phie calculation (calc_phie_jax): 10-50x faster with many electrodes
#    - Larger grids benefit more from GPU
#
# 4. CURRENT CODE IS GPU-READY:
#    - Uses JIT compilation (@jax.jit)
#    - Uses vectorized operations (vmap, lax.scan)
#    - Avoids Python loops in hot paths
#    - All array operations use jnp (not np) in compiled functions
#
# 5. MONITORING GPU USAGE:
#    - Use nvidia-smi to check GPU utilization
#    - First run will be slow (compilation), subsequent runs fast
#
# 6. MEMORY CONSIDERATIONS:
#    - GPU has limited memory vs CPU
#    - For 100x100 grid, 100 electrodes, 1500 timesteps: ~few GB
#    - If OOM, reduce: ncells, num electrodes, or batch simulations
# =============================================================================

@dataclass(frozen=True)
class Params:
    # Time / pacing
    dt: float               # AU
    tend: float             # AU
    BCL: float              # AU
    ncyc: int
    extra: float            
    stimdur: float          # AU
    gathert: int            # save every this many steps (integer iterations)

    # Grid
    ncells: int             # interior size (ncells x ncells)
    h: float                # spatial step

    # Model
    k: float
    mu1: float
    mu2: float
    epsi: float
    D_scalar: float         # baseline D0

    # Fields (spatial heterogeneity)
    a0: float               # baseline a (scalar)
    b0: float               # baseline b (scalar)
    a_field: Optional[np.ndarray] = None  # (X,X) including ghost; if None -> a0 everywhere
    b_field: Optional[np.ndarray] = None  # (X,X) including ghost; if None -> b0 everywhere
    D_matrix: Optional[np.ndarray] = None # (X,X) including ghost; if None -> D_scalar everywhere

    # Stimulation
    stim_mask: Optional[np.ndarray] = None  # interior (ncells,ncells) boolean/0-1
    stim_amp_scale: float = 0.1             # Ia = stim_amp_scale * stim_mask (your MATLAB uses 0.1*stimgeo)
    
    # Spiral wave generation (cross-field stimulation)
    cross_stim_mask: Optional[np.ndarray] = None  # interior (ncells,ncells) boolean/0-1 for cross-field stimulus
    cross_stim_time: float = 42.0                  # AU, time at which cross-field stimulus is applied
    cross_stim_duration: float = 1.0               # AU, duration of cross-field stimulus
    
    cyclic: bool = False                    # only Neumann BC implemented here


def _pad_with_ghost(interior: np.ndarray, ghost_value: float = 0.0) -> np.ndarray:
    """Pad (ncells,ncells) -> (ncells+2,ncells+2) with ghost border."""
    return np.pad(interior, pad_width=1, mode="constant", constant_values=ghost_value)


def _ensure_fields(params: Params) -> Dict[str, np.ndarray]:
    """Build full-grid (with ghost) fields on host as numpy arrays."""
    X = params.ncells + 2

    if params.D_matrix is None:
        D = np.full((X, X), params.D_scalar, dtype=np.float32)
    else:
        D = np.asarray(params.D_matrix, dtype=np.float32)
        assert D.shape == (X, X)

    if params.a_field is None:
        a = np.full((X, X), params.a0, dtype=np.float32)
    else:
        a = np.asarray(params.a_field, dtype=np.float32)
        assert a.shape == (X, X)

    if params.b_field is None:
        b = np.full((X, X), params.b0, dtype=np.float32)
    else:
        b = np.asarray(params.b_field, dtype=np.float32)
        assert b.shape == (X, X)

    if params.stim_mask is None:
        stim_interior = np.zeros((params.ncells, params.ncells), dtype=np.float32)
    else:
        stim_interior = np.asarray(params.stim_mask, dtype=np.float32)
        assert stim_interior.shape == (params.ncells, params.ncells)

    stim_full = _pad_with_ghost(stim_interior, ghost_value=0.0).astype(np.float32)
    
    # Cross-field stimulation for spiral wave
    if params.cross_stim_mask is None:
        cross_stim_interior = np.zeros((params.ncells, params.ncells), dtype=np.float32)
    else:
        cross_stim_interior = np.asarray(params.cross_stim_mask, dtype=np.float32)
        assert cross_stim_interior.shape == (params.ncells, params.ncells)
    
    cross_stim_full = _pad_with_ghost(cross_stim_interior, ghost_value=0.0).astype(np.float32)

    return {"D": D, "a": a, "b": b, "stim_full": stim_full, "cross_stim_full": cross_stim_full}


def _apply_neumann_bc(V: Array) -> Array:
    """No-flux BC: copy adjacent interior into ghost border."""
    # top/bottom rows
    V = V.at[0, :].set(V[1, :])   # i.e. V[0,: ] = V[1,:]
    V = V.at[-1, :].set(V[-2, :])
    # left/right cols
    V = V.at[:, 0].set(V[:, 1])
    V = V.at[:, -1].set(V[:, -2])
    return V


def _grad_central(F: Array, h: float) -> Tuple[Array, Array]:
    """Central differences using roll; assumes ghost cells valid for BC."""
    Fx = (jnp.roll(F, -1, axis=0) - jnp.roll(F, 1, axis=0)) / (2.0 * h)
    Fy = (jnp.roll(F, -1, axis=1) - jnp.roll(F, 1, axis=1)) / (2.0 * h)
    return Fx, Fy


def _laplacian_5pt(F: Array, h: float) -> Array:
    """5-point Laplacian on full grid."""
    return (
        jnp.roll(F, -1, axis=0)
        + jnp.roll(F, 1, axis=0)
        + jnp.roll(F, -1, axis=1)
        + jnp.roll(F, 1, axis=1)
        - 4.0 * F
    ) / (h * h)


def _rhs_alpan(V: Array, W: Array, Istim: Array, a: Array, b: Array, D: Array, Dx: Array, Dy: Array,
               h: float, k: float, mu1: float, mu2: float, epsi: float) -> Tuple[Array, Array]:
    """Compute dV/dt and dW/dt."""
    # Ensure BC before stencils
    V = _apply_neumann_bc(V)

    Vx, Vy = _grad_central(V, h)
    lapV = _laplacian_5pt(V, h)

    diffusion = D * lapV + Dx * Vx + Dy * Vy

    dWdt = (epsi + mu1 * W / (mu2 + V)) * (-W - k * V * (V - b - 1.0))
    dVdt = (-k * V * (V - a) * (V - 1.0) - W * V) + diffusion + Istim

    return dVdt, dWdt


def _rk4_step(V: Array, W: Array, Istim: Array, a: Array, b: Array, D: Array, Dx: Array, Dy: Array,
              dt: float, h: float, k: float, mu1: float, mu2: float, epsi: float) -> Tuple[Array, Array]:
    """One explicit RK4 step."""
    k1V, k1W = _rhs_alpan(V, W, Istim, a, b, D, Dx, Dy, h, k, mu1, mu2, epsi)
    k2V, k2W = _rhs_alpan(V + 0.5 * dt * k1V, W + 0.5 * dt * k1W, Istim, a, b, D, Dx, Dy, h, k, mu1, mu2, epsi)
    k3V, k3W = _rhs_alpan(V + 0.5 * dt * k2V, W + 0.5 * dt * k2W, Istim, a, b, D, Dx, Dy, h, k, mu1, mu2, epsi)
    k4V, k4W = _rhs_alpan(V + dt * k3V, W + dt * k3W, Istim, a, b, D, Dx, Dy, h, k, mu1, mu2, epsi)

    Vn = V + (dt / 6.0) * (k1V + 2.0 * k2V + 2.0 * k3V + k4V)
    Wn = W + (dt / 6.0) * (k1W + 2.0 * k2W + 2.0 * k3W + k4W)

    # Enforce BC on updated V (as in MATLAB)
    Vn = _apply_neumann_bc(Vn)
    return Vn, Wn


def simulate(params: Params,
             V0: Optional[np.ndarray] = None,
             W0: Optional[np.ndarray] = None,
             dtype=jnp.float32) -> Dict[str, np.ndarray]:
    """
    Run the simulation fully in JAX (JIT+scan), returning saved states.

    Returns dict:
      Vsav: (nsaves, ncells, ncells) interior only (ghost removed)
      Wsav: (nsaves, ncells, ncells)
      t_sav: (nsaves,) times corresponding to saved frames
    """
    if params.cyclic:
        raise NotImplementedError("Periodic BC not implemented in this skeleton.")

    fields = _ensure_fields(params)
    X = params.ncells + 2

    # Host -> device constants
    a = jnp.asarray(fields["a"], dtype=dtype)
    b = jnp.asarray(fields["b"], dtype=dtype)
    D = jnp.asarray(fields["D"], dtype=dtype)
    stim_full = jnp.asarray(fields["stim_full"], dtype=dtype)
    cross_stim_full = jnp.asarray(fields["cross_stim_full"], dtype=dtype)

    # Precompute grad(D) once (your D_matrix is static)
    Dx, Dy = _grad_central(D, params.h)

    # Initial conditions (with ghost)
    if V0 is None:
        V_init = np.zeros((X, X), dtype=np.float32)
    else:
        V_init = np.asarray(V0, dtype=np.float32)
        assert V_init.shape == (X, X)

    if W0 is None:
        W_init = np.full((X, X), 0.01, dtype=np.float32)
    else:
        W_init = np.asarray(W0, dtype=np.float32)
        assert W_init.shape == (X, X)

    V_init = jnp.asarray(V_init, dtype=dtype)
    W_init = jnp.asarray(W_init, dtype=dtype)

    # Timesteps
    nsteps = int(np.floor(params.tend / params.dt))
    # Save every gathert iterations like MATLAB: mod(ind,gathert)==0
    nsaves = nsteps // params.gathert

    # Preallocate save buffers on device
    Vsav0 = jnp.zeros((nsaves, params.ncells, params.ncells), dtype=dtype)
    Wsav0 = jnp.zeros((nsaves, params.ncells, params.ncells), dtype=dtype)

    dt = params.dt
    h = params.h

    k = params.k
    mu1 = params.mu1
    mu2 = params.mu2
    epsi = params.epsi

    BCL = params.BCL
    ncyc = params.ncyc
    stimdur = params.stimdur
    cross_stim_time = params.cross_stim_time
    cross_stim_duration = params.cross_stim_duration

    # Ia = 0.1*stimgeo in MATLAB; here stim_full already includes ghost border
    Ia = params.stim_amp_scale * stim_full
    Ia_cross = params.stim_amp_scale * cross_stim_full

    def step_fn(carry, n):
        V, W, kk, save_idx, Vsav, Wsav = carry
        t = (n + 1) * dt  # MATLAB loop starts at dt:dt:tend

        # Determine stimulation window for current cycle kk (integer)
        kk_f = kk.astype(dtype)  # for t comparisons with BCL*kk
        stim_on = (kk < ncyc) & (t >= BCL * kk_f) & (t < (BCL * kk_f + stimdur * 2.0))
        
        # Cross-field stimulation for spiral wave (applied at specific time)
        cross_stim_on = (t >= cross_stim_time) & (t < (cross_stim_time + cross_stim_duration))
        
        # Combine both stimulations
        Istim_primary = jnp.where(stim_on, Ia, jnp.zeros_like(Ia))
        Istim_cross = jnp.where(cross_stim_on, Ia_cross, jnp.zeros_like(Ia_cross))
        Istim = Istim_primary + Istim_cross

        # Update kk when stimulation window has ended
        kk_inc = (kk < ncyc) & (t >= (BCL * kk_f + stimdur * 2.0))
        kk_next = kk + kk_inc.astype(kk.dtype)

        # RK4 step
        Vn, Wn = _rk4_step(V, W, Istim, a, b, D, Dx, Dy, dt, h, k, mu1, mu2, epsi)

        # Save every gathert steps
        do_save = ((n + 1) % params.gathert) == 0

        def save_branch(args):
            Vn, Wn, save_idx, Vsav, Wsav = args
            V_interior = Vn[1:-1, 1:-1]
            W_interior = Wn[1:-1, 1:-1]
            Vsav = Vsav.at[save_idx].set(V_interior)
            Wsav = Wsav.at[save_idx].set(W_interior)
            return (save_idx + 1, Vsav, Wsav)

        def nosave_branch(args):
            _, _, save_idx, Vsav, Wsav = args
            return (save_idx, Vsav, Wsav)

        save_idx2, Vsav2, Wsav2 = lax.cond(
            do_save,
            save_branch,
            nosave_branch,
            (Vn, Wn, save_idx, Vsav, Wsav)
        ) # lax.cond: conditionally apply true_fun (save_branch) or false_fun (nosave_branch).

        # Optional: early abort on NaN/Inf (JAX-friendly way is to track a flag;
        # true early-break isn't supported inside scan)
        return (Vn, Wn, kk_next, save_idx2, Vsav2, Wsav2), None

    # Carry: (V, W, kk, save_idx, Vsav, Wsav)
    carry0 = (V_init, W_init, jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32), Vsav0, Wsav0)

    # JIT the whole scan
    @jax.jit
    def run():
        carryT, _ = lax.scan(step_fn, carry0, jnp.arange(nsteps))
        _, _, _, _, Vsav, Wsav = carryT
        return Vsav, Wsav

    Vsav_dev, Wsav_dev = run()

    # Bring back to host numpy
    Vsav = np.array(Vsav_dev)
    Wsav = np.array(Wsav_dev)

    # Saved times (correspond to ind = gathert, 2*gathert, ...)
    t_sav = (np.arange(1, nsaves + 1) * params.gathert) * params.dt

    return {"Vsav": Vsav, "Wsav": Wsav, "t_sav": t_sav}


# -------------------------------------------------------------------------
# Generate rectangular fibrotic patches
# -------------------------------------------------------------------------

def make_D_with_rect_patches(ncells: int, D0: float, Dfac: float, npatches: int,
                             patch_w_rng=(15, 25), patch_h_rng=(15, 25),
                             margin=7, min_sep=5, seed: int = 0) -> Tuple[np.ndarray, list]:
    """
    Create D_matrix with ghost border and non-overlapping rectangular patches in the interior.
    
    Parameters:
    -----------
    ncells : int
        Interior grid size (domain will be ncells+2 to include ghost cells)
    D0 : float
        Baseline diffusion coefficient
    Dfac : float
        Diffusion reduction factor (D_fibrotic = D0 * Dfac)
    npatches : int
        Number of rectangular patches
    patch_w_rng, patch_h_rng : tuple
        (min, max) range for patch width and height
    margin : int
        Minimum distance from domain boundary
    min_sep : int
        Minimum separation between patches
    seed : int
        Random seed
        
    Returns:
    --------
    D_matrix : (X, X) array where X=ncells+2
        Diffusion coefficient map with ghost border
    fiblocs : list of arrays
        Each element is (N_patch, 2) array of [row, col] coordinates (0-based indices)
        
    Note:
    -----
    Arrays use (row, col) indexing where:
    - row index increases from 0 to X-1 (bottom to top when plotted)
    - col index increases from 0 to X-1 (left to right when plotted)
    """
    rng = np.random.default_rng(seed)
    X = ncells + 2
    D = np.full((X, X), D0, dtype=np.float32)

    occ = np.zeros((X, X), dtype=bool)
    fiblocs: list[np.ndarray] = []

    n = 0
    while n < npatches:
        w = rng.integers(patch_w_rng[0], patch_w_rng[1] + 1)
        h = rng.integers(patch_h_rng[0], patch_h_rng[1] + 1)

        # choose bottom-left in interior, keep away from boundaries/ghosts
        i0 = rng.integers(margin, ncells - w - margin + 1)
        j0 = rng.integers(margin, ncells - h - margin + 1)

        I = np.arange(i0, i0 + w + 1)  # +1 to mimic your MATLAB inclusive range
        J = np.arange(j0, j0 + h + 1)

        # separation check (in full-grid coords)
        p = min_sep
        r0 = max(0, I[0] - p)
        r1 = min(X, I[-1] + p + 1)
        c0 = max(0, J[0] - p)
        c1 = min(X, J[-1] + p + 1)

        if not occ[r0:r1, c0:c1].any():
            # Mark occupied region
            occ[np.ix_(I, J)] = True
            D[np.ix_(I, J)] = D0 * Dfac
            
            # Store pixel coordinates [row, col] for this patch
            rows, cols = np.where(occ & ~(occ ^ occ))  # get just this patch
            # Better: directly compute from I, J
            patch_rows, patch_cols = np.meshgrid(I, J, indexing='ij')
            fiblocs.append(np.stack([patch_rows.ravel(), patch_cols.ravel()], axis=1).astype(np.int32))
            n += 1
    
    # visualise the patches generated
    plt.imshow(occ, origin="lower", cmap="gray")
    plt.title(f"Rectangular patches (n={npatches})")
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.show()
    
    return D, occ
#%% 
# -------------------------------------------------------------------------
# Generate irregularly shaped fibrotic patches
# Caution ⚠️: This easily blows up the simulation with Dfac=0.1, but Dfac>=0.2 seems okay.
# -------------------------------------------------------------------------

def _poly_area(x: np.ndarray, y: np.ndarray) -> float:
    # Shoelace (x,y as 1D arrays, polygon assumed closed implicitly)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _dilate_mask(M: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return M
    k = np.ones((2 * pad + 1, 2 * pad + 1), dtype=np.uint8)
    return convolve2d(M.astype(np.uint8), k, mode="same", boundary="fill", fillvalue=0) > 0


def _estimate_aspect(mask: np.ndarray) -> float:
    pts = np.argwhere(mask)  # (N,2) [x(row), y(col)]
    if pts.shape[0] < 3:
        return 1.0
    pts = pts.astype(np.float64)
    pts -= pts.mean(axis=0, keepdims=True)
    C = (pts.T @ pts) / pts.shape[0]
    s = np.sqrt(np.linalg.eigvalsh(C))
    s = np.sort(s)[::-1]
    return float(s[0] / max(s[1], 1e-12))


def irregular_shape_generation(ncells: int, D0: float, Dfac: float, npatches: int, seed: int | None = None):
    """
    Generate irregularly shaped fibrotic patches.
    
    Parameters:
    -----------
    ncells : int
        Interior grid size (domain will be ncells+2 to include ghost cells)
    D0 : float
        Baseline diffusion coefficient
    Dfac : float
        Diffusion reduction factor (D_fibrotic = D0 * Dfac)
    npatches : int
        Number of patches to generate
    seed : int or None
        Random seed
        
    Returns:
    --------
    D_matrix : (X, X) array where X=ncells+2
        Diffusion coefficient map with ghost border
    fiblocs : list of arrays
        Each element is (N_patch, 2) array of [row, col] coordinates (0-based indices)
        
    Note:
    -----
    Arrays use (row, col) indexing where:
    - row index increases from 0 to X-1 (bottom to top when plotted)
    - col index increases from 0 to X-1 (left to right when plotted)
    """
    rng = np.random.default_rng(seed)

    X = ncells + 2
    Y = ncells + 2

    # Initialize D_matrix with baseline diffusion
    D_matrix = np.full((X, Y), D0, dtype=np.float32)

    # Defaults matching MATLAB code
    area_min, area_max = 300, 450
    max_tries = 800
    min_vertices, max_vertices = 6, 12
    p = 10  # minimum separation between patches
    margin = 5  # minimum distance from interior boundaries

    elong_prob = 0.45
    max_aspect = 2.0
    roughness = 0.3
    
    # Interior boundaries: indices 1 to ncells (in full grid coords)
    # With margin, patches must stay in [1+margin, ncells-margin+1] = [6, ncells-4] for ncells=100
    interior_min = 1 + margin  # e.g., 6
    interior_max = ncells + 1 - margin  # e.g., 97 for ncells=100

    mask_all = np.zeros((X, Y), dtype=bool)
    fiblocs: list[np.ndarray] = []

    # Grid of pixel centers in 1-based coordinates (to mirror MATLAB)
    # meshgrid with indexing='ij': Xq varies along axis 0 (rows), Yq varies along axis 1 (cols)
    # So pts_grid has points [x, y] where x is row-index, y is col-index
    Xq, Yq = np.meshgrid(np.arange(1, X + 1), np.arange(1, Y + 1), indexing="ij")
    pts_grid = np.stack([Xq.ravel(), Yq.ravel()], axis=1)

    n, tries = 0, 0
    while n < npatches and tries < max_tries:
        tries += 1

        A_tgt = int(rng.integers(area_min, area_max + 1))
        # Choose center within allowed interior region
        cx = int(rng.integers(interior_min + p, interior_max - p + 1))
        cy = int(rng.integers(interior_min + p, interior_max - p + 1))

        nv = int(rng.integers(min_vertices, max_vertices + 1))
        theta = np.linspace(0.0, 2.0 * np.pi, nv, endpoint=False)
        r0 = np.sqrt(A_tgt / np.pi)
        r = r0 * (1.0 - roughness / 2.0 + roughness * rng.random(nv))

        # Base polygon around origin
        px = r * np.cos(theta)
        py = r * np.sin(theta)

        # Optional elongation (area-preserving)
        if rng.random() < elong_prob:
            s = float(np.exp(np.log(max_aspect) * rng.random()))  # log-uniform in [1,max_aspect]
            ang = float(2.0 * np.pi * rng.random())
            c, sA = np.cos(ang), np.sin(ang)
            R = np.array([[c, -sA], [sA, c]])
            S = np.array([[s, 0.0], [0.0, 1.0 / s]])              # det=1
            A = R @ S @ R.T
            XY = A @ np.vstack([px, py])
            px, py = XY[0], XY[1]

        # Rescale to hit target area
        A_now = _poly_area(px, py)
        if A_now <= 0:
            continue
        s_area = np.sqrt(A_tgt / A_now)
        px *= s_area
        py *= s_area

        # Translate to center (cx, cy)
        x = px + cx
        y = py + cy

        # Fit inside interior bounds by shrinking about (cx,cy) if needed
        # Must stay within [interior_min, interior_max]
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        if xmin < interior_min or ymin < interior_min or xmax > interior_max or ymax > interior_max:
            eps = 1e-12
            sx = min((cx - interior_min) / max(eps, cx - xmin), (interior_max - cx) / max(eps, xmax - cx))
            sy = min((cy - interior_min) / max(eps, cy - ymin), (interior_max - cy) / max(eps, ymax - cy))
            sfit = max(0.0, min(sx, sy, 1.0))
            x = cx + sfit * (x - cx)
            y = cy + sfit * (y - cy)
            if x.min() < interior_min or y.min() < interior_min or x.max() > interior_max or y.max() > interior_max:
                continue

        # Rasterize polygon into mask using matplotlib Path
        poly = Path(np.stack([x, y], axis=1))
        inside = poly.contains_points(pts_grid, radius=1e-9)  # small radius includes boundary-ish
        mask = inside.reshape(X, Y)

        # Buffer check: forbid overlap within p pixels
        if (_dilate_mask(mask_all, p) & mask).any():
            continue

        # Accept patch: modify diffusion tensors inside mask
        D_matrix[mask] = D0 * Dfac

        mask_all |= mask

        # Store patch pixel coordinates: np.where returns (rows, cols)
        # Store as [row, col] coordinates (0-based indices)
        rows, cols = np.where(mask)
        fiblocs.append(np.stack([rows, cols], axis=1).astype(np.int32))

        asp = _estimate_aspect(mask)
        print(f"Patch {n+1}: {mask.sum()} px, aspect~{asp:.2f}, center=[{cx},{cy}]")
        n += 1

    if n < npatches:
        print(f"Warning: placed {n}/{npatches} patches. Consider reducing p/max_aspect or widening area range.")
    # visualise the patches generated
    plt.imshow(mask_all, origin="lower", cmap="gray")
    plt.title(f"Irregular patches (n={n})")
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.show()

    return D_matrix, fiblocs

#%%
def generate_random_stim_mask(
    ncells: int,
    stim_size: int = 10,
    seed: Optional[int] = None,
    forbidden_mask: Optional[np.ndarray] = None,
    max_tries: int = 500,
) -> np.ndarray:
    """Generate a random stimulation mask with a square patch, avoiding forbidden areas."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    stim_mask = np.zeros((ncells, ncells), dtype=np.float32)
    half_size = stim_size // 2
    
    if forbidden_mask is not None:
        forbidden_mask = np.asarray(forbidden_mask, dtype=bool)
        if forbidden_mask.shape != (ncells, ncells):
            raise ValueError("forbidden_mask must have shape (ncells, ncells)")

    # Generate random center position, ensuring the stim patch fits within boundaries
    centre_stim_x = None
    centre_stim_y = None
    for _ in range(max_tries):
        cx = rng.integers(half_size, ncells - half_size)
        cy = rng.integers(half_size, ncells - half_size)
        if forbidden_mask is None:
            centre_stim_x, centre_stim_y = cx, cy
            break
        patch = forbidden_mask[cx - half_size:cx + half_size, cy - half_size:cy + half_size]
        if not patch.any():
            centre_stim_x, centre_stim_y = cx, cy
            break

    if centre_stim_x is None:
        raise ValueError("Failed to place stimulus outside fibrotic areas; increase max_tries or reduce stim_size.")

    stim_mask[centre_stim_x - half_size:centre_stim_x + half_size,
              centre_stim_y - half_size:centre_stim_y + half_size] = 1.0
    
    return stim_mask, (centre_stim_x, centre_stim_y)


def generate_spiral_wave_stim_masks(ncells: int, 
                                    planar_width: int = 5,
                                    cross_width: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate stimulation masks for spiral wave generation.
    
    Follows the MATLAB approach:
    - First stimulus: planar wave from top (stimulates top rows)
    - Cross-field stimulus: wave from left (stimulates left columns)
    
    Parameters:
    -----------
    ncells : int
        Interior grid size (ncells x ncells)
    planar_width : int
        Number of rows to stimulate at the top for planar wave (default: 5)
    cross_width : int, optional
        Width of cross-field stimulus from left. If None, uses ncells//3 (default in MATLAB)
        
    Returns:
    --------
    stim_mask : (ncells, ncells) ndarray
        Primary stimulation mask for planar wave (top rows)
    cross_stim_mask : (ncells, ncells) ndarray
        Cross-field stimulation mask (left columns)
        
    Example:
    --------
    >>> stim_mask, cross_stim_mask = generate_spiral_wave_stim_masks(100, planar_width=5)
    >>> params = Params(..., stim_mask=stim_mask, cross_stim_mask=cross_stim_mask,
    ...                 cross_stim_time=42.0, cross_stim_duration=1.0)
    """
    if cross_width is None:
        cross_width = ncells // 3
    
    # Primary stimulus: top rows (planar wave going downward)
    stim_mask = np.zeros((ncells, ncells), dtype=np.float32)
    stim_mask[:planar_width, :] = 1.0
    
    # Cross-field stimulus: left columns (wave going rightward)
    cross_stim_mask = np.zeros((ncells, ncells), dtype=np.float32)
    cross_stim_mask[:, :cross_width] = 1.0
    
    return stim_mask, cross_stim_mask


def run_single_simulation(ncells: int, 
                          patch_type: str,
                          npatches: int,
                          seed: int,
                          D0: float = 0.1,
                          Dfac: float = 0.2,
                          save_dir: Optional[str] = None,
                          return_data: bool = True) -> Dict:
    """
    Run a single AP simulation with either 'irregular' or 'rectangular' fibrotic patches.
    
    Parameters:
    -----------
    ncells : int
        Grid size (interior)
    patch_type : str
        Either 'irregular' or 'rectangular'
    npatches : int
        Number of fibrotic patches
    seed : int
        Random seed for reproducibility
    D0 : float
        Baseline diffusion coefficient
    Dfac : float
        Diffusion reduction factor for fibrotic regions (D_fib = D0 * Dfac)
    save_dir : str, optional
        Directory to save results. If None, results are not saved to disk.
    
    Returns:
    --------
    dict containing: Vsav, Wsav, t_sav, D_matrix, fiblocs, stim_center, params
    """
    X = ncells + 2
    
    # Generate fibrotic patches
    if patch_type == 'irregular':
        D_matrix, fiblocs = irregular_shape_generation(ncells, D0=D0, Dfac=Dfac, 
                                                       npatches=npatches, seed=seed)
    elif patch_type == 'rectangular':
        D_matrix, fiblocs = make_D_with_rect_patches(ncells, D0=D0, Dfac=Dfac, 
                                                     npatches=npatches, seed=seed)
    else:
        raise ValueError(f"patch_type must be 'irregular' or 'rectangular', got {patch_type}")
    
    # Generate random stimulation location, avoiding fibrotic areas
    fibrotic_mask = D_matrix[1:-1, 1:-1] < D0
    stim_mask, stim_center = generate_random_stim_mask(
        ncells, stim_size=10, seed=seed, forbidden_mask=fibrotic_mask
    )
    
    # Create parameters
    p = Params(
        dt=0.01,
        tend=50*3,
        BCL=50.0,
        ncyc=3,
        extra=0.0,
        stimdur=1.0,
        gathert=10,
        ncells=ncells,
        h=0.1,
        k=8.0,
        mu1=0.2,
        mu2=0.3,
        epsi=0.002,
        D_scalar=D0,
        a0=0.01,
        b0=0.15,
        D_matrix=D_matrix,
        stim_mask=stim_mask,
        stim_amp_scale=0.1,
        cyclic=False,
    )
    
    # Run simulation
    print(f"Running {patch_type} simulation {seed}...")
    out = simulate(p)
    
    # Calculate phie (extracellular potential) using JAX
    print(f"  Calculating phie...")
    elecposX, elecposY = make_electrodes_from_domain(
        X=ncells+2, Y=ncells+2, numelec_x=10, numelec_y=10
    )
    phie, elecpos = calc_phie_jax(
        h=p.h,
        D_matrix=D_matrix,
        elecposX=np.array(elecposX),
        elecposY=np.array(elecposY),
        Vsav=out["Vsav"],
        vsav_layout="TXY",
    )
    
    result = {
        "Vsav": out["Vsav"],
        "Wsav": out["Wsav"],
        "t_sav": out["t_sav"],
        "D_matrix": D_matrix,
        "fiblocs": fiblocs,
        "stim_center": stim_center,
        "phie": phie,
        "elecpos": elecpos,
        "params": p,
        "patch_type": patch_type,
        "seed": seed
    }
    
    # Optionally save to disk
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"sim_{patch_type}_{seed}.npz")
        
        # Convert fiblocs to object array if it's a list (irregular patches)
        # fiblocs is a list of arrays, each (N_patch, 2) with [row, col] coordinates
        fiblocs_save = result["fiblocs"]
        if isinstance(fiblocs_save, list):
            fiblocs_save = np.array(fiblocs_save, dtype=object)
        
        # savez_compressed: save several arrays into a single file in compressed .npz format.
        np.savez_compressed(
            filename,
            # Vsav=result["Vsav"],            # (T, ncells, ncells) - [time, row, col]
            # Wsav=result["Wsav"],            # (T, ncells, ncells) - [time, row, col]
            t_sav=result["t_sav"],          # (T,) - time values
            D_matrix=result["D_matrix"],    # (X, X) - [row, col] where X=ncells+2
            fiblocs=fiblocs_save,           # object array of (N_patch, 2) arrays - each [row, col]
            phie=result["phie"],            # (E, T) - [electrode, time]
            elecpos=result["elecpos"],      # (E, 2) - [electrode, (x, y)]
            stim_center=np.array(result["stim_center"]),  # (2,) - (row, col)
            patch_type=patch_type,
            seed=seed,
            ncells=ncells,
            D0=D0,
            Dfac=Dfac,
            npatches=npatches
        )
        print(f"  Saved to {filename}")

    if not return_data:
        # Return minimal metadata to avoid retaining large arrays in memory
        return {
            "patch_type": patch_type,
            "seed": seed,
            "ncells": ncells,
            "D0": D0,
            "Dfac": Dfac,
            "npatches": npatches,
            "save_dir": save_dir,
        }

    return result


def run_simulations_same_patches(ncells: int,
                                 patch_type: str,
                                 npatches: int,
                                 patch_seed: int,
                                 n_stim: int = 3,
                                 stim_seeds: Optional[list[int]] = None,
                                 D0: float = 0.1,
                                 Dfac: float = 0.2,
                                 save_dir: Optional[str] = None,
                                 return_data: bool = True) -> list:
    """
    Run multiple simulations with the same fibrotic patch configuration and
    different pacing stimulus locations.

    Parameters:
    -----------
    patch_seed : int
        Random seed used to generate the fibrotic patches (fixed across runs)
    n_stim : int
        Number of pacing locations to simulate (ignored if stim_seeds provided)
    stim_seeds : list[int], optional
        Seeds to generate distinct pacing locations. If None, uses
        [patch_seed + 1, patch_seed + 2, ...].

    Returns:
    --------
    list of dicts, each containing simulation results for a different pacing location
    """
    if n_stim < 1:
        raise ValueError(f"n_stim must be >= 1, got {n_stim}")

    # Generate fibrotic patches once
    if patch_type == "irregular":
        D_matrix, fiblocs = irregular_shape_generation(
            ncells, D0=D0, Dfac=Dfac, npatches=npatches, seed=patch_seed
        )
    elif patch_type == "rectangular":
        D_matrix, fiblocs = make_D_with_rect_patches(
            ncells, D0=D0, Dfac=Dfac, npatches=npatches, seed=patch_seed
        )
    else:
        raise ValueError(f"patch_type must be 'irregular' or 'rectangular', got {patch_type}")

    if stim_seeds is None:
        stim_seeds = [patch_seed + i + 1 for i in range(n_stim)]
    else:
        if len(stim_seeds) != n_stim:
            raise ValueError("stim_seeds length must match n_stim")

    results = []
    for i, stim_seed in enumerate(stim_seeds, start=1):
        fibrotic_mask = D_matrix[1:-1, 1:-1] < D0
        stim_mask, stim_center = generate_random_stim_mask(
            ncells, stim_size=10, seed=stim_seed, forbidden_mask=fibrotic_mask
        )

        p = Params(
            dt=0.01,
            tend=50*3,
            BCL=50.0,
            ncyc=3,
            extra=0.0,
            stimdur=1.0,
            gathert=10,
            ncells=ncells,
            h=0.1,
            k=8.0,
            mu1=0.2,
            mu2=0.3,
            epsi=0.002,
            D_scalar=D0,
            a0=0.01,
            b0=0.15,
            D_matrix=D_matrix,
            stim_mask=stim_mask,
            stim_amp_scale=0.1,
            cyclic=False,
        )

        print(f"Running {patch_type} simulation {i}/{n_stim} (stim_seed={stim_seed})...")
        out = simulate(p)

        print("  Calculating phie...")
        elecposX, elecposY = make_electrodes_from_domain(
            X=ncells+2, Y=ncells+2, numelec_x=10, numelec_y=10
        )
        phie, elecpos = calc_phie_jax(
            h=p.h,
            D_matrix=D_matrix,
            elecposX=np.array(elecposX),
            elecposY=np.array(elecposY),
            Vsav=out["Vsav"],
            vsav_layout="TXY",
        )

        result = {
            "Vsav": out["Vsav"],
            "Wsav": out["Wsav"],
            "t_sav": out["t_sav"],
            "D_matrix": D_matrix,
            "fiblocs": fiblocs,
            "stim_center": stim_center,
            "phie": phie,
            "elecpos": elecpos,
            "params": p,
            "patch_type": patch_type,
            "patch_seed": patch_seed,
            "stim_seed": stim_seed,
        }

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(
                save_dir, f"sim_{patch_type}_patch{patch_seed}_stim{stim_seed}.npz"
            )
            fiblocs_save = fiblocs
            if isinstance(fiblocs_save, list):
                fiblocs_save = np.array(fiblocs_save, dtype=object)
            np.savez_compressed(
                filename,
                t_sav=result["t_sav"],
                D_matrix=result["D_matrix"],
                fiblocs=fiblocs_save,
                phie=result["phie"],
                elecpos=result["elecpos"],
                stim_center=np.array(result["stim_center"]),
                patch_type=patch_type,
                patch_seed=patch_seed,
                stim_seed=stim_seed,
                ncells=ncells,
                D0=D0,
                Dfac=Dfac,
                npatches=npatches,
            )
            print(f"  Saved to {filename}")

        if return_data:
            results.append(result)

    if not return_data:
        return [
            {
                "patch_type": patch_type,
                "patch_seed": patch_seed,
                "stim_seed": s,
                "ncells": ncells,
                "D0": D0,
                "Dfac": Dfac,
                "npatches": npatches,
                "save_dir": save_dir,
            }
            for s in stim_seeds
        ]

    return results


# if __name__ == "__main__":
#     # Example: single simulation (original code)
#     ncells = 100
#     X = ncells + 2

#     # D_matrix, patches = make_D_with_rect_patches(ncells, D0=0.1, Dfac=0.1, npatches=5, seed=1)
#     D_matrix, fiblocs = irregular_shape_generation(ncells, D0=0.1, Dfac=0.2, npatches=5, seed=20)

#     # Stim mask: stimulate a small block in the interior
#     stim_mask = np.zeros((ncells, ncells), dtype=np.float32)
#     # Generate a random centre position for the square of size 10x10
#     centre_stim_x = np.random.randint(10, ncells - 10)
#     centre_stim_y = np.random.randint(10, ncells - 10)
#     stim_mask[centre_stim_x-5:centre_stim_x+5, centre_stim_y-5:centre_stim_y+5] = 1.0

#     p = Params(
#         dt=0.01,
#         tend=50*3,
#         BCL=50.0,
#         ncyc=3,
#         extra=0.0,
#         stimdur=1.0,
#         gathert=10,
#         ncells=ncells,
#         h=0.1,
#         k=8.0,
#         mu1=0.2,
#         mu2=0.3,
#         epsi=0.002,
#         D_scalar=0.1,
#         a0=0.01,
#         b0=0.15,
#         D_matrix=D_matrix,
#         stim_mask=stim_mask,
#         stim_amp_scale=0.1,
#         cyclic=False,
#     )

#     out = simulate(p)
#     print(out["Vsav"].shape, out["Wsav"].shape, out["t_sav"].shape)
    # later on will want to save Vsav, Wsav, t_sav, D_matrix or fiblocs

# print(np.nanmin(out["Vsav"]), np.nanmax(out["Vsav"]))
# print(np.isnan(out["Vsav"]).any(), np.isnan(out["Wsav"]).any())
# plt.plot(out["t_sav"], out["Vsav"][:,51, 51])
# plt.xlabel("Time (TU)")
#%% save video

def save_VW_video(Vsav, Wsav, fiblocs, filename="aliev_panfilov.mp4",
                  fps=10, vmin=0.0, vmax=1.0):
    """
    Save V and W fields as a side-by-side MP4 video.

    Vsav, Wsav: (T, nx, ny)
    """
    T = Vsav.shape[0]

    with imageio.get_writer(filename, fps=fps, codec="libx264") as writer:
        for k in range(T):
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            axes[0].imshow(Vsav[k], origin="lower",
                           vmin=vmin, vmax=vmax, cmap="viridis")
            axes[0].imshow(fiblocs, origin="lower", cmap="gray", alpha=0.3)  # overlay fibrotic patches
            axes[0].set_title("V")
            axes[0].axis("off")

            axes[1].imshow(Wsav[k], origin="lower",
                           vmin=vmin, vmax=vmax, cmap="viridis")
            axes[1].imshow(fiblocs, origin="lower", cmap="gray", alpha=0.3)  # overlay fibrotic patches
            axes[1].set_title("W")
            axes[1].axis("off")
            # add timing as shared title
            fig.suptitle(f"Time = {k*0.01*10:.2f} TU or {k*0.01*10*12.9:.2f} ms")  # assuming dt=0.01 and gathert=10, 1 TU is 12.9ms
            # add contours of fibrotic patches

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())
            frame = frame[:, :, :3]  # drop alpha channel
            # ----------------------

            writer.append_data(frame)
            plt.close(fig)

def save_VW_video_fast(Vsav, Wsav, fiblocs, filename="aliev_panfilov.mp4",
                       fps=10, vmin=0.0, vmax=1.0, dt=0.01, gathert=10,
                       add_labels=True, add_time=True, separator_width=4,
                       upscale_factor=4):
    """
    video creation with fibrotic overlay and optional labels.
    Expected time: ~5-10 seconds for 1500 frames (with labels).
    
    Parameters:
    -----------
    separator_width : int
        Width of white separator between V and W panels (pixels)
    add_labels : bool
        Add "V" and "W" titles to each panel
    add_time : bool
        Add timestamp at top
    upscale_factor : int
        Upscale frames by this factor for better text quality (e.g., 4 = 400x400 from 100x100)
    """
    T = Vsav.shape[0]
    viridis = cm.get_cmap('viridis')
    
    # Normalize function
    def to_rgb(x, overlay_mask):
        x_norm = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
        rgb = viridis(x_norm)[:, :, :3]  # (H, W, 3) float in [0,1]
        # Darken fibrotic regions
        rgb[overlay_mask] *= 0.5
        return (255 * rgb).astype(np.uint8)
    
    # Remove ghost border if present
    if fiblocs.shape[0] == Vsav.shape[1] + 2:
        overlay = fiblocs[1:-1, 1:-1].astype(bool)
    else:
        overlay = fiblocs.astype(bool)
    
    # Try to load a font (fallback to default if not available)
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    with imageio.get_writer(filename, fps=fps, codec="libx264") as writer:
        for k in range(T):
            V_rgb = to_rgb(Vsav[k], overlay)
            W_rgb = to_rgb(Wsav[k], overlay)
            
            # Upscale for better text quality
            if upscale_factor > 1:
                V_img = Image.fromarray(V_rgb).resize(
                    (V_rgb.shape[1] * upscale_factor, V_rgb.shape[0] * upscale_factor),
                    Image.LANCZOS
                )
                W_img = Image.fromarray(W_rgb).resize(
                    (W_rgb.shape[1] * upscale_factor, W_rgb.shape[0] * upscale_factor),
                    Image.LANCZOS
                )
                V_rgb = np.array(V_img)
                W_rgb = np.array(W_img)
            
            # Create separator (white vertical bar)
            if separator_width > 0:
                sep = np.ones((V_rgb.shape[0], separator_width * upscale_factor, 3), dtype=np.uint8) * 255
                frame = np.concatenate([V_rgb, sep, W_rgb], axis=1)
            else:
                frame = np.concatenate([V_rgb, W_rgb], axis=1)
            
            # Add labels using PIL (much faster than matplotlib)
            if add_labels or add_time:
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                
                # Add "V" and "W" titles
                if add_labels:
                    v_x = V_rgb.shape[1] // 2 - 10
                    w_x = V_rgb.shape[1] + separator_width * upscale_factor + W_rgb.shape[1] // 2 - 10
                    draw.text((v_x, 5), "V", fill=(255, 255, 255), font=font_large)
                    draw.text((w_x, 5), "W", fill=(255, 255, 255), font=font_large)
                
                # Add timestamp
                if add_time:
                    t_tu = k * dt * gathert
                    t_ms = t_tu * 12.9
                    time_str = f"Time = {t_tu:.2f} TU ({t_ms:.2f} ms)"
                    draw.text((10, frame.shape[0] - 25), time_str, 
                             fill=(255, 255, 255), font=font_small)
                
                frame = np.array(img)
            
            writer.append_data(frame)
    
    print(f"Video saved: {filename}")

#%%
# save_VW_video_fast(
#     out["Vsav"],
#     out["Wsav"],
#     D_matrix < p.D_scalar,
#     filename="test_AP_irregular_Dfac0.2.mp4",
#     fps=10,
#     vmin=0.0,
#     vmax=1.0,
# )
# %% Calculate phie

def make_electrodes_from_domain(X: int, Y: int, numelec_x: int, numelec_y: int):
    centerX = X / 2.0
    centerY = Y / 2.0
    spacing_x = X / (numelec_x + 1)
    spacing_y = Y / (numelec_y + 1)

    gridx = np.linspace(centerX - (spacing_x * (numelec_x - 1)) / 2.0,
                        centerX + (spacing_x * (numelec_x - 1)) / 2.0,
                        numelec_x)
    gridy = np.linspace(centerY - (spacing_y * (numelec_y - 1)) / 2.0,
                        centerY + (spacing_y * (numelec_y - 1)) / 2.0,
                        numelec_y)

    # MATLAB ndgrid: first dimension corresponds to x-grid (rows), second to y-grid (cols)
    elecposX, elecposY = np.meshgrid(gridx, gridy, indexing="ij")
    return elecposX.reshape(-1), elecposY.reshape(-1)


# def calc_phie_numpy(
#     h: float,
#     D_matrix: np.ndarray,
#     elecposX: np.ndarray,
#     elecposY: np.ndarray,
#     Vsav: np.ndarray,
#     *,
#     vsav_layout: str = "TXY",
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Python rewrite of MATLAB calcphie.

#     Parameters
#     ----------
#     h : float
#         grid spacing
#     D_matrix : (X,Y) or (X+2,Y+2)
#         diffusion map. If includes ghost border, pass full and we'll crop to interior.
#     elecposX, elecposY : (E,)
#         electrode positions in "grid index" coordinates consistent with MATLAB usage.
#         MATLAB uses (2:X-1)-elecposX etc. We replicate that.
#     Vsav : array
#         Either (T, X, Y) if vsav_layout="TXY" (your JAX solver),
#         or (X, Y, T) if vsav_layout="XYT" (MATLAB style).

#     Returns
#     -------
#     phie : (E, T) numpy array
#     elecpos : (E, 2) numpy array
#     """
#     # Arrange Vsav to (T, X, Y)
#     if vsav_layout.upper() == "TXY":
#         Vt = Vsav
#     elif vsav_layout.upper() == "XYT":
#         Vt = np.moveaxis(Vsav, 2, 0)
#     else:
#         raise ValueError("vsav_layout must be 'TXY' or 'XYT'")

#     T, X, Y = Vt.shape

#     # Match MATLAB: param.D_matrix = param.D_matrix(2:end-1,2:end-1)
#     if D_matrix.shape == (X + 2, Y + 2):
#         D = D_matrix[1:-1, 1:-1].astype(np.float64, copy=False)
#     elif D_matrix.shape == (X, Y):
#         D = D_matrix.astype(np.float64, copy=False)
#     else:
#         raise ValueError(f"D_matrix shape {D_matrix.shape} incompatible with Vsav shape {(T,X,Y)}")

#     # Compute grad(D) on core using the full interior D (needs one layer around core)
#     # We need D on [0..X-1,0..Y-1] to compute core grads at [1..X-2,1..Y-2]
#     Dx_core, Dy_core = _grad_central_core(D, h)   # (X-2, Y-2)

#     # Electrode positions: add tiny offset if integer (MATLAB mod(...,1)==0)
#     ex = np.array(elecposX, dtype=np.float64, copy=True)
#     ey = np.array(elecposY, dtype=np.float64, copy=True)
#     ex[np.isclose(ex % 1.0, 0.0)] += 1e-4
#     ey[np.isclose(ey % 1.0, 0.0)] += 1e-4

#     E = ex.shape[0]
#     phie = np.zeros((E, T), dtype=np.float64)

#     # MATLAB uses indices 2:X-1 (1-based). Create those index values.
#     # In Python, the corresponding core indices are 1..X-2 (0-based),
#     # but MATLAB's "coordinate values" are 2..X-1. We'll use those values.
#     ix_val = np.arange(2, X, dtype=np.float64)  # length X-2 : 2..X-1
#     iy_val = np.arange(2, Y, dtype=np.float64)  # length Y-2 : 2..Y-1
#     x_phys = ix_val * h
#     y_phys = iy_val * h

#     # Precompute distance grids per electrode: (E, X-2, Y-2)
#     # ndgrid((2:X-1)-ex, (2:Y-1)-ey)
#     matx = ix_val[None, :, None] - ex[:, None, None]
#     maty = iy_val[None, None, :] - ey[:, None, None]
#     distance = np.sqrt(matx * matx + maty * maty) * h  # (E, X-2, Y-2)

#     # Main loop over time (vectorized over electrodes)
#     for t in range(T):
#         V = Vt[t].astype(np.float64, copy=False)

#         gx_core, gy_core = _grad_central_core(V, h)     # (X-2, Y-2)
#         lap_core = _laplacian_core(V, h)                # (X-2, Y-2)

#         # du on core:
#         # du = D_core*lap + Dx_core*gx + Dy_core*gy
#         D_core = D[1:-1, 1:-1]                           # (X-2, Y-2)
#         du_core = D_core * lap_core + Dx_core * gx_core + Dy_core * gy_core

#         integrand = du_core[None, :, :] / distance      # (E, X-2, Y-2)

#         # phie(k,t)= -trapz(y, trapz(x, integrand, 1), 2) in MATLAB
#         # With our shapes: axis=1 corresponds to x-dim (rows), axis=2 is y-dim (cols)
#         inner = np.trapz(integrand, x_phys, axis=1)     # (E, Y-2)
#         phie[:, t] = -np.trapz(inner, y_phys, axis=1)   # (E,)

#     elecpos = np.stack([ex, ey], axis=1)
#     return phie, elecpos
#%%

def _trapz_jax(y: jnp.ndarray, x: jnp.ndarray, axis: int) -> jnp.ndarray:
    dx = x[1:] - x[:-1]
    y_m = jnp.moveaxis(y, axis, -1)
    return jnp.sum(0.5 * (y_m[..., 1:] + y_m[..., :-1]) * dx, axis=-1)

def calc_phie_jax(
    Vsav: np.ndarray,              # (T, ncells, ncells) interior only
    h: float,
    D_matrix: np.ndarray,          # (X, Y) with ghost, X=ncells+2
    elecposX: np.ndarray,          # (E,) electrode positions
    elecposY: np.ndarray,          # (E,)
    vsav_layout: str = "TXY",
    dtype=jnp.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    
    if vsav_layout.upper() == "TXY":
        Vt = Vsav  # (T, X, Y)
    elif vsav_layout.upper() == "XYT":
        Vt = np.moveaxis(Vsav, 2, 0)  # Convert (X, Y, T) -> (T, X, Y)
    else:
        raise ValueError("vsav_layout must be 'TXY' or 'XYT'")
    
    ncells = Vt.shape[1]
    
    Dx, Dy = _grad_central(D_matrix, h)   # (102,102)
    Dx_core = Dx[1:-1, 1:-1]              # (100,100)
    Dy_core = Dy[1:-1, 1:-1]              # (100,100)

    # Electrode positions with small nudge to avoid exact integers
    ex = np.array(elecposX, dtype=np.float32, copy=True)
    ey = np.array(elecposY, dtype=np.float32, copy=True)
    ex[np.isclose(ex % 1.0, 0.0)] += 1e-4
    ey[np.isclose(ey % 1.0, 0.0)] += 1e-4
    elecpos = np.stack([ex, ey], axis=1)
    
    x_coords = np.arange(0, ncells+2, dtype=np.float32) #(ncells+2,) e.g., (102,)
    y_coords = np.arange(0, ncells+2, dtype=np.float32) #(ncells+2,) e.g., (102,)
    x_phys = jnp.asarray(x_coords * h, dtype=dtype)
    y_phys = jnp.asarray(y_coords * h, dtype=dtype)
    
    # Convert to JAX arrays
    Vt_jax = jnp.asarray(Vt, dtype=dtype)
    Dx_jax = jnp.asarray(Dx_core, dtype=dtype)
    Dy_jax = jnp.asarray(Dy_core, dtype=dtype)
    
    ex_jax = jnp.asarray(ex, dtype=dtype)
    ey_jax = jnp.asarray(ey, dtype=dtype)
    x_coords_jax = jnp.asarray(x_coords, dtype=dtype)
    y_coords_jax = jnp.asarray(y_coords, dtype=dtype)
    
    @jax.jit
    def compute_phie_all_timesteps(Vt_arr, D_matrix, Dx_core, Dy_core, ex, ey, x_coords, y_coords, x_phys, y_phys):
        """Compute phie for all timesteps - JIT compiled."""
        
        def compute_phie_one_timestep(V: jnp.ndarray) -> jnp.ndarray:
            """Compute phie for one timestep and all electrodes."""
            # V is (ncells, ncells) core only
            # Use jnp.gradient which handles boundaries with forward/backward diff
            gx_core = jnp.gradient(V, h, axis=0)  # (ncells, ncells)
            gy_core = jnp.gradient(V, h, axis=1)  # (ncells, ncells)
            
            # Compute Laplacian using jnp.gradient
            # ∇²V = ∂²V/∂x² + ∂²V/∂y²
            gxx = jnp.gradient(gx_core, h, axis=0)  # ∂²V/∂x²
            gyy = jnp.gradient(gy_core, h, axis=1)  # ∂²V/∂y²
            lap_core = gxx + gyy
            
            D_core = D_matrix[1:-1, 1:-1]            
            du_core = D_core * lap_core + Dx_core * gx_core + Dy_core * gy_core
            
            def compute_phie_one_electrode(elec_x, elec_y):
                """Compute phie for one electrode."""
                # Create 2D meshgrid for distance calculation
                # Core indices are 1 to ncells (in 0-indexed: [1:-1])
                x_core = x_coords[1:-1]  # (ncells,)
                y_core = y_coords[1:-1]  # (ncells,)
                # Create 2D grid: X[i,j] has x-coord, Y[i,j] has y-coord
                X, Y = jnp.meshgrid(x_core, y_core, indexing='ij')
                distance = jnp.sqrt((X - elec_x)**2 + (Y - elec_y)**2) * h   # (ncells, ncells)
                
                # Integration: matches MATLAB trapz(y, trapz(x, du_core ./ distance, 1), 2)
                integrand = du_core / distance
                inner = _trapz_jax(integrand, x_phys[1:-1], axis=0)  # Integrate over x first
                result = -_trapz_jax(inner, y_phys[1:-1], axis=0)    # Then integrate over y
                return result
            
            # Vectorize over all electrodes
            phie_t = jax.vmap(compute_phie_one_electrode)(ex, ey)
            return phie_t
        
        # Vectorize over all timesteps
        phie = jax.vmap(compute_phie_one_timestep)(Vt_arr)  # (T, E)
        return phie
    
    # Compute phie: (T, E)
    phie_t = compute_phie_all_timesteps(
        Vt_jax, D_matrix, Dx_jax, Dy_jax, ex_jax, ey_jax, 
        x_coords_jax, y_coords_jax, x_phys, y_phys
    )
    
    # Transpose to (E, T) to match expected output
    phie = jnp.transpose(phie_t, (1, 0))

    return np.array(phie), elecpos

# %%
# =============================================================================
# BATCH SIMULATION GENERATION
# =============================================================================
# Generate multiple simulations with different random pacing locations and 
# fibrotic patches. Half will be irregular patches, half will be rectangular.

def generate_batch_simulations(n_simulations: int = 10,
                               ncells: int = 100,
                               npatches: int = 5,
                               D0: float = 0.1,
                               Dfac: float = 0.2,
                               save_dir: str = "simulation_batch",
                               start_seed: int = 0,
                               keep_results: bool = True,
                               same_patches: bool = False,
                               n_stim: int = 3) -> list:
    """
    Generate multiple simulations with varying fibrotic patterns and pacing locations.
    
    Parameters:
    -----------
    n_simulations : int
        Total number of simulations to generate (all with irregular patches)
    ncells : int
        Grid size (interior)
    npatches : int
        Number of fibrotic patches per simulation
    D0 : float
        Baseline diffusion coefficient
    Dfac : float
        Diffusion reduction factor for fibrotic regions
    save_dir : str
        Directory to save simulation results
    start_seed : int
        Starting seed for random number generation
        
    Returns:
    --------
    list of dicts, each containing simulation results
    """
    results = []
    
    print(f"\n{'='*70}")
    print(f"GENERATING {n_simulations} SIMULATIONS")
    print(f"  - All with irregular patches")
    print(f"  - Grid size: {ncells}x{ncells}")
    print(f"  - Patches per simulation: {npatches}")
    print(f"  - D0={D0}, Dfac={Dfac}")
    print(f"  - Results will be saved to: {save_dir}/")
    if same_patches:
        print(f"  - Same patches per seed with {n_stim} pacing locations")
    print(f"{'='*70}\n")
    
    # Generate irregular patch simulations
    for i in range(n_simulations):
        seed = start_seed + i
        if same_patches:
            print(f"\n[{i+1}/{n_simulations}] Irregular patch set (seed={seed})")
            batch_results = run_simulations_same_patches(
                ncells=ncells,
                patch_type="irregular",
                npatches=npatches,
                patch_seed=seed,
                n_stim=n_stim,
                D0=D0,
                Dfac=Dfac,
                save_dir=save_dir,
                return_data=keep_results,
            )
            if keep_results:
                results.extend(batch_results)
            del batch_results
        else:
            print(f"\n[{i+1}/{n_simulations}] Irregular patch simulation (seed={seed})")
            result = run_single_simulation(
                ncells=ncells,
                patch_type='irregular',
                npatches=npatches,
                seed=seed,
                D0=D0,
                Dfac=Dfac,
                save_dir=save_dir,
                return_data=keep_results
            )
            if keep_results:
                results.append(result)
            del result

        # Proactively release memory between simulations
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass
    
    print(f"\n{'='*70}")
    print(f"BATCH GENERATION COMPLETE")
    if keep_results:
        print(f"Total simulations: {len(results)}")
    else:
        total = n_simulations * n_stim if same_patches else n_simulations
        print(f"Total simulations: {total}")
    print(f"Results saved to: {save_dir}/")
    print(f"{'='*70}\n")
    
    return results


# Example usage: Uncomment to run batch simulation generation
# =============================================================================
results = generate_batch_simulations(
    n_simulations=1000,
    ncells=100,
    npatches=3,
    D0=0.1,
    Dfac=0.2,
    save_dir="AP_simulations_npatches3_nstim3",
    start_seed=15127,
    keep_results=False,
    same_patches=True,
    n_stim=3
)

# Optional: Generate videos for each simulation
# for i, result in enumerate(results):
#     video_filename = f"AP_simulations_batch/video_{result['patch_type']}_{result['seed']}.mp4"
#     save_VW_video_fast(
#         result["Vsav"],
#         result["Wsav"],
#         result["D_matrix"] < result["params"].D_scalar,
#         filename=video_filename,
#         fps=10,
#         vmin=0.0,
#         vmax=1.0,
#         dt=result["params"].dt,
#         gathert=result["params"].gathert,
#     )
#     print(f"Video {i+1}/{len(results)} saved: {video_filename}")

# # %%
# data = np.load("AP_simulations_batch_400elec/sim_irregular_1009_400elec.npz", allow_pickle=True)
# Vsav = data["Vsav"]
# Wsav = data["Wsav"] 
# phie = data["phie"]
# D_matrix = data["D_matrix"]
# plt.figure()
# plt.plot(phie[5,:])
# # plt.figure()
# # plt.imshow(D_matrix < 0.1, origin="lower", cmap="gray")
# # %%
# fiblocs = data["fiblocs"]
# print(fiblocs.shape)
# # %%
# # calculate extra phie values from saved V values and save them as separate files
# # read all files in AP_simulations_batch
# import os
# data_dir = "AP_simulations_batch"
# output_dir = "AP_simulations_batch_400elec"
# os.makedirs(output_dir, exist_ok=True)
# # read files ending with .npz
# for filename in os.listdir(data_dir):
#     if filename.endswith(".npz"):
#         with np.load(os.path.join(data_dir, filename), allow_pickle=True) as data:
#             Vsav = data["Vsav"]
#             D_matrix = data["D_matrix"]
#             ncells = data["ncells"].item() if "ncells" in data else Vsav.shape[1]

#             # Create electrode positions for 20x20 grid
#             elecposX, elecposY = make_electrodes_from_domain(
#                 X=ncells+2, Y=ncells+2, numelec_x=20, numelec_y=20
#             )

#             # Calculate phie
#             phie, elecpos = calc_phie_jax(
#                 h=0.1,
#                 D_matrix=D_matrix,
#                 elecposX=np.array(elecposX),
#                 elecposY=np.array(elecposY),
#                 Vsav=Vsav,
#                 vsav_layout="TXY",
#             )

#             T = Vsav.shape[0]
#             chunk = 50  # try 10–100
#             phie_chunks = []
#             for i in range(0, T, chunk):
#                 phie_i, elecpos = calc_phie_jax(
#                     h=0.1,
#                     D_matrix=D_matrix,
#                     elecposX=np.array(elecposX),
#                     elecposY=np.array(elecposY),
#                     Vsav=Vsav[i:i+chunk],
#                     vsav_layout="TXY",
#                 )
#                 phie_chunks.append(np.array(phie_i))
#             phie = np.concatenate(phie_chunks, axis=0)

#             # Save new file with phie and elecpos
#             output_filepath = os.path.join(output_dir, f"{filename[:-4]}_400elec.npz")
#             np.savez_compressed(
#                 output_filepath,
#                 Vsav=Vsav,
#                 Wsav=data["Wsav"],
#                 t_sav=data["t_sav"],
#                 D_matrix=D_matrix,
#                 fiblocs=data["fiblocs"],
#                 phie=phie,
#                 elecpos=elecpos,
#                 stim_center=data["stim_center"],
#                 patch_type=data["patch_type"],
#                 seed=data["seed"],
#                 ncells=ncells,
#             )
#             print(f"Saved: {output_filepath}")
# %%    
