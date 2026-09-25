"""
solveAP_surface_fenicsx.py

Aliev-Panfilov on a curved 2D surface using FEniCSx (dolfinx).

Geometry: ellipsoid patch parametrised by (theta, phi).
    x = rx * sin(theta) * cos(phi)
    y = ry * sin(theta) * sin(phi)
    z = rz * cos(theta)
Equal semi-axes (rx == ry == rz == R) recover a spherical cap; used for
the sanity check (LB eigenvalue of f=z is 2/R^2).

Time integration: IMEX -- implicit diffusion (one CG solve / step),
explicit reaction. Boundary: natural Neumann (no-flux).

Install:
    fenics-dolfinx installed via Docker

Open docker container:
    cmd+shift+P -> Dev Containers: Repopen in container
    (to reopen locally and use Claude Code, cmd+shift+P -> Dev Containers: Reopen locally)
Run (in terminal inside container):
    python solveAP_surface_fenicsx_phie.py
Visualise: open {saved_results}.xdmf in ParaView.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import ufl
import basix.ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.io import XDMFFile


# ----------------------------------------------------------------- geometry

def make_ellipsoid_patch(rx, ry, rz,
                         theta_range=(0.1, np.pi / 2),
                         phi_range=(0.0, np.pi),
                         n_theta=40, n_phi=40):
    """Structured triangulation of a (theta, phi) rectangle lifted onto the
    ellipsoid. Equal semi-axes give a spherical cap.

    theta_range starts >0 to avoid the polar coordinate singularity.
    Mesh is non-periodic in phi (a patch, not a closed band).
    """
    theta = np.linspace(*theta_range, n_theta + 1)
    phi = np.linspace(*phi_range, n_phi + 1)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    coords = np.column_stack([
        (rx * np.sin(TH) * np.cos(PH)).ravel(),
        (ry * np.sin(TH) * np.sin(PH)).ravel(),
        (rz * np.cos(TH)).ravel(),
    ])
    ij = np.arange((n_theta + 1) * (n_phi + 1)).reshape(n_theta + 1, n_phi + 1)
    v00, v10 = ij[:-1, :-1].ravel(), ij[1:, :-1].ravel()
    v01, v11 = ij[:-1, 1:].ravel(),  ij[1:, 1:].ravel()
    cells = np.vstack([
        np.column_stack([v00, v10, v11]),
        np.column_stack([v00, v11, v01]),
    ]).astype(np.int64)

    coords = np.asarray(coords, dtype=np.float64, order="C")
    cells = np.asarray(cells, dtype=np.int64, order="C")

    elem = basix.ufl.element("Lagrange", "triangle", 1, shape=(3,))
    return dmesh.create_mesh(MPI.COMM_WORLD, cells, ufl.Mesh(elem), coords)


def ellipsoid_point(rx, ry, rz, theta, phi):
    """3D point on the ellipsoid at parameter (theta, phi)."""
    return np.array([rx * np.sin(theta) * np.cos(phi),
                     ry * np.sin(theta) * np.sin(phi),
                     rz * np.cos(theta)], dtype=np.float64)


def ellipsoid_inverse_params(coords, rx, ry, rz):
    """Inverse of the ellipsoid map: recover (theta, phi) for surface points.
    With x = rx sin(t) cos(p), y = ry sin(t) sin(p), z = rz cos(t):

        theta = arccos(z / rz),   phi = atan2(y / ry, x / rx)

    coords : (N, 3) surface points.  returns: (N, 2) array of (theta, phi).
    Used by surface_to_graph.py to give each mesh node a 2D surface coordinate.
    """
    coords = np.asarray(coords, dtype=np.float64)
    theta = np.arccos(np.clip(coords[:, 2] / rz, -1.0, 1.0))
    phi = np.arctan2(coords[:, 1] / ry, coords[:, 0] / rx)
    return np.column_stack([theta, phi])


def make_D_field(msh, D0, center_xyz, radius, fib_factor=0.1):
    """Piecewise-constant (DG0) conductivity field for diffuse fibrosis:
    D = D0 everywhere, reduced to D0 * fib_factor inside a ball of `radius`
    about `center_xyz` (a circular fibrotic patch on the surface).

    Returned Function is used both for the diffusion form and (optionally) as
    the D-weighted source in compute_phie. DG0 keeps a crisp patch boundary.
    """
    Dh = fem.functionspace(msh, ("DG", 0))
    D_fld = fem.Function(Dh, name="D")
    c = np.asarray(center_xyz, dtype=np.float64).reshape(3, 1)

    def expr(x):
        d2 = np.sum((x - c) ** 2, axis=0)
        vals = np.full(x.shape[1], D0, dtype=np.float64)
        vals[d2 < radius**2] = D0 * fib_factor
        return vals

    D_fld.interpolate(expr)
    return D_fld


def make_nodal_patch_field(msh, base, factor, center_xyz, radius, name="f"):
    """P1 (nodal) field for an in-patch parameter modification:
    value = `base` everywhere, scaled to `base * factor` inside a ball of
    `radius` about `center_xyz`.

    P1 (not DG0) so the returned `.x.array` aligns dof-for-dof with the
    transmembrane V used in the explicit reaction update. Mirrors the per-patch
    a/b modification in the flat generator (run_single_simulation in
    solveAP_2D_jax.py, which raises a and lowers b inside the patch, not just D).
    """
    Fh = fem.functionspace(msh, ("Lagrange", 1))
    fld = fem.Function(Fh, name=name)
    c = np.asarray(center_xyz, dtype=np.float64).reshape(3, 1)

    def expr(x):
        d2 = np.sum((x - c) ** 2, axis=0)
        vals = np.full(x.shape[1], base, dtype=np.float64)
        vals[d2 < radius**2] = base * factor
        return vals

    fld.interpolate(expr)
    return fld


# -------------------------------------------------------- Aliev-Panfilov

@dataclass
class APParams:
    D: float = 0.1
    k: float = 8.0
    a: float = 0.01      # V-cubic excitation threshold (matches flat solver a0)
    b: float = 0.15      # W-recovery threshold        (matches flat solver b0)
    eps0: float = 0.002
    mu1: float = 0.2
    mu2: float = 0.3
    dt: float = 0.05
    tend: float = 15.0
    save_every: int = 20
    stim_amp: float = 1.0
    stim_duration: float = 1.0


@dataclass
class Stimulus:
    """A single stimulus event: where it is, when it fires, for how long.

    mask_fn  : callable(x) -> mask values at DOFs; x has shape (3, N).
               Defines the spatial footprint (e.g. a ball on the surface).
    t_start  : time the stimulus turns on.
    duration : how long it stays on; falls back to APParams.stim_duration.
    amp      : injected current amplitude; falls back to APParams.stim_amp.
    """
    mask_fn: callable
    t_start: float = 0.0
    duration: float | None = None
    amp: float | None = None


def make_cyclic_stimuli(mask_fns, cycle_length, n_cycles=None,
                        t0=0.0, duration=None, amp=None):
    """Build a list of Stimulus events, one per cycle, fired at the start of
    each cycle (t0, t0 + cycle_length, t0 + 2*cycle_length, ...).

    mask_fns    : either a single callable(x) reused every cycle, or a list of
                  callables giving a (possibly different) location per cycle.
    cycle_length: basic cycle length (time between successive stimuli).
    n_cycles    : number of cycles; if None, inferred from len(mask_fns).
    duration/amp: per-event overrides (None -> APParams defaults in solver).
    """
    if callable(mask_fns):
        if n_cycles is None:
            raise ValueError("n_cycles is required when mask_fns is a single callable")
        mask_fns = [mask_fns] * n_cycles
    else:
        mask_fns = list(mask_fns)
        n_cycles = len(mask_fns) if n_cycles is None else n_cycles
        if len(mask_fns) < n_cycles:  # reuse last location for trailing cycles
            mask_fns = mask_fns + [mask_fns[-1]] * (n_cycles - len(mask_fns))

    return [Stimulus(mask_fn=mask_fns[i], t_start=t0 + i * cycle_length,
                     duration=duration, amp=amp)
            for i in range(n_cycles)]


def solve_aliev_panfilov(msh, p, stim_mask_fn=None, stimuli=None,
                         D_field=None, a_field=None, b_field=None, k_field=None,
                         xdmf_path=None):
    """
    msh          : dolfinx Mesh (tdim=2, gdim=3)
    p            : APParams
    stim_mask_fn : callable(x) -> mask values at DOFs; x has shape (3, N).
                   Convenience for a single stimulus at t in [0, stim_duration).
    stimuli      : list[Stimulus] for multi-cycle / multi-site stimulation.
                   Each fires over [t_start, t_start + duration). Takes
                   precedence over stim_mask_fn; the two can be combined.
    D_field      : optional DG0/P1 Function for heterogeneous diffusion;
                   falls back to the scalar p.D if None
    a_field      : optional P1 Function for a heterogeneous excitation threshold
                   a (e.g. raised inside the fibrotic patch); falls back to the
                   scalar p.a if None. Its .x.array must align with V's DOFs.
    b_field      : optional P1 Function for a heterogeneous recovery threshold b
                   (e.g. lowered inside the patch); falls back to scalar p.b.
    k_field      : optional P1 Function for a heterogeneous excitability gain k
                   (e.g. raised inside a heightened-excitability patch); falls
                   back to scalar p.k if None. Its .x.array must align with V.
    xdmf_path    : if given, write V snapshots there

    Returns (V_n, W_n, snap_t, snap_V):
        snap_t : (T,) array of snapshot times
        snap_V : (T, N) array of transmembrane snapshots (for phie post-proc)
    """
    Vh = fem.functionspace(msh, ("Lagrange", 1))
    V_n = fem.Function(Vh, name="V")
    W_n = fem.Function(Vh, name="W")
    V_star = fem.Function(Vh)

    # Assemble the list of stimulus events. The legacy single-mask argument
    # becomes one event covering [0, stim_duration).
    events = list(stimuli) if stimuli is not None else []
    if stim_mask_fn is not None:
        events.append(Stimulus(mask_fn=stim_mask_fn, t_start=0.0))

    # Precompute each event's spatial mask once (interpolation is the costly
    # bit); in the time loop we only scale by amplitude and switch on/off.
    stim_events = []
    for ev in events:
        mask = fem.Function(Vh)
        mask.interpolate(ev.mask_fn)
        duration = ev.duration if ev.duration is not None else p.stim_duration
        amp = ev.amp if ev.amp is not None else p.stim_amp
        stim_events.append((ev.t_start, ev.t_start + duration, amp, mask.x.array))

    if msh.comm.rank == 0:
        print(f"{len(stim_events)} stimulus event(s):")
        for i, (t0, t1, amp, m) in enumerate(stim_events):
            print(f"  [{i}] t=[{t0:.2f}, {t1:.2f}), amp={amp:g}, "
                  f"nonzero dofs={np.count_nonzero(m)}")

    u, v = ufl.TrialFunction(Vh), ufl.TestFunction(Vh)
    dt = fem.Constant(msh, PETSc.ScalarType(p.dt))
    D = D_field if D_field is not None else fem.Constant(msh, PETSc.ScalarType(p.D))

    a_form = fem.form((u * v + dt * D * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx)
    L_form = fem.form(V_star * v * ufl.dx)
    A = assemble_matrix(a_form); A.assemble()
    b = create_vector(L_form.function_spaces)

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("cg"); ksp.getPC().setType("jacobi")
    ksp.setTolerances(rtol=1e-8)

    xdmf = None
    if xdmf_path is not None:
        xdmf = XDMFFile(msh.comm, xdmf_path, "w")
        xdmf.write_mesh(msh)
        xdmf.write_function(V_n, 0.0)

    # Per-node reaction thresholds: scalar p.a/p.b, or heterogeneous P1 fields
    # (e.g. modified inside the fibrotic patch). a_arr/b_arr broadcast against
    # the nodal V array in the explicit reaction update below.
    a_arr = a_field.x.array if a_field is not None else p.a
    b_arr = b_field.x.array if b_field is not None else p.b
    k_arr = k_field.x.array if k_field is not None else p.k

    snap_t = [0.0]
    snap_V = [V_n.x.array.copy()]

    nsteps = int(p.tend / p.dt)
    for n in range(1, nsteps + 1):
        t = n * p.dt
        Va, Wa = V_n.x.array, W_n.x.array
        rV = -k_arr * Va * (Va - a_arr) * (Va - 1.0) - Va * Wa
        eps = p.eps0 + p.mu1 * Wa / (Va + p.mu2)
        rW = eps * (-Wa - k_arr * Va * (Va - b_arr - 1.0))
        Iext = 0.0
        for t0, t1, amp, mask in stim_events:
            if t0 <= t < t1:
                Iext = Iext + amp * mask
        V_star.x.array[:] = Va + p.dt * (rV + Iext)
        W_n.x.array[:] = Wa + p.dt * rW
        W_n.x.scatter_forward()

        with b.localForm() as loc:
            loc.set(0.0)
        assemble_vector(b, L_form)
        ksp.solve(b, V_n.x.petsc_vec)
        V_n.x.scatter_forward()

        stim_active = any(t0 <= t < t1 for t0, t1, _, _ in stim_events)
        # if msh.comm.rank == 0 and (n % 100 == 0 or stim_active):
        #     print(
        #         f"t={t:.2f}, "
        #         f"V min/max=[{V_n.x.array.min():.4f}, {V_n.x.array.max():.4f}], "
        #         f"W min/max=[{W_n.x.array.min():.4f}, {W_n.x.array.max():.4f}]"
        #     )

        if n % p.save_every == 0:
            snap_t.append(t)
            snap_V.append(V_n.x.array.copy())
            if xdmf is not None:
                xdmf.write_function(V_n, t)

    if xdmf is not None:
        xdmf.close()
    return V_n, W_n, np.asarray(snap_t), np.asarray(snap_V)


# ----------------------------------------- extracellular potential (phie)

def make_electrodes_on_shell(rx, ry, rz, offset,
                             theta_range, phi_range, n_theta_e, n_phi_e):
    """Electrode coordinates on an ellipsoidal shell offset OUTWARD from the
    tissue surface by `offset`, along the ellipsoid normal. Mimics a
    non-contact array sitting off the surface (keeps |r - x_i| > 0).

    Returns (positions, theta_phi):
        positions : (E, 3) electrode positions in R^3.
        theta_phi : (E, 2) the (theta, phi) parameters each electrode sits at
                    (the surface parametrisation surface_to_graph.py uses to
                    interpolate electrode features onto mesh nodes).
    """
    theta = np.linspace(*theta_range, n_theta_e)
    phi = np.linspace(*phi_range, n_phi_e)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    x = rx * np.sin(TH) * np.cos(PH)
    y = ry * np.sin(TH) * np.sin(PH)
    z = rz * np.cos(TH)
    nx, ny, nz = x / rx**2, y / ry**2, z / rz**2          # outward normal dir
    nrm = np.sqrt(nx**2 + ny**2 + nz**2)
    positions = np.column_stack([
        (x + offset * nx / nrm).ravel(),
        (y + offset * ny / nrm).ravel(),
        (z + offset * nz / nrm).ravel(),
    ])
    theta_phi = np.column_stack([TH.ravel(), PH.ravel()])
    return positions, theta_phi


def compute_phie(msh, snap_V, electrodes, gain=1.0, D_field=None):
    """Extracellular potential at off-surface electrodes from transmembrane
    snapshots, via the monopole (stiffness) forward model for a thin surface
    in an infinite homogeneous volume conductor:

        phie(r, t) = gain * sum_i (K V(t))_i / |r - x_i|

    where K is the (Laplace-Beltrami) stiffness matrix and x_i are DOF coords.
    Equivalently a lead-field matmul: phie = (gain * G @ K) @ V, with
    G[e,i] = 1/|r_e - x_i|.

    If `D_field` is given, K is the D-weighted stiffness int D grad u . grad v,
    so the source is div(D grad V) -- consistent with the propagation operator
    and letting fibrosis (low D) directly reduce phie amplitude. With D_field
    None, K is the plain Laplace-Beltrami stiffness.

    `gain` lumps sigma_i * thickness / (4 pi sigma_e); it scales magnitude
    only, not the spatio-temporal pattern. Do NOT per-electrode normalise --
    amplitude is the discriminative feature.

    snap_V     : (T, N) transmembrane snapshots (N = number of DOFs)
    electrodes : (E, 3)
    returns    : phie (T, E)

    Note: serial (single-rank) assembly of K via getValuesCSR.
    """
    import scipy.sparse as sp
    from scipy.spatial.distance import cdist

    Vh = fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(Vh), ufl.TestFunction(Vh)
    integ = ufl.inner(ufl.grad(u), ufl.grad(v))
    if D_field is not None:
        integ = D_field * integ
    Kp = assemble_matrix(fem.form(integ * ufl.dx))
    Kp.assemble()
    indptr, indices, data = Kp.getValuesCSR()
    xnodes = Vh.tabulate_dof_coordinates()                # (N, 3)
    N = xnodes.shape[0]
    K = sp.csr_matrix((data, indices, indptr), shape=(N, N))

    G = 1.0 / cdist(electrodes, xnodes)                   # (E, N)
    snap_V = np.atleast_2d(np.asarray(snap_V))            # (T, N)
    Q = snap_V @ K.T                                      # (T, N) nodal sources
    return gain * (Q @ G.T)                               # (T, E)


def plot_phie_traces(snap_t, phie, n=5, seed=0, path="phie_traces.png",
                     electrodes=None, lowD_center=None, lowD_radius=None):
    """Plot phie(t) at n electrodes and save to `path`. Returns the chosen
    electrode indices.

    Electrodes are picked at random (reproducible via `seed`). When
    `electrodes`, `lowD_center` and `lowD_radius` are supplied, at least one
    electrode sitting over the low-D (fibrotic) region is guaranteed to be in
    the set, so the fibrosis signature is always visible; that trace is drawn
    thicker and flagged "(low-D)" in the legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    n = min(n, phie.shape[1])
    idx = rng.choice(phie.shape[1], size=n, replace=False)

    # Identify electrodes over the low-D patch (offset shell sits ~on the
    # surface, so electrode-to-center distance < radius matches the DG0 patch).
    lowD_idx = set()
    if electrodes is not None and lowD_center is not None and lowD_radius is not None:
        c = np.asarray(lowD_center, dtype=np.float64).reshape(3)
        in_lowD = np.where(
            np.linalg.norm(np.asarray(electrodes) - c, axis=1) < lowD_radius)[0]
        if in_lowD.size and not np.intersect1d(idx, in_lowD).size:
            idx[0] = rng.choice(in_lowD)   # force one low-D electrode into the set
        lowD_idx = set(idx.tolist()) & set(in_lowD.tolist())

    plt.figure(figsize=(8, 5))
    for i in idx:
        if i in lowD_idx:
            plt.plot(snap_t, phie[:, i], lw=2.2, label=f"elec {i} (low-D)")
        else:
            plt.plot(snap_t, phie[:, i], label=f"elec {i}")
    plt.xlabel("t (AU)"); plt.ylabel("phie (AU)")
    plt.title(f"Extracellular potential at {n} electrodes")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    return idx


def _set_equal_aspect_3d(ax, *point_arrays):
    pts = np.vstack(point_arrays)
    center = (pts.min(0) + pts.max(0)) / 2
    radius = (pts.max(0) - pts.min(0)).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_surface_with_electrodes(msh, electrodes, highlight_idx=None,
                                 path="surface_electrodes.png"):
    """3D view of the surface mesh with the electrode array overlaid.
    Electrodes in `highlight_idx` are emphasised and labelled (e.g. the ones
    whose phie traces were plotted)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = msh.geometry.x
    cells = np.asarray(msh.geometry.dofmap).reshape(-1, 3)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(x[:, 0], x[:, 1], x[:, 2], triangles=cells,
                    color="lightsteelblue", alpha=0.35, linewidth=0)
    ax.scatter(*electrodes.T, c="0.6", s=12, depthshade=False, label="electrodes")
    if highlight_idx is not None:
        hl = electrodes[np.asarray(highlight_idx)]
        ax.scatter(*hl.T, c="red", s=45, depthshade=False, label="plotted")
        for i, p3 in zip(highlight_idx, hl):
            ax.text(p3[0], p3[1], p3[2], f" {i}", color="red", fontsize=8)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Surface mesh + electrode array")
    ax.legend(loc="upper left")
    _set_equal_aspect_3d(ax, x, electrodes)
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


# ---------------------------------------------------------------------- main

if __name__ == "__main__":
    import os

    # Option C "reduce bend": flatten the patch while keeping its physical area
    # (and the rest of the simulation) unchanged. Scale the ellipsoid radii by F
    # and shrink each angular half-width by 1/F about its midpoint: arc length
    # R*dangle -- hence area, feature sizes and propagation distances -- is
    # preserved, while the surface normal turns F x less across the patch. F = 1.0
    # is the baseline curvature; larger F is flatter.
    #
    # This script generates the few-shot FINE-TUNING set: N sims that differ in
    # BOTH the fibrotic patch location (FT_FIB_SITES) and the curvature
    # (FT_FLATTEN_FACTORS), so the model sees fibrosis in different places AND at a
    # range of curvatures. Because F now varies per sim, the whole geometry (mesh,
    # electrodes, stimuli) is rebuilt inside the loop. Outputs go in their own
    # folder so the held-out eval graph (reduce_bend_s2/) stays separate.
    # Abnormality type. "fibrosis" runs the reduce_bend_ft fibrosis loop below;
    # the "excitability_*" modes run a single heightened-excitability sim (D
    # unchanged, in-patch a -> NEG_A and/or k -> k*K_FACTOR) and exit early.
    # fibrosis | excitability_neg_a | excitability_higher_k | excitability_both
    MODE = os.environ.get("MODE", "excitability_neg_a")
    # EX_FT=1 -> generate the 10-sim excitability fine-tuning SET (varied site +
    # curvature, sampled k_factor) into sim_results/<MODE>_ft/ instead of one sim.
    EX_FT = os.environ.get("EX_FT", "0") == "1"
    # EX_EVAL=1 / FIB_EVAL=1 -> generate a 5-site HELD-OUT EVAL set at curvature
    # EX_F / FIB_F (for mean +/- std test metrics), into <MODE>_eval /
    # reduce_bend_eval. Sites are interior and distinct from the fine-tune sites.
    EX_EVAL = os.environ.get("EX_EVAL", "0") == "1"
    EVAL_SITES = [(0.90, 1.30), (0.70, 1.30), (1.00, 1.90),
                  (0.95, 1.15), (0.80, 1.70)]

    RUN_NAME = ("reduce_bend_ft" if MODE == "fibrosis"
                else f"{MODE}_eval" if EX_EVAL
                else f"{MODE}_ft" if EX_FT else MODE)
    OUT_DIR = os.path.join("sim_results", RUN_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)

    def out_path(stem, ext):
        """sim_results/<RUN_NAME>/<stem>_<RUN_NAME>.<ext>"""
        return os.path.join(OUT_DIR, f"{stem}_{RUN_NAME}.{ext}")

    # Sanity checks (LB operator + phie forward model) and the mesh-coarseness
    # convergence study now live in tests_solveAP.py:  python tests_solveAP.py

    # Baseline geometry (the F=1 frame; per-sim F rescales it).
    RX0, RY0, RZ0 = 10.0, 10.0, 15.0
    THETA_RANGE0 = (0.1, np.pi / 2)
    PHI_RANGE0 = (0.0, np.pi)

    def _reduce_bend_range(rng, s):
        """Narrow an angular range to 1/s of its width about the same midpoint."""
        lo, hi = rng
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) / s
        return (mid - half, mid + half)

    def reduce_bend_site(theta, phi, s):
        """Map a baseline-frame (theta, phi) into the F=s narrowed window, keeping
        its relative position in the patch -- so its physical location, the
        feature radii and the propagation distances are all preserved."""
        tmid = 0.5 * (THETA_RANGE0[0] + THETA_RANGE0[1])
        pmid = 0.5 * (PHI_RANGE0[0] + PHI_RANGE0[1])
        return (tmid + (theta - tmid) / s, pmid + (phi - pmid) / s)

    def make_stim_mask_on_ellipsoid(rx, ry, rz, theta0, phi0, radius):
        x0 = np.array([
            [rx * np.sin(theta0) * np.cos(phi0)],
            [ry * np.sin(theta0) * np.sin(phi0)],
            [rz * np.cos(theta0)],
        ])
        def stim_mask(x):
            d2 = np.sum((x - x0) ** 2, axis=0)
            return (d2 < radius**2).astype(np.float64)
        return stim_mask

    p = APParams(D=0.1, a=0.01, b=0.15, dt=0.05, tend=150.0, save_every=2,
                 stim_amp=1.0, stim_duration=3.0)

    # Multi-cycle stimulation (baseline-frame sites; remapped per F via
    # reduce_bend_site): pace the first beats from one corner and a later beat
    # from a different site to mimic an ectopic focus / shifting pacing lead.
    CYCLE_LENGTH = 60.0
    stim_sites = [(0.18, 0.20), (0.18, 0.20), (0.80, 2.50)]

    # ------------------------------------------------------------------ #
    # Heightened-excitability mode: D is uniform (no fibrosis); the patch instead
    # has a raised excitability -- a set to a fixed negative value and/or k scaled
    # up -- mirroring solveAP_2D_jax.run_single_simulation. The label is the patch
    # region itself (D no longer distinguishes it). EX_FT=1 generates a 10-sim
    # fine-tuning SET (varied site + curvature, sampled k_factor); else one sim at
    # the held-out eval frame (site (0.9,1.3), F=2.0).
    # ------------------------------------------------------------------ #
    if MODE.startswith("excitability"):
        NEG_A = -0.025                         # in-patch a (2D neg_a_value)
        K_FACTOR = float(os.environ.get("K_FACTOR", "1.5"))  # in-patch k *= (2D 1.2-2.0)
        EX_F = float(os.environ.get("EX_F", "2.0"))  # single-sim curvature (F=2.0 less
        #   curved kappa 2.23e-3; F=0.94 more curved kappa 1.01e-2, matches paper grid)
        do_neg_a = MODE in ("excitability_neg_a", "excitability_both")
        do_higher_k = MODE in ("excitability_higher_k", "excitability_both")
        EX_RADIUS = 3.0

        def run_ex_sim(F, th0, ph0, k_factor, stem, write_xdmf):
            """One excitability sim at curvature F and baseline site (th0, ph0).
            Writes <stem>.npz (+ traces, optional XDMF). Returns npz path on rank 0."""
            RX, RY, RZ = RX0 * F, RY0 * F, RZ0 * F
            THETA_RANGE = _reduce_bend_range(THETA_RANGE0, F)
            PHI_RANGE = _reduce_bend_range(PHI_RANGE0, F)
            msh = make_ellipsoid_patch(rx=RX, ry=RY, rz=RZ,
                                       theta_range=THETA_RANGE, phi_range=PHI_RANGE,
                                       n_theta=120, n_phi=160)
            stim_mask_fns = [
                make_stim_mask_on_ellipsoid(RX, RY, RZ, *reduce_bend_site(th, ph, F), 1.5)
                for th, ph in stim_sites
            ]
            stimuli = make_cyclic_stimuli(stim_mask_fns, cycle_length=CYCLE_LENGTH)
            electrodes, elec_thetaphi = make_electrodes_on_shell(
                rx=RX, ry=RY, rz=RZ, offset=0.01,
                theta_range=THETA_RANGE, phi_range=PHI_RANGE, n_theta_e=12, n_phi_e=16)
            node_coords = fem.functionspace(msh, ("Lagrange", 1)).tabulate_dof_coordinates()

            ex_center = ellipsoid_point(RX, RY, RZ, *reduce_bend_site(th0, ph0, F))
            a_field = (make_nodal_patch_field(msh, base=p.a, factor=NEG_A / p.a,
                            center_xyz=ex_center, radius=EX_RADIUS, name="a")
                       if do_neg_a else None)
            k_field = (make_nodal_patch_field(msh, base=p.k, factor=k_factor,
                            center_xyz=ex_center, radius=EX_RADIUS, name="k")
                       if do_higher_k else None)

            # D uniform -> pass D_field=None to BOTH the solve and the phie source.
            xdmf_path = out_path(f"AP_curved_surface_{stem}", "xdmf") if write_xdmf else None
            V, W, snap_t, snap_V = solve_aliev_panfilov(
                msh, p, stimuli=stimuli, D_field=None,
                a_field=a_field, k_field=k_field, xdmf_path=xdmf_path)
            phie = compute_phie(msh, snap_V, electrodes, gain=1.0, D_field=None)

            # Label = the excitability patch (D no longer marks it). D_nodal kept
            # uniform for back-compat with surface_to_graph.py's schema.
            d2_nodal = np.sum((node_coords - ex_center[None, :]) ** 2, axis=1)
            label_nodal = (d2_nodal < EX_RADIUS**2).astype(np.int64)
            D_nodal = np.full(node_coords.shape[0], p.D, dtype=np.float64)

            if msh.comm.rank != 0:
                return None
            n_ab = int(label_nodal.sum())
            print(f"[{MODE} {stem}] F={F:g} site=({th0:.2f},{ph0:.2f}) "
                  f"k_factor={k_factor:g} center={np.round(ex_center, 2)}  "
                  f"phie{phie.shape} range[{phie.min():.2e}, {phie.max():.2e}]  "
                  f"V_end[{V.x.array.min():.3f}, {V.x.array.max():.3f}]  "
                  f"abnormal {n_ab} ({100 * n_ab / node_coords.shape[0]:.1f}%)")
            npz_path = out_path(stem, "npz")
            np.savez(npz_path,
                     phie=phie, t=snap_t, electrodes=electrodes,
                     elec_thetaphi=elec_thetaphi,
                     node_coords=node_coords, D_nodal=D_nodal, label_nodal=label_nodal,
                     D0=p.D, fib_factor=1.0, a=p.a, b=p.b, k=p.k,
                     mode=MODE,
                     neg_a=(NEG_A if do_neg_a else p.a),
                     k_factor=(k_factor if do_higher_k else 1.0),
                     fib_center=ex_center, fib_radius=EX_RADIUS,
                     rx=RX, ry=RY, rz=RZ,
                     theta_range=np.asarray(THETA_RANGE, dtype=np.float64),
                     phi_range=np.asarray(PHI_RANGE, dtype=np.float64),
                     flatten_factor=F, rx0=RX0, ry0=RY0, rz0=RZ0,
                     theta_range0=np.asarray(THETA_RANGE0, dtype=np.float64),
                     phi_range0=np.asarray(PHI_RANGE0, dtype=np.float64))
            plot_phie_traces(snap_t, phie, n=5, seed=0,
                             path=out_path(f"phie_traces_{stem}", "png"),
                             electrodes=electrodes, lowD_center=ex_center,
                             lowD_radius=EX_RADIUS)
            print(f"  wrote {npz_path}")
            return npz_path

        if EX_FT:
            # 10-sim fine-tuning set: sites EXCLUDE the held-out eval site (0.9,1.3)
            # and curvatures bracket F=2.0, mirroring the fibrosis reduce_bend_ft
            # set. k_factor is sampled per sim in (1.2, 2.0) for higher_k / both.
            EX_FT_SITES = [
                (0.62, 1.05), (0.62, 1.60), (0.62, 2.10),
                (0.86, 0.95),               (0.86, 2.15),
                (1.08, 1.05), (1.08, 1.60), (1.08, 2.10),
                (0.74, 1.85), (1.00, 1.50),
            ]
            EX_FT_FACTORS = list(np.round(np.linspace(1.6, 2.4, len(EX_FT_SITES)), 3))
            rng = np.random.default_rng(0)
            written = []
            for ii, ((th0, ph0), F) in enumerate(zip(EX_FT_SITES, EX_FT_FACTORS)):
                kf = float(rng.uniform(1.2, 2.0)) if do_higher_k else 1.0
                npz_path = run_ex_sim(F, th0, ph0, kf,
                                      stem=f"phie_surface_ex{ii:02d}", write_xdmf=False)
                if npz_path is not None:
                    written.append(npz_path)
            if MPI.COMM_WORLD.rank == 0:
                print(f"\nWrote {len(written)} excitability fine-tuning sim(s) to {OUT_DIR}/")
        elif EX_EVAL:
            # 5-site held-out eval set at curvature EX_F (F in the stem so F=2.0 and
            # F=0.94 sets coexist in <MODE>_eval/). For mean +/- std test metrics.
            written = []
            for ii, (th0, ph0) in enumerate(EVAL_SITES):
                npz_path = run_ex_sim(EX_F, th0, ph0, K_FACTOR,
                                      stem=f"phie_surface_ev{ii:02d}_F{EX_F:g}",
                                      write_xdmf=False)
                if npz_path is not None:
                    written.append(npz_path)
            if MPI.COMM_WORLD.rank == 0:
                print(f"\nWrote {len(written)} excitability eval sim(s) "
                      f"(F={EX_F:g}) to {OUT_DIR}/")
        else:
            # Held-out single eval sim at site (0.9,1.3). Default F=2.0 keeps the
            # stem "phie_surface"; a different EX_F (e.g. 0.94, more curved) writes
            # a distinct stem so it does not overwrite the F=2.0 test sim.
            stem = "phie_surface" if EX_F == 2.0 else f"phie_surface_F{EX_F:g}"
            run_ex_sim(EX_F, 0.9, 1.3, K_FACTOR, stem=stem, write_xdmf=True)
        raise SystemExit(0)

    # ---- few-shot fine-tuning set: per-sim fibrosis location AND curvature.
    # Fibrosis centers are baseline-frame (theta, phi) spread across the patch
    # interior, clear of the pole edge and stim sites, and EXCLUDING the held-out
    # eval location (0.9, 1.3). Because Option C preserves physical geometry
    # across F, a site valid at one curvature is valid at all of them.
    FIB_RADIUS = 3.0
    FIB_FACTOR = 0.2
    A_FACTOR = 2.0      # in-patch a/b modification mirrors the flat generator
    B_FACTOR = 0.85
    FT_FIB_SITES = [
        (0.62, 1.05), (0.62, 1.60), (0.62, 2.10),
        (0.86, 0.95),               (0.86, 2.15),
        (1.08, 1.05), (1.08, 1.60), (1.08, 2.10),
        (0.74, 1.85), (1.00, 1.50),
    ]
    # A slightly different curvature per sim, bracketing the eval's F=2.0 (held
    # out). Tweak the range to taste; physical size/features stay fixed (Option C).
    FT_FLATTEN_FACTORS = list(np.round(
        np.linspace(1.6, 2.4, len(FT_FIB_SITES)), 3))

    # ---- held-out fibrosis EVAL sim(s) at a chosen curvature ----------------
    # FIB_F=<F> generates fibrosis eval sim(s) at curvature F, matching
    # reduce_bend_s2's construction. Default: ONE sim at site (0.9,1.3) ->
    # reduce_bend_more/ (the paper-grid more-curved column). FIB_EVAL=1: the 5-site
    # held-out EVAL set -> reduce_bend_eval/ (for mean +/- std test metrics).
    FIB_F = os.environ.get("FIB_F")
    FIB_EVAL = os.environ.get("FIB_EVAL", "0") == "1"
    if FIB_F is not None:
        F = float(FIB_F)
        fib_dirname = "reduce_bend_eval" if FIB_EVAL else "reduce_bend_more"
        fib_out = os.path.join("sim_results", fib_dirname)
        os.makedirs(fib_out, exist_ok=True)

        def fib_path(stem, ext):
            return os.path.join(fib_out, f"{stem}_{fib_dirname}.{ext}")

        def run_fib_sim(th0, ph0, stem, write_xdmf):
            RX, RY, RZ = RX0 * F, RY0 * F, RZ0 * F
            THETA_RANGE = _reduce_bend_range(THETA_RANGE0, F)
            PHI_RANGE = _reduce_bend_range(PHI_RANGE0, F)
            msh = make_ellipsoid_patch(rx=RX, ry=RY, rz=RZ,
                                       theta_range=THETA_RANGE, phi_range=PHI_RANGE,
                                       n_theta=120, n_phi=160)
            stim_mask_fns = [
                make_stim_mask_on_ellipsoid(RX, RY, RZ, *reduce_bend_site(th, ph, F), 1.5)
                for th, ph in stim_sites
            ]
            stimuli = make_cyclic_stimuli(stim_mask_fns, cycle_length=CYCLE_LENGTH)
            electrodes, elec_thetaphi = make_electrodes_on_shell(
                rx=RX, ry=RY, rz=RZ, offset=0.01,
                theta_range=THETA_RANGE, phi_range=PHI_RANGE, n_theta_e=12, n_phi_e=16)
            node_coords = fem.functionspace(msh, ("Lagrange", 1)).tabulate_dof_coordinates()

            fib_center = ellipsoid_point(RX, RY, RZ, *reduce_bend_site(th0, ph0, F))
            D_field = make_D_field(msh, D0=p.D, center_xyz=fib_center,
                                   radius=FIB_RADIUS, fib_factor=FIB_FACTOR)
            a_field = make_nodal_patch_field(msh, base=p.a, factor=A_FACTOR,
                                             center_xyz=fib_center, radius=FIB_RADIUS, name="a")
            b_field = make_nodal_patch_field(msh, base=p.b, factor=B_FACTOR,
                                             center_xyz=fib_center, radius=FIB_RADIUS, name="b")
            V, W, snap_t, snap_V = solve_aliev_panfilov(
                msh, p, stimuli=stimuli, D_field=D_field,
                a_field=a_field, b_field=b_field,
                xdmf_path=(fib_path(f"AP_curved_surface_{stem}", "xdmf")
                           if write_xdmf else None))
            phie = compute_phie(msh, snap_V, electrodes, gain=1.0, D_field=D_field)

            d2_nodal = np.sum((node_coords - fib_center[None, :]) ** 2, axis=1)
            D_nodal = np.full(node_coords.shape[0], p.D, dtype=np.float64)
            D_nodal[d2_nodal < FIB_RADIUS**2] = p.D * FIB_FACTOR

            if msh.comm.rank != 0:
                return None
            n_fib = int(np.count_nonzero(D_nodal < p.D))
            print(f"[fibrosis {stem} F={F:g}] site=({th0:.2f},{ph0:.2f}) "
                  f"center={np.round(fib_center, 2)}  phie{phie.shape} "
                  f"range[{phie.min():.2e}, {phie.max():.2e}]  "
                  f"V_end[{V.x.array.min():.3f}, {V.x.array.max():.3f}]  "
                  f"fibrotic {n_fib} ({100 * n_fib / node_coords.shape[0]:.1f}%)")
            npz_path = fib_path(stem, "npz")
            np.savez(npz_path,
                     phie=phie, t=snap_t, electrodes=electrodes,
                     elec_thetaphi=elec_thetaphi,
                     node_coords=node_coords, D_nodal=D_nodal,
                     D0=p.D, fib_factor=FIB_FACTOR, a=p.a, b=p.b,
                     a_factor=A_FACTOR, b_factor=B_FACTOR,
                     fib_center=fib_center, fib_radius=FIB_RADIUS,
                     rx=RX, ry=RY, rz=RZ,
                     theta_range=np.asarray(THETA_RANGE, dtype=np.float64),
                     phi_range=np.asarray(PHI_RANGE, dtype=np.float64),
                     flatten_factor=F, rx0=RX0, ry0=RY0, rz0=RZ0,
                     theta_range0=np.asarray(THETA_RANGE0, dtype=np.float64),
                     phi_range0=np.asarray(PHI_RANGE0, dtype=np.float64))
            plot_phie_traces(snap_t, phie, n=5, seed=0,
                             path=fib_path(f"phie_traces_{stem}", "png"),
                             electrodes=electrodes, lowD_center=fib_center,
                             lowD_radius=FIB_RADIUS)
            print(f"  wrote {npz_path}")
            return npz_path

        if FIB_EVAL:
            written = []
            for ii, (th0, ph0) in enumerate(EVAL_SITES):
                npz_path = run_fib_sim(th0, ph0,
                                       f"phie_surface_ev{ii:02d}_F{F:g}", write_xdmf=False)
                if npz_path is not None:
                    written.append(npz_path)
            if MPI.COMM_WORLD.rank == 0:
                print(f"\nWrote {len(written)} fibrosis eval sim(s) (F={F:g}) to {fib_out}/")
        else:
            # single held-out sim (0.9,1.3) -> keeps the paper-grid more-curved name
            run_fib_sim(0.9, 1.3, f"phie_surface_fibF{F:g}", write_xdmf=True)
        raise SystemExit(0)

    written = []
    for ii, ((th0, ph0), F) in enumerate(zip(FT_FIB_SITES, FT_FLATTEN_FACTORS)):
        # ---- geometry for THIS sim's curvature F ----
        RX, RY, RZ = RX0 * F, RY0 * F, RZ0 * F
        THETA_RANGE = _reduce_bend_range(THETA_RANGE0, F)
        PHI_RANGE = _reduce_bend_range(PHI_RANGE0, F)
        msh = make_ellipsoid_patch(rx=RX, ry=RY, rz=RZ,
                                   theta_range=THETA_RANGE, phi_range=PHI_RANGE,
                                   n_theta=120, n_phi=160)

        stim_mask_fns = [
            make_stim_mask_on_ellipsoid(RX, RY, RZ, *reduce_bend_site(th, ph, F), 1.5)
            for th, ph in stim_sites
        ]
        stimuli = make_cyclic_stimuli(stim_mask_fns, cycle_length=CYCLE_LENGTH)

        electrodes, elec_thetaphi = make_electrodes_on_shell(
            rx=RX, ry=RY, rz=RZ, offset=0.01,
            theta_range=THETA_RANGE, phi_range=PHI_RANGE,
            n_theta_e=12, n_phi_e=16)
        node_coords = fem.functionspace(msh, ("Lagrange", 1)).tabulate_dof_coordinates()

        # ---- fibrosis patch at this sim's site ----
        fib_center = ellipsoid_point(RX, RY, RZ, *reduce_bend_site(th0, ph0, F))
        D_field = make_D_field(msh, D0=p.D, center_xyz=fib_center,
                               radius=FIB_RADIUS, fib_factor=FIB_FACTOR)
        a_field = make_nodal_patch_field(msh, base=p.a, factor=A_FACTOR,
                                         center_xyz=fib_center, radius=FIB_RADIUS, name="a")
        b_field = make_nodal_patch_field(msh, base=p.b, factor=B_FACTOR,
                                         center_xyz=fib_center, radius=FIB_RADIUS, name="b")

        # Pass D_field so fibrosis enters BOTH the propagation operator and the
        # phie source directly: div(D grad V). No per-sim V-XDMF (10 of them would
        # be heavy); the npz holds everything surface_to_graph.py needs.
        V, W, snap_t, snap_V = solve_aliev_panfilov(
            msh, p, stimuli=stimuli, D_field=D_field,
            a_field=a_field, b_field=b_field, xdmf_path=None)
        phie = compute_phie(msh, snap_V, electrodes, gain=1.0, D_field=D_field)

        d2_nodal = np.sum((node_coords - fib_center[None, :]) ** 2, axis=1)
        D_nodal = np.full(node_coords.shape[0], p.D, dtype=np.float64)
        D_nodal[d2_nodal < FIB_RADIUS**2] = p.D * FIB_FACTOR

        if msh.comm.rank == 0:
            n_fib = int(np.count_nonzero(D_nodal < p.D))
            print(f"[sim {ii:02d}] F={F:g} fib baseline=({th0:.2f}, {ph0:.2f}) "
                  f"center={np.round(fib_center, 2)}  phie{phie.shape} "
                  f"range[{phie.min():.2e}, {phie.max():.2e}]  "
                  f"fibrotic nodes {n_fib} ({100 * n_fib / node_coords.shape[0]:.1f}%)")
            npz_path = out_path(f"phie_surface_fib{ii:02d}", "npz")
            np.savez(npz_path,
                     phie=phie, t=snap_t, electrodes=electrodes,
                     elec_thetaphi=elec_thetaphi,
                     node_coords=node_coords, D_nodal=D_nodal,
                     D0=p.D, fib_factor=FIB_FACTOR, a=p.a, b=p.b,
                     a_factor=A_FACTOR, b_factor=B_FACTOR,
                     fib_center=fib_center, fib_radius=FIB_RADIUS,
                     rx=RX, ry=RY, rz=RZ,
                     theta_range=np.asarray(THETA_RANGE, dtype=np.float64),
                     phi_range=np.asarray(PHI_RANGE, dtype=np.float64),
                     flatten_factor=F,
                     rx0=RX0, ry0=RY0, rz0=RZ0,
                     theta_range0=np.asarray(THETA_RANGE0, dtype=np.float64),
                     phi_range0=np.asarray(PHI_RANGE0, dtype=np.float64))
            plot_phie_traces(snap_t, phie, n=5, seed=0,
                             path=out_path(f"phie_traces_fib{ii:02d}", "png"),
                             electrodes=electrodes, lowD_center=fib_center,
                             lowD_radius=FIB_RADIUS)
            plot_surface_with_electrodes(
                msh, electrodes,
                path=out_path(f"surface_electrodes_fib{ii:02d}", "png"))
            written.append(npz_path)

    if msh.comm.rank == 0:
        print(f"\nWrote {len(written)} fine-tuning sim(s) to {OUT_DIR}/ "
              f"(F in {min(FT_FLATTEN_FACTORS):g}..{max(FT_FLATTEN_FACTORS):g}):")
        for w in written:
            print(f"  {w}")
