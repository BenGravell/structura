from time import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import constants
import utils
import math_utils
import app_options as ao
from app_options import FilterType


def initialize_density(method, nx, ny):
    n = nx * ny
    if method == "constant":
        x = np.ones(n, dtype=float)
    elif method == "random":
        x = np.random.uniform(size=n)
    else:
        raise ValueError(f"Invalid density initialization method {method}.")
    return x


def optimality_criterion_based_step(x, g, dcdx, dvdx, move=0.1):
    # Initial bounds for the Lagrange multiplier
    lagrange_lo = 0
    lagrange_hi = 1e9

    # Numerical damping coefficient
    eta = 0.5

    # Bisection convergence tolerance for Lagrange multiplier
    tol = 1e-3

    # Maximum change in design variables permitted
    # NOTE: if x is too far from feasibility and if move is too small,
    # it may cause the bisection to fail to find a suitable multiplier
    # move = 0.1

    # Max iters for bisection search
    max_iters = 10000

    # Initialize outputs
    xnew = x
    gnew = g

    lagrange_delta = lagrange_hi - lagrange_lo
    lagrange_sum = lagrange_lo + lagrange_hi

    # Bisection algorithm to find suitable Lagrange multiplier
    for iter_count in range(max_iters):
        if lagrange_sum <= 0:
            raise RuntimeError(
                "Bisection bounds have collapsed to zero, failed to find a feasible Lagrange multiplier!"
            )

        if lagrange_delta / lagrange_sum < tol:
            return xnew, gnew

        lagrange_mid = 0.5 * (lagrange_hi + lagrange_lo)

        # Get the proposed new x
        B = -dcdx / (lagrange_mid * dvdx)
        B_damped = B**eta
        xnew = x * B_damped

        # Clamp the delta to [-move, move] to prevent taking too large of a step
        dx = xnew - x
        dx = np.clip(dx, -move, move)
        xnew = x + dx

        # Clamp the new x to [0, 1]
        xnew = np.clip(xnew, 0, 1)

        # Get final delta x
        dx = xnew - x

        # Get new g
        gnew = g + np.sum(dvdx * dx)

        # Update bisection bounds for Lagrange multiplier
        if gnew > 0:
            lagrange_lo = lagrange_mid
        else:
            lagrange_hi = lagrange_mid

        lagrange_delta = lagrange_hi - lagrange_lo
        lagrange_sum = lagrange_lo + lagrange_hi

    raise RuntimeError("Maximum Iterations exceeded in Lagrange multiplier bisection!")


def design_variable_to_youngs_modulus(x, penal):
    return constants.YOUNGS_MODULUS_MIN + x**penal * (constants.YOUNGS_MODULUS_MAX - constants.YOUNGS_MODULUS_MIN)


def optimize(options: ao.Options, image_container=None):
    (
        nel,
        volfrac,
        rmin,
        penal,
        filter_type,
        density_initialization_method,
        move,
        change_tol,
        max_iters,
        cmap,
        upscale_factor,
        upscale_method,
        mirror,
    ) = options.unpack()
    nelx, nely = math_utils.compute_rectangle_dims_from_area(nel, ratio=3)

    # dofs:
    n_dof_x = nelx + 1
    n_dof_y = nely + 1
    n_dof = 2 * n_dof_x * n_dof_y

    # Allocate design variables (as array), initialize and allocate sens.
    x = initialize_density(density_initialization_method, nelx, nely)

    # Ensure the initial density is feasible
    volfrac_x = x.sum() / x.size
    x *= volfrac / volfrac_x

    xold = x.copy()
    xPhys = x.copy()
    g = 0  # must be initialized to use the Nguyen/Paulino Optimality Criterion approach
    dcdx = np.zeros((nely, nelx), dtype=float)

    # Finite Element Model: Build the index vectors for the for coo matrix format.
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1]
            )

    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    # Filter: Build (and assemble) the index+data vectors for the COO matrix format
    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
            kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
            ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
            ll2 = int(np.minimum(j + np.ceil(rmin), nely))
            for index_k in range(kk1, kk2):
                for index_l in range(ll1, ll2):
                    col = index_k * nely + index_l
                    fac = rmin - np.sqrt(((i - index_k) * (i - index_k) + (j - index_l) * (j - index_l)))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc = cc + 1

    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)

    # Boundary conditions and support
    dofs = np.arange(n_dof)
    fixed = np.union1d(dofs[0 : 2 * (nely + 1) : 2], np.array([n_dof - 1]))
    free = np.setdiff1d(dofs, fixed)

    # Initialize force vector
    f = np.zeros((n_dof, 1))

    # Initialize displacement vector
    u = np.zeros((n_dof, 1))

    # Loads
    f[1, 0] = -1

    # Set loop counter and gradient vectors
    loop = 0
    change = 1e9
    dvdx = np.ones(nely * nelx)
    dcdx = np.ones(nely * nelx)
    ce = np.ones(nely * nelx)

    time_of_last_draw = time()
    time_elapsed_since_last_draw = 0.0

    while change > change_tol and loop < max_iters:
        loop = loop + 1
        # Setup and solve FE problem

        # Sparse stiffness matrix data
        sK = (
            (constants.ELEMENT_STIFFNESS_MATRIX.flatten()[np.newaxis]).T
            * (
                constants.YOUNGS_MODULUS_MIN
                + xPhys**penal * (constants.YOUNGS_MODULUS_MAX - constants.YOUNGS_MODULUS_MIN)
            )
        ).flatten(order="F")

        # Stiffness matrix
        K = coo_matrix((sK, (iK, jK)), shape=(n_dof, n_dof)).tocsc()

        # Remove constrained dofs from matrix
        K_free = K[free, :][:, free]

        # Solve system for displacements
        u[free, 0] = spsolve(K_free, f[free, 0])

        # unit complicance - quadratic form
        u_vec = u[edofMat].reshape(nelx * nely, 8)
        ce[:] = np.einsum("ij,jk,ik->i", u_vec, constants.ELEMENT_STIFFNESS_MATRIX, u_vec)

        # Objective
        youngs_modulus_elementwise = design_variable_to_youngs_modulus(xPhys, penal)
        objective_value = np.sum(youngs_modulus_elementwise * ce)

        # Sensitivity of objective to design variables i.e. the gradient
        dcdx[:] = (-penal * xPhys ** (penal - 1) * (constants.YOUNGS_MODULUS_MAX - constants.YOUNGS_MODULUS_MIN)) * ce

        # Sensitivity of volume to design variables
        dvdx[:] = np.ones(nely * nelx)

        # Filtering:
        if filter_type == FilterType.NO_FILTER:
            pass
        elif filter_type == FilterType.DENSITY_FILTER:
            dcdx[:] = np.asarray(H * (dcdx[np.newaxis].T / Hs))[:, 0]
            dvdx[:] = np.asarray(H * (dvdx[np.newaxis].T / Hs))[:, 0]
        elif filter_type == FilterType.SENSITIVITY_FILTER:
            dcdx[:] = np.asarray((H * (x * dcdx))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)

        # Optimality criteria
        xold[:] = x
        x[:], g = optimality_criterion_based_step(x, g, dcdx, dvdx, move)

        # Filter design variables
        if filter_type == FilterType.NO_FILTER:
            xPhys[:] = x
        elif filter_type == FilterType.DENSITY_FILTER:
            xPhys[:] = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]
        elif filter_type == FilterType.SENSITIVITY_FILTER:
            xPhys[:] = x

        # Compute the change by the inf norm
        change = np.linalg.norm(x.reshape(nelx * nely, 1) - xold.reshape(nelx * nely, 1), np.inf)

        time_elapsed_since_last_draw = time() - time_of_last_draw
        frame = None
        if image_container is not None:
            if time_elapsed_since_last_draw > constants.DRAW_UPDATE_SEC:
                # Draw the frame
                frame = utils.x2frame(xPhys, nelx, nely, cmap, upscale_factor, upscale_method, mirror)
                image_container.image(frame, caption=f"Iteration {loop:6d}, Objective = {objective_value:12.3f}", use_column_width=True)
                time_of_last_draw = time()
    
    # Always compute a frame at the very end to return
    frame = utils.x2frame(xPhys, nelx, nely, cmap, upscale_factor, upscale_method, mirror)
    return frame, objective_value


if __name__ == "__main__":
    options = ao.Options()
    optimize(options)
