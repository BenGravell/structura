import numpy as np
import matplotlib.cm as cm
from PIL import Image
import pyvista as pv


def upscale_image(img_array, scale_factor, method=None):
    # Convert numpy array to PIL Image
    img = Image.fromarray(img_array)

    # Calculate the new dimensions
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)

    # Resize the image using the Catrom interpolation method
    resizer = getattr(Image, method)
    img_resized = img.resize((new_width, new_height), resizer)

    # Convert the PIL Image back to a numpy array
    img_resized_array = np.array(img_resized)

    return img_resized_array


def reshape(x, nelx, nely):
    return x.reshape((nelx, nely)).T


def clip(x):
    return np.clip(x, 0, 1)


def x2frame(x, cmap=None, upscale_factor=1, upscale_method="NEAREST", mirror=False):
    frame = np.copy(x)
    if cmap is None:
        cmap = cm.get_cmap("Greys")
    frame = upscale_image(frame, upscale_factor, upscale_method)
    if mirror:
        frame = np.hstack([np.fliplr(frame), frame])
    frame = cmap(frame)
    frame = (frame[:, :, 0:3] * 255).astype(np.uint8)  # Convert to [0,255] RGB image
    return frame


def get_pv_plotter(depth_data_in, mirror=False, cmap=None):
    z = np.copy(depth_data_in)

    if mirror:
        # Mirror
        z = np.hstack([np.fliplr(z), z])

    z = np.flipud(z)

    # Create coordinate data
    SCALE = 40
    ny, nx = z.shape
    nmax = max(nx, ny)

    # Normalize dimensions by the max dimension to get unitless dimensions
    nxu = nx / nmax
    nyu = ny / nmax

    ux = np.linspace(0, SCALE * nxu, nx + 1)
    uy = np.linspace(0, SCALE * nyu, ny + 1)

    # Thickness resolution
    nz = 20
    uz = np.linspace(-1, 1, 2 * nz + 1)

    mux, muy, muz = np.meshgrid(ux, uy, uz)
    mesh = pv.StructuredGrid(mux, muy, muz)

    mask = np.abs((muz[:-1, :-1, :-1] + muz[:-1, :-1, 1:]) / 2) > z[:, :, None]
    flat_mask = mask.flatten(order="F")

    # Thickness scalar values for cells
    thickness = np.repeat(z[:, :, None], 2 * nz, -1).flatten(order="F")

    mesh["Thickness"] = thickness

    # Cast the StructuredGrid mesh to an UnstructuredGrid
    # NOTE: This is required for the remove_cells() method to work.
    mesh = mesh.cast_to_unstructured_grid()
    # Remove cells corresponding to areas where there is no thickness
    mesh = mesh.remove_cells(flat_mask)

    # Set up plotter
    plotter = pv.Plotter()

    # Add mesh
    add_mesh_kwargs = dict(scalars="Thickness", show_edges=False, edge_color="black")
    if cmap is None:
        add_mesh_kwargs["color"] = "white"
    else:
        add_mesh_kwargs["cmap"] = cmap
    plotter.add_mesh(mesh, **add_mesh_kwargs)

    # Final touches
    plotter.window_size = [1000, 400]
    plotter.background_color = "white"
    plotter.camera.zoom(3.0)
    plotter.enable_parallel_projection()
    plotter.view_xy()

    return plotter
