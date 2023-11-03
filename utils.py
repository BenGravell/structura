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


def get_pv_xyz_data(depth_data_in, mirror=False, threshold=0.01):
    z = np.copy(depth_data_in)

    if mirror:
        # Mirror
        z = np.hstack([np.fliplr(z), z])

    z = np.flipud(z)

    # Create coordinate data
    SCALE = 5
    nx, ny = z.shape
    xg = SCALE * np.arange(ny) / nx
    yg = SCALE * np.arange(nx) / nx
    x, y = np.meshgrid(xg, yg)

    x[z < threshold] = None
    y[z < threshold] = None
    z[z < threshold] = None

    return x, y, z


def get_pv_plotter(x, y, z):
    # Set up plotter
    plotter = pv.Plotter()

    # Add surface meshes
    surface_up = pv.StructuredGrid(x, y, z)
    surface_dn = pv.StructuredGrid(x, y, -z)
    for surface in [surface_up, surface_dn]:
        plotter.add_mesh(surface, color=[224, 224, 224], show_edges=False)

    # Final touches
    plotter.window_size = [640, 360]
    plotter.background_color = "white"
    plotter.camera.zoom(3.0)
    plotter.enable_parallel_projection()
    plotter.view_xy()

    return plotter
