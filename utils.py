import numpy as np
import matplotlib.cm as cm
from PIL import Image


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


def x2frame(x, nelx, nely, cmap=None, upscale_factor=1, upscale_method="NEAREST", mirror=False):
    if cmap is None:
        cmap = cm.get_cmap("Greys")
    frame = x.reshape((nelx, nely)).T
    frame = np.clip(frame, 0, 1)
    frame = upscale_image(frame, upscale_factor, upscale_method)
    if mirror:
        frame = np.hstack([np.fliplr(frame), frame])
    frame = cmap(frame)
    frame = (frame[:, :, 0:3] * 255).astype(np.uint8)  # Convert to [0,255] RGB image
    return frame
