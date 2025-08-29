import os
import napari
import numpy as np
import dask.array as da
from imaris_ims_file_reader.ims import ims
import psutil

# ========================
# Constants
# ========================
IMG_PATH = "../images/NMLNP001-AN6.ims"
MEM_LIMIT = 0.5  # fraction of total RAM allowed

# ========================
# Helper functions
# ========================
def estimate_ram_usage(shape, dtype):
    """Estimate memory in GB needed for array with given shape and dtype."""
    return np.prod(shape) * np.dtype(dtype).itemsize / 1e9

# ========================
# Load IMS
# ========================
print(f"Loading IMS file: {IMG_PATH}")
ims_obj = ims(IMG_PATH)
raw_shape = ims_obj.shape
dtype = ims_obj.dtype
print(f"Raw IMS shape: {raw_shape}")
print(f"IMS dtype: {dtype}")

# Estimate RAM usage
est_size_gb = estimate_ram_usage(raw_shape, dtype)
total_ram_gb = psutil.virtual_memory().total / 1e9
mem_limit_gb = total_ram_gb * MEM_LIMIT
print(f"Estimated IMS size: {est_size_gb:.2f} GB, MEM_LIMIT: {mem_limit_gb:.2f} GB")

# ========================
# Decide load mode
# ========================
if est_size_gb < mem_limit_gb:
    print("Loading full image into RAM")

    # Explicit full slicing to get proper ndarray
    data = ims_obj[:, :, :, :, :]
    print(f"Raw data shape from ims_obj: {data.shape}")

    # Collapse leading singleton T dimension if present
    if data.shape[0] == 1:
        data = data[0]
        print(f"Collapsed leading T dimension: {data.shape}")
else:
    print("Using Dask for lazy loading")
    chunk_size = (1, 1, 1, 512, 512)  # adjust chunks for Z/Y/X
    data = da.from_array(ims_obj[:, :, :, :, :], chunks=chunk_size, lock=True)
    print(f"Dask array shape: {data.shape}, chunks: {data.chunks}")

# ========================
# Detect channels
# ========================
if data.ndim == 4:
    channel_axis = 0
    print(f"Detected 3D + channels: {data.shape}")
elif data.ndim == 3:
    channel_axis = None
    print(f"Detected 3D data without explicit channels: {data.shape}")
else:
    channel_axis = None

# Assign channel colors
num_channels = data.shape[channel_axis] if channel_axis is not None else 1
default_colors = ["blue", "green", "red", "white"]
channel_colors = default_colors[:num_channels]

# ========================
# Standalone Napari test
# ========================
viewer = napari.Viewer()
print("Adding image to Napari...")
layer = viewer.add_image(
    data,
    channel_axis=channel_axis,
    multiscale=False,
    colormap=channel_colors
)
print("Layer added, starting Napari run...")
napari.run()
