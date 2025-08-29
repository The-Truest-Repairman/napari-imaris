import napari
import numpy as np
import dask.array as da
from imaris_ims_file_reader.ims import ims

# ========================
# Constants
# ========================
IMG_PATH = "../images/NMLNP001-AN6.ims"

# ========================
# Load IMS object
# ========================
print(f"Opening IMS file: {IMG_PATH}")
ims_obj = ims(IMG_PATH)

n_levels = ims_obj.ResolutionLevels
print(f"IMS reports {n_levels} resolution levels")
print(f"TimePoints: {ims_obj.TimePoints}, Channels: {ims_obj.Channels}")
print(f"Base shape (level 0): {ims_obj.shape}, dtype: {ims_obj.dtype}")

# ========================
# Build pyramid
# ========================
pyramid = []
for lvl in range(n_levels):
    print(f"\n--- Resolution level {lvl} ---")
    ims_obj.change_resolution_lock(lvl)
    arr = ims_obj[:, :, :, :, :]   # full slice at this resolution
    print(f" Level {lvl} shape={arr.shape}, dtype={arr.dtype}")

    # Collapse leading T dim if singleton
    if arr.shape[0] == 1:
        arr = arr[0]
        print(f"  Collapsed T axis → shape={arr.shape}")

    # Wrap in Dask if huge
    est_gb = np.prod(arr.shape) * arr.dtype.itemsize / 1e9
    if est_gb > 2:  # arbitrary cutoff for demo
        arr = da.from_array(arr, chunks="auto")
        print(f"  Converted to Dask → chunks={arr.chunks}")

    pyramid.append(arr)

# ========================
# Detect channels
# ========================
base = pyramid[0]
if base.ndim == 4:   # (C,Z,Y,X)
    channel_axis = 0
    print(f"Detected channels: shape={base.shape}, channel_axis={channel_axis}")
elif base.ndim == 3: # (Z,Y,X)
    channel_axis = None
    print(f"No explicit channel axis, shape={base.shape}")
else:
    channel_axis = None

# Assign default colors
num_channels = base.shape[channel_axis] if channel_axis is not None else 1
default_colors = ["blue", "green", "red", "white"]
channel_colors = default_colors[:num_channels]
print(f"Using channel colors: {channel_colors}")

# ========================
# Napari viewer
# ========================
print("Launching Napari with multiscale pyramid...")
viewer = napari.Viewer()
viewer.add_image(
    pyramid,
    multiscale=True,
    channel_axis=channel_axis,
    colormap=channel_colors
)
napari.run()
