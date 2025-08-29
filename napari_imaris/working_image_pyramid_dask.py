import napari
import numpy as np
import dask.array as da
from imaris_ims_file_reader.ims import ims

# ========================
# Constants
# ========================
IMG_PATH = "../images/NMLNP001-AN6.ims"

# ========================
# Helper functions
# ========================
def extract_channel_names(ims_obj):
    """
    Extract channel names from IMS DataSetInfo container.
    Defaults to "Channel {i}" if Name/Description is missing.
    """
    channel_names = []
    try:
        info = ims_obj.hf['DataSetInfo']
        for c in range(ims_obj.Channels):
            ch_name = f"Channel {c}"  # default
            ch_group_name = f"Channel {c}"
            if ch_group_name in info:
                ch_attrs = info[ch_group_name].attrs
                if 'Name' in ch_attrs:
                    ch_name = ch_attrs['Name']
                elif 'Description' in ch_attrs:
                    ch_name = ch_attrs['Description']
            channel_names.append(ch_name)
    except Exception as e:
        print(f"[WARN] Could not extract channel names: {e}")
        channel_names = [f"Channel {i}" for i in range(ims_obj.Channels)]
    
    print(f"[INFO] Detected channels: {channel_names}")
    return channel_names


def get_channel_histograms(ims_obj, dtype):
    """
    Returns a list of (min, max) tuples for each channel.
    Uses DataSetInfo Min/Max if available, otherwise dtype defaults.
    """
    histograms = []
    try:
        info = ims_obj.hf['DataSetInfo']
        for c in range(ims_obj.Channels):
            ch_group_name = f"Channel {c}"
            ch_attrs = info[ch_group_name].attrs if ch_group_name in info else {}
            ch_min = ch_attrs.get('Min', None)
            ch_max = ch_attrs.get('Max', None)

            if ch_min is not None and ch_max is not None:
                histograms.append((float(ch_min), float(ch_max)))
            else:
                # fallback to dtype
                if np.issubdtype(dtype, np.integer):
                    iinfo = np.iinfo(dtype)
                    histograms.append((iinfo.min, iinfo.max))
                elif np.issubdtype(dtype, np.floating):
                    histograms.append((0.0, 1.0))
                else:
                    histograms.append((0.0, 1.0))
    except Exception as e:
        print(f"[WARN] Could not extract channel histogram info: {e}")
        if np.issubdtype(dtype, np.integer):
            iinfo = np.iinfo(dtype)
            histograms = [(iinfo.min, iinfo.max)] * ims_obj.Channels
        else:
            histograms = [(0.0, 1.0)] * ims_obj.Channels
    
    print(f"[INFO] Channel histograms: {histograms}")
    return histograms


def extract_voxel_size(ims_obj):
    """
    Extract voxel size (scale) from Imaris DataSetInfo.
    Returns default [1.0, 1.0, 1.0] if missing.
    """
    try:
        info = ims_obj.hf['DataSetInfo']['Image'].attrs
        voxel_size = [
            float(info.get('ExtMax0', 1.0) - info.get('ExtMin0', 0.0)),
            float(info.get('ExtMax1', 1.0) - info.get('ExtMin1', 0.0)),
            float(info.get('ExtMax2', 1.0) - info.get('ExtMin2', 0.0))
        ]
        print(f"[INFO] Voxel size (X,Y,Z): {voxel_size}")
    except Exception as e:
        print(f"[WARN] Could not extract voxel size: {e}")
        voxel_size = [1.0, 1.0, 1.0]
    return voxel_size


# ========================
# Open IMS file
# ========================
print(f"[INFO] Opening IMS file: {IMG_PATH}")
ims_obj = ims(IMG_PATH)

n_levels = ims_obj.ResolutionLevels
print(f"[INFO] IMS reports {n_levels} resolution levels")
print(f"[INFO] TimePoints: {ims_obj.TimePoints}, Channels: {ims_obj.Channels}")
print(f"[INFO] Base shape (level 0): {ims_obj.shape}, dtype: {ims_obj.dtype}")

# Extract metadata
channel_names = extract_channel_names(ims_obj)
voxel_size = extract_voxel_size(ims_obj)
channel_histograms = get_channel_histograms(ims_obj, ims_obj.dtype)

# ========================
# Build pyramid
# ========================
pyramid = []
for lvl in range(n_levels):
    print(f"\n--- Resolution level {lvl} ---")
    ims_obj.change_resolution_lock(lvl)
    arr = ims_obj[:, :, :, :, :]   # full slice at this resolution
    print(f"[DEBUG] Level {lvl} raw shape={arr.shape}, dtype={arr.dtype}")

    # Collapse leading T dim if singleton
    if arr.shape[0] == 1:
        arr = arr[0]
        print(f"[DEBUG] Collapsed T axis → shape={arr.shape}")

    # Wrap in Dask if large
    est_gb = np.prod(arr.shape) * arr.dtype.itemsize / 1e9
    if est_gb > 2:
        arr = da.from_array(arr, chunks="auto")
        print(f"[DEBUG] Converted to Dask → chunks={arr.chunks}")

    pyramid.append(arr)

# ========================
# Detect channel axis
# ========================
base = pyramid[0]
if base.ndim == 4:   # (C,Z,Y,X)
    channel_axis = 0
    print(f"[INFO] Detected channels in axis {channel_axis}, shape={base.shape}")
elif base.ndim == 3: # (Z,Y,X)
    channel_axis = None
    print(f"[INFO] No explicit channel axis, shape={base.shape}")
else:
    channel_axis = None
    print(f"[INFO] Unexpected number of dimensions: {base.ndim}")

# Assign default colors
default_colors = ["blue", "green", "red", "magenta", "yellow", "cyan", "orange", "white"]
num_channels = base.shape[channel_axis] if channel_axis is not None else 1
channel_colors = default_colors[:num_channels]
print(f"[INFO] Using channel colors: {channel_colors}")

# ========================
# Napari viewer
# ========================
print("[INFO] Launching Napari with multiscale pyramid...")
viewer = napari.Viewer()
viewer.add_image(
    pyramid,
    multiscale=True,
    channel_axis=channel_axis,
    colormap=channel_colors,
    scale=voxel_size,
    name='IMS Image Pyramid',
    contrast_limits=channel_histograms
)
napari.run()
