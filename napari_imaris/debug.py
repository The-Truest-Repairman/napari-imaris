import os
import numpy as np
import napari
from skimage import img_as_float32
from imaris_ims_file_reader import ims

# -----------------------------
# SETTINGS
# -----------------------------
IMS_FILE = "../images/NMLNP001-AN6.ims"
USE_ZARR_THRESHOLD_GB = 2
DEBUG = True

# -----------------------------
# HELPERS
# -----------------------------
def get_file_size_gb(file_path):
    return os.path.getsize(file_path) / 1e9

def debug_print(*args):
    if DEBUG:
        print(*args)

def get_channel_metadata(ims_obj, channel=0):
    """Return metadata for a channel, including realistic contrast limits"""
    try:
        meta = ims_obj.metaData[0, 0, channel]
        name = meta.get("ChannelName", f"Channel {channel}")
        scale = meta.get("Resolution", (1.0, 1.0, 1.0))
    except Exception:
        name = f"Channel {channel}"
        scale = (1.0, 1.0, 1.0)

    # Compute contrast limits from actual data in highest-res level
    data = ims_obj[0, 0, channel, :, :, :]
    data = img_as_float32(data)  # convert to float for Napari
    contrast_limits = (float(data.min()), float(data.max()))

    return {"name": name, "scale": scale, "contrast_limits": contrast_limits}

def build_multiscale(ims_obj, time_point=0):
    """Returns list of arrays (multiscale pyramid) with shape (channels, z, y, x)"""
    multiscale = []
    for r in range(ims_obj.ResolutionLevels):
        shape = ims_obj.metaData[r, time_point, 0, "shape"]
        # Handle 5-element shape (1, Z, Y, X) or 6-element shape
        if len(shape) >= 5:
            z, y, x = shape[-3:]
        else:
            raise ValueError(f"Unexpected shape: {shape}")
        debug_print(f"Resolution level {r}, shape={shape}")

        vol_channels = []
        for c in range(ims_obj.Channels):
            vol = ims_obj[r, time_point, c, :, :, :]
            vol_channels.append(img_as_float32(vol))
        vol_channels = np.stack(vol_channels, axis=0)
        multiscale.append(vol_channels)
    return multiscale

# -----------------------------
# MAIN SCRIPT
# -----------------------------
def main():
    file_size = get_file_size_gb(IMS_FILE)
    debug_print(f"IMS file size: {file_size:.2f} GB")
    use_zarr = file_size > USE_ZARR_THRESHOLD_GB
    debug_print("Using Zarr store" if use_zarr else "Loading into RAM directly")

    ims_obj = ims(IMS_FILE, aszarr=use_zarr, verbose=True)
    debug_print(f"Loaded file: {IMS_FILE}")
    debug_print(f"TimePoints: {ims_obj.TimePoints}, Channels: {ims_obj.Channels}, ResolutionLevels: {ims_obj.ResolutionLevels}")

    base_name = os.path.splitext(os.path.basename(IMS_FILE))[0]

    viewer = napari.Viewer()
    multiscale = build_multiscale(ims_obj, time_point=0)

    # Per-channel metadata
    channel_names = []
    contrast_limits_list = []
    for c in range(ims_obj.Channels):
        meta = get_channel_metadata(ims_obj, c)
        channel_names.append(meta["name"])
        contrast_limits_list.append(meta["contrast_limits"])
        debug_print(f"Processed channel: {meta['name']} with contrast limits: {meta['contrast_limits']}")

    # Add Napari image layer with channel_axis
    img_layer = viewer.add_image(
        multiscale,
        name=base_name,
        channel_axis=0,
        contrast_limits=contrast_limits_list,
        multiscale=True,
        visible=True,
    )

    # Store channel names in metadata dictionary
    #img_layer.metadata = {"channel_names": channel_names}

    debug_print("All channels added. Launching Napari...")
    napari.run()

if __name__ == "__main__":
    main()
