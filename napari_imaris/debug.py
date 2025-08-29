import os
import numpy as np
import napari
import dask as da
from skimage import img_as_float32
from imaris_ims_file_reader import ims

# -----------------------------
# SETTINGS
# -----------------------------

#3D Data sets:
#IMS_FILE = "../images/NMLNP001-AN6.ims" # 1.07 GB
IMS_FILE = "/mnt/h/Test data/MesoSpim/Mouse-Thy1-5x-CNS_fused.ims" # 273GB
#IMS_FILE = "../images/CellDevelopment-time_series3D.ims"

#2D Datasets, currently does not work for 2D time-series files, but these file types are probably not used a lot
#IMS_FILE = "../images/SwimmingAlgae_with_objects-time_series2D.ims" 
#IMS_FILE = "../images/MULTI_IMAGE-2D.ims"

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

def get_channel_metadata(ims_obj, channel=0, time_point=0, resolution_level=0):
    """Return metadata for a channel, including contrast limits"""
    meta = ims_obj.metaData[resolution_level, time_point, channel]
    name = meta.get("ChannelName", f"Channel {channel}")
    scale = meta.get("resolution", (1.0, 1.0, 1.0))
    
    # Compute contrast limits from highest-res level if array is small enough
    if ims_obj.ResolutionLevels <= 2 and not isinstance(ims_obj, da.Array):
        data = ims_obj[resolution_level, time_point, channel, :, :, :]
        data = img_as_float32(data)
        contrast_limits = (float(data.min()), float(data.max()))
    else:
        # For very large Zarr arrays, leave contrast limits as default [0,1]
        contrast_limits = (0.0, 1.0)
    
    return {"name": name, "scale": scale, "contrast_limits": contrast_limits}

def build_multiscale(ims_obj, time_point=0, use_zarr=False):
    """
    Returns a multiscale pyramid: list of arrays (channels, z, y, x)
    Works for both RAM and Zarr-backed IMS files.
    """
    multiscale = []

    for r in range(ims_obj.ResolutionLevels):
        vol_channels = []

        for c in range(ims_obj.Channels):
            key = (r, time_point, c, slice(None), slice(None), slice(None))
            arr = ims_obj[key]

            if use_zarr:
                arr = da.from_array(arr)  # lazy evaluation
            else:
                arr = img_as_float32(arr)

            vol_channels.append(arr)

        if use_zarr:
            vol_channels = da.stack(vol_channels, axis=0)
        else:
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

    multiscale = build_multiscale(ims_obj, time_point=0, use_zarr=use_zarr)

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

    debug_print("All channels added. Launching Napari...")
    napari.run()

if __name__ == "__main__":
    main()