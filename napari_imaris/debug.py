# debug.py
# Visible-slices-only lazy loading for Imaris .ims into Napari (0.6.4+)
# Uses Dask fromfunction to build one dask array per IMS resolution level.
# Each block fetch logs (res level + zyx window) when Napari requests it.

import math
import numpy as np
import dask.array as da
import napari

# imaris-ims-file-reader 0.1.8 (conda-forge) API:
# The package exposes a top-level factory `ims(...)` that returns an ims_reader-like object.
from imaris_ims_file_reader import ims as ims_open
from skimage.transform import resize


# --------- Config ---------
IMS_PATH = "../images/NMLNP001-AN6.ims"   # your requested path
CHANNEL = 0                               # which channel to display
MAX_LEVELS = None                         # None = use all levels available in the file
CHUNK_YX = 512                            # target Y/X chunk size (napari+dask friendly)
CHUNK_Z = 1                               # target Z chunk size (per-plane)
CHUNKS = (1, 512, 512)
# --------------------------

def downsample_level(arr, factors):
    """Downsample a numpy array by integer factors, keeping minimum size 1."""
    z, y, x = arr.shape
    new_shape = (
        max(1, int(z / factors[0])),
        max(1, int(y / factors[1])),
        max(1, int(x / factors[2]))
    )
    return resize(
        arr,
        new_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True
    ).astype(arr.dtype)

def _level_shape(reader, res_level, t=0, c=0):
    """Return (Z, Y, X) shape for a given resolution/time/channel from ims metadata."""
    # In the published ims_reader (and 0.1.8), metaData[(r,t,c,'shape')] is (1,1,Z,Y,X)
    sh = reader.metaData[(res_level, t, c, 'shape')]
    return int(sh[-3]), int(sh[-2]), int(sh[-1])

def _make_level_array(reader, res_level, channel, chunks):
    """
    Build a Dask array for a specific resolution level that lazily fetches
    only requested (z, y, x) blocks from the .ims file.
    """
    z, y, x = _level_shape(reader, res_level, t=0, c=channel)

    dtype = np.uint16

    def loader(*args, block_info=None):
        if block_info is None:
            # Probe call
            return np.zeros((1, 1, 1), dtype=dtype)

        bi = block_info[None]
        (z0, z1), (y0, y1), (x0, x1) = bi["array-location"]

        # Clamp to valid ranges
        z1 = min(z1, level_shape[0])
        y1 = min(y1, level_shape[1])
        x1 = min(x1, level_shape[2])

        arr = reader[
            res_level, 0, channel,
            slice(z0, z1), slice(y0, y1), slice(x0, x1)
        ]
        arr = np.asarray(arr, dtype=dtype)

        # Expected block shape from block_info
        expected_shape = tuple(bi["chunk-shape"])

        # Pad if needed (e.g. at edges)
        pad_width = [(0, es - s) for s, es in zip(arr.shape, expected_shape)]
        if any(p[1] > 0 for p in pad_width):
            arr = np.pad(arr, pad_width, mode="constant")

        return arr

    arr = da.fromfunction(
        loader,
        shape=(z, y, x),
        chunks=chunks,
        dtype=dtype,
        meta=np.empty((0, 0, 0), dtype=dtype),
    )
    return arr



def _pick_chunks(y, x, z, target_yx=CHUNK_YX, target_z=CHUNK_Z):
    """Choose chunk sizes <= dims, trying to keep ~target tile size."""
    cz = max(1, min(target_z, z))
    cy = max(32, min(target_yx, y))   # keep at least 32 to avoid tiny tiles
    cx = max(32, min(target_yx, x))
    return (cz, cy, cx)

def _make_multiscale(reader, channel, max_levels=6):
    """Build multiscale pyramid using consistent downsampling."""
    multiscale = []

    # base level from IMS (highest resolution)
    base = _make_level_array(reader, 0, channel, CHUNKS)
    base_np = base.compute()  # materialize level 0
    multiscale.append(da.from_array(base_np, chunks=CHUNKS))
    print(f"Level 0: shape={base_np.shape}, chunks={CHUNKS}")

    # build downsampled levels
    current = base_np
    for lvl in range(1, max_levels):
        factors = (1, 2, 2)  # only downsample Y and X
        current = downsample_level(current, factors)
        arr = da.from_array(current, chunks=(1, min(512, current.shape[1]), min(512, current.shape[2])))
        multiscale.append(arr)
        print(f"Level {lvl}: shape={current.shape}, chunks={arr.chunksize}")

    return multiscale

def main():
    # Open the IMS with squeeze_output=False so slicing returns exact shapes
    reader = ims_open(
        IMS_PATH,
        squeeze_output=False,
        verbose=True
    )
    print(f"Loaded IMS file: {IMS_PATH}, shape: {reader.shape}")

    # Optional: print how many resolution levels the file reports
    try:
        print(f"Resolution levels reported by file: {reader.ResolutionLevels}")
    except Exception:
        pass

    # Build multiscale dask arrays for the selected channel
    multiscale = _make_multiscale(reader, CHANNEL, max_levels=MAX_LEVELS)

    # Napari viewer in 3D
    viewer = napari.Viewer(ndisplay=3)

    # Contrast limits: try histogram metadata, otherwise automatic
    clim = None
    try:
        # Typically stored at (res, t, c, 'HistogramMin/Max'); use res=0 (full res), t=0
        hmin = reader.metaData[(0, 0, CHANNEL, 'HistogramMin')]
        hmax = reader.metaData[(0, 0, CHANNEL, 'HistogramMax')]
        clim = (float(hmin), float(hmax))
    except Exception:
        pass

    layer = viewer.add_image(
        multiscale,
        name=f"IMS Channel {CHANNEL}",
        multiscale=True,
        contrast_limits=clim,
        rendering='mip',   # try 'mip' or 'translucent' for volumes; change as needed
        depiction='volume' # napari 0.6.4 volume rendering toggle
    )

    # Helpful debug: print current level that napari says is in use, on camera changes
    @viewer.camera.events.zoom.connect
    def _on_zoom_change(event):
        # Napari determines which level to draw internally, but we can infer:
        # Effective size in screen space scales down with zoom; we just log zoom.
        print(f"[zoom] Camera zoom -> {viewer.camera.zoom:.3f}")

    print("Viewer ready. Interact with the volume; "
          "you should see [load] lines only for visible chunks.")
    napari.run()

    # Keep a reference so IMS file is not GC-closed prematurely.
    # If you want to close explicitly after napari exits:
    try:
        reader.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
