import numpy as np
import asyncio
from napari_plugin_engine import napari_hook_implementation
from imaris_ims_file_reader import ims

MAX_TEXTURE_SIZE = 2048  # Maximum GPU texture size

class AsyncImarisDynamicZoomReader:
    def __init__(self, path):
        self.path = path
        self.ims = ims(path)
        self.shape = self.ims.shape  # (T, C, Z, Y, X)
        self.ndim = len(self.shape)
        self.has_channels = self.shape[1] > 1 if self.ndim >= 5 else False
        print(f"Loaded IMS file: {path}, shape: {self.shape}")
        print(f"Resolution levels: {self.ims.ResolutionLevels}")

    def get_zoom_factor(self, viewer, layer_name=None):
        zoom_factor = 1.0
        if viewer is not None:
            try:
                if layer_name:
                    layer = viewer.layers[layer_name]
                else:
                    layer = next(l for l in viewer.layers if l.name and l.name.startswith("image"))
                scale = np.array(layer.scale[-3:])  # Z,Y,X
                camera_scale = np.ones(3)
                if hasattr(viewer.camera, 'scale'):
                    camera_scale = np.array(viewer.camera.scale[-3:])
                zoom_factor = np.min(scale / camera_scale)
            except Exception as e:
                print(f"[Zoom Detection] Could not compute zoom factor: {e}")
        return zoom_factor

    def choose_resolution_level(self, slices, zoom_factor=1.0):
        slices_full = list(slices)
        while len(slices_full) < 5:
            slices_full.append(slice(None))
        slices_full = tuple(slices_full)

        res_level = 0
        # Iterate over resolution levels
        for r in range(self.ims.ResolutionLevels):
            level_shape = np.array(self.ims.metaData[r, 0, 0, 'shape'][-3:])  # Z,Y,X
            apparent_shape = level_shape / zoom_factor  # Adjust for zoom
            if np.all(apparent_shape <= MAX_TEXTURE_SIZE):
                res_level = r
                break
        return res_level

    async def get_chunk_async(self, slices, viewer=None, layer_name=None):
        zoom_factor = self.get_zoom_factor(viewer, layer_name)
        res_level = self.choose_resolution_level(slices, zoom_factor)
        self.ims.change_resolution_lock(res_level)

        slices_full = list(slices)
        while len(slices_full) < 5:
            slices_full.append(slice(None))
        slices_full = tuple(slices_full)

        print(f"[Async Request] slices: {slices_full}, zoom_factor: {zoom_factor:.3f}, resolution level: {res_level}")

        chunk = self.ims[slices_full]

        # Move channel to last axis if present
        if self.has_channels:
            if chunk.ndim == 5:  # T,C,Z,Y,X
                chunk = np.moveaxis(chunk, 1, -1)
            elif chunk.ndim == 4:  # C,Z,Y,X
                chunk = np.moveaxis(chunk, 0, -1)

        print(f"[Async Return] chunk shape: {chunk.shape}")
        return chunk

    def __getitem__(self, slices):
        return asyncio.run(self.get_chunk_async(slices))

# ========================
# Napari plugin hook
# ========================

@napari_hook_implementation
def napari_get_reader(path):
    if str(path).endswith('.ims'):
        reader_instance = AsyncImarisDynamicZoomReader(path)
        def reader_func(path):
            return reader_instance
        return reader_func

# ========================
# Standalone 3D test
# ========================

if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    ims_file = "../images/NMLNP001-AN6.ims"  # replace with your IMS path
    reader = AsyncImarisDynamicZoomReader(ims_file)

    # Full 3D volume for first timepoint and all channels
    t_slice = 0
    slices_3d = (slice(t_slice, t_slice+1), slice(None),
                 slice(None), slice(None), slice(None))  # T,C,Z,Y,X

    # Fetch asynchronously with dynamic zoom
    volume_3d = asyncio.run(reader.get_chunk_async(slices_3d, viewer=viewer))
    print(f"Returned 3D volume shape: {volume_3d.shape}")

    # Add to Napari
    viewer.add_image(volume_3d, name="3D_volume")
    napari.run()
