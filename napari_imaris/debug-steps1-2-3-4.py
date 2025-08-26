import numpy as np
import asyncio
from napari_plugin_engine import napari_hook_implementation
from imaris_ims_file_reader import ims

# ========================
# Async multi-resolution Imaris Reader
# ========================

MAX_TEXTURE_SIZE = 2048  # GPU limitation (can be adjusted)

class AsyncImarisMultiResReader:
    def __init__(self, path):
        self.path = path
        self.ims = ims(path)  # official ims_reader
        self.shape = self.ims.shape  # (T, C, Z, Y, X)
        self.ndim = len(self.shape)
        self.has_channels = self.shape[1] > 1 if self.ndim >= 5 else False
        print(f"Loaded IMS file: {path}, shape: {self.shape}")
        print(f"Resolution levels: {self.ims.ResolutionLevels}")

    def choose_resolution_level(self, slices):
        """
        Select resolution level based on requested chunk size
        """
        # Compute requested shape
        slices_full = list(slices)
        while len(slices_full) < 5:
            slices_full.append(slice(None))
        slices_full = tuple(slices_full)

        t_size = len(range(self.shape[0])[slices_full[0]])
        c_size = len(range(self.shape[1])[slices_full[1]])
        z_size = len(range(self.shape[2])[slices_full[2]])
        y_size = len(range(self.shape[3])[slices_full[3]])
        x_size = len(range(self.shape[4])[slices_full[4]])

        req_sizes = np.array([z_size, y_size, x_size])

        # Start at full resolution
        res_level = 0
        for r in range(self.ims.ResolutionLevels):
            level_shape = self.ims.metaData[r,0,0,'shape'][-3:]
            if all(rs <= MAX_TEXTURE_SIZE for rs in np.array(level_shape)):
                res_level = r
                break
            # Otherwise keep next lower resolution

        return res_level

    async def get_chunk_async(self, slices):
        """
        Async-compatible method to return a chunk at appropriate resolution
        """
        # Choose resolution
        res_level = self.choose_resolution_level(slices)
        self.ims.change_resolution_lock(res_level)

        # Fill slices to 5D: (T, C, Z, Y, X)
        slices_full = list(slices)
        while len(slices_full) < 5:
            slices_full.append(slice(None))
        slices_full = tuple(slices_full)

        print(f"[Async Request] slices: {slices_full}, using resolution level: {res_level}")

        # Get chunk from ims_reader
        chunk = self.ims[slices_full]

        # Move channel axis to last if needed
        if self.has_channels:
            if chunk.ndim == 5:  # (T, C, Z, Y, X)
                chunk = np.moveaxis(chunk, 1, -1)
            elif chunk.ndim == 4:  # (C, Z, Y, X)
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
        reader_instance = AsyncImarisMultiResReader(path)
        def reader_func(path):
            return reader_instance
        return reader_func


# ========================
# Standalone test
# ========================

if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    ims_file = "/home/brandon/workspace/local/brandon/napari-imaris/images/BP003_2-3.ims"  # replace with your path
    reader = AsyncImarisMultiResReader(ims_file)

    # Test: first Z slice
    z_slice = 0
    slice_2d = asyncio.run(reader.get_chunk_async((slice(0, 1), slice(None), slice(z_slice, z_slice+1))))
    print(f"Returned slice shape: {slice_2d.shape}")

    viewer.add_image(slice_2d)
    napari.run()
