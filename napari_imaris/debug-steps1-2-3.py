import numpy as np
import napari
from napari_plugin_engine import napari_hook_implementation
from imaris_ims_file_reader import ims
import asyncio

# ========================
# Async-compatible Imaris Reader
# ========================

class AsyncImarisDebugReader:
    def __init__(self, path):
        self.path = path
        self.ims = ims(path)  # official ims_reader
        self.shape = self.ims.shape  # (T, C, Z, Y, X)
        self.ndim = len(self.shape)
        self.has_channels = self.shape[1] > 1 if self.ndim >= 5 else False
        print(f"Loaded IMS file: {path}, shape: {self.shape}")

    async def get_chunk_async(self, slices):
        """
        Async-compatible method to return a chunk.
        Currently only returns full 2D slices along Z.
        """
        # Fill slices to 5D: (T, C, Z, Y, X)
        slices_full = list(slices)
        while len(slices_full) < 5:
            slices_full.append(slice(None))
        slices_full = tuple(slices_full)

        # For debug: print requested slices
        print(f"[Async Request] slices: {slices_full}")

        # Get chunk from ims_reader
        chunk = self.ims[slices_full]

        # Move channel axis to last if needed
        if self.has_channels:
            if chunk.ndim == 5:  # (T, C, Z, Y, X)
                chunk = np.moveaxis(chunk, 1, -1)
            elif chunk.ndim == 4:  # (C, Z, Y, X)
                chunk = np.moveaxis(chunk, 0, -1)

        # Return chunk
        print(f"[Async Return] shape: {chunk.shape}")
        return chunk

    def __getitem__(self, slices):
        """
        For Napari async mode, this can wrap the async method.
        """
        return asyncio.run(self.get_chunk_async(slices))


# ========================
# Napari plugin hook
# ========================

@napari_hook_implementation
def napari_get_reader(path):
    if str(path).endswith('.ims'):
        reader_instance = AsyncImarisDebugReader(path)
        def reader_func(path):
            return reader_instance
        return reader_func


# ========================
# Standalone test
# ========================

if __name__ == "__main__":
    viewer = napari.Viewer()
    ims_file = "../images/NMLNP001-AN6.ims"  # replace with your path
    reader = AsyncImarisDebugReader(ims_file)

    # Test async 2D slice: first Z slice
    z_slice = 0
    slice_2d = asyncio.run(reader.get_chunk_async((slice(0, 1), slice(None), slice(None))))
    print(f"Returned slice shape: {slice_2d.shape}")

    viewer.add_image(slice_2d)
    napari.run()
