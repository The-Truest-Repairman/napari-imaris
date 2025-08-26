import numpy as np
import napari
from napari_plugin_engine import napari_hook_implementation
from imaris_ims_file_reader import ims

# ========================
# Imaris Reader for Napari
# ========================

class NapariImarisDebugReader:
    def __init__(self, path):
        self.path = path
        
        # Use the official ims_reader
        self.ims = ims(path)
        self.shape = self.ims.shape  # (T, C, Z, Y, X)
        self.ndim = len(self.shape)
        print(f"Loaded IMS file: {path}")
        print(f"Shape: {self.shape}")

        # Determine if channels exist
        self.has_channels = self.shape[1] > 1 if self.ndim >= 5 else False

    def get_chunk(self, slices):
        """
        Return the requested chunk.
        Napari will provide slices like (Z, Y, X) for 2D view or full 3D chunks.
        We'll convert them to the full 5D slice format required by ims_reader: (T, C, Z, Y, X)
        """

        # Fill slices to 5D: (T, C, Z, Y, X)
        slices_full = list(slices)
        while len(slices_full) < 5:
            slices_full.append(slice(None))
        slices_full = tuple(slices_full)

        # Convert to IMS-style indexing: (T, C, Z, Y, X)
        chunk = self.ims[slices_full]

        # Move channel to last axis if needed
        if self.has_channels:
            if chunk.ndim == 5:  # (T, C, Z, Y, X)
                chunk = np.moveaxis(chunk, 1, -1)
            elif chunk.ndim == 4:  # (C, Z, Y, X)
                chunk = np.moveaxis(chunk, 0, -1)

        print(f"Returning chunk shape: {chunk.shape} | slices: {slices_full}")
        return chunk

# ========================
# Napari plugin hook
# ========================

@napari_hook_implementation
def napari_get_reader(path):
    if str(path).endswith('.ims'):
        reader_instance = NapariImarisDebugReader(path)
        def reader_func(path):
            return reader_instance
        return reader_func

# ========================
# Standalone test
# ========================

if __name__ == "__main__":
    viewer = napari.Viewer()
    ims_file = "/home/brandon/workspace/local/brandon/napari-imaris/images/BP003_2-3.ims"  # Replace with your file path
    reader = NapariImarisDebugReader(ims_file)

    # Test 2D slice: first Z slice
    z_slice = 0
    slice_2d = reader.get_chunk((slice(z_slice, z_slice+1), slice(None), slice(None)))
    print(f"2D slice shape: {slice_2d.shape}")

    viewer.add_image(slice_2d)
    napari.run()
