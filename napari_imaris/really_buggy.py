import os
import numpy as np
import napari
from imaris_ims_file_reader import ims, ims_zarr_store

class LazyIMSLazyArray:
    """Lazy wrapper for Napari multiscale from ims_zarr_store."""
    def __init__(self, ims_obj, resolution=0, time_point=0, channel=0):
        self.ims_obj = ims_obj
        self.resolution = resolution
        self.time_point = time_point
        self.channel = channel

        md = ims_obj.ims.metaData
        self.shape = md[resolution, time_point, channel, 'shape'][-3:]  # Z,Y,X
        self.ndim = len(self.shape)
        self.dtype = ims_obj.dtype
        self.size = np.prod(self.shape)

    def __getitem__(self, idx):
        # Ensure tuple of length 3
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = tuple(idx) + (slice(None),) * (3 - len(idx))
        z, y, x = idx

        def fix_slice(s, dim_len):
            if isinstance(s, slice):
                start = 0 if s.start is None else s.start
                stop = dim_len if s.stop is None else s.stop
                step = 1 if s.step is None else s.step
                return range(start, stop, step)
            else:
                return [s]

        z_range = fix_slice(z, self.shape[0])
        y_range = fix_slice(y, self.shape[1])
        x_range = fix_slice(x, self.shape[2])

        out = np.zeros((len(z_range), len(y_range), len(x_range)), dtype=self.dtype)

        for idx_z, z_layer in enumerate(z_range):
            out[idx_z, :, :] = self.ims_obj[
                self.resolution,
                self.time_point,
                self.channel,
                z_layer,
                slice(y_range[0], y_range[-1]+1),
                slice(x_range[0], x_range[-1]+1)
            ]

        return np.squeeze(out)

def load_ims_multiscale(path, time_point=0, channel=0, verbose=False):
    ims_obj = ims(path, aszarr=True, verbose=verbose)
    multiscale = [
        LazyIMSLazyArray(ims_obj, resolution=r, time_point=time_point, channel=channel)
        for r in range(ims_obj.ResolutionLevels)
    ]
    return multiscale

if __name__ == "__main__":
    ims_path = "/mnt/h/Test data/MesoSpim/Mouse-Thy1-5x-CNS_fused.ims"
    viewer = napari.Viewer()
    multiscale_data = load_ims_multiscale(ims_path, time_point=0, channel=0, verbose=True)
    viewer.add_image(multiscale_data, multiscale=True, name="IMS Multiscale")
    napari.run()

