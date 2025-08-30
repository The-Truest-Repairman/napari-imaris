import dask.array as da
import zarr
import h5py
import numpy as np
import cupy as cp
import time
import pynvml
import csv
import threading

# --------------------------
# GPU monitoring
# --------------------------
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def gpu_status():
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_gb = mem.used / (1024**3)
    total_gb = mem.total / (1024**3)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return round(used_gb,2), round(total_gb,2), util.gpu, util.memory  # GB, GB, %, %

def live_monitor(stop_event):
    while not stop_event.is_set():
        used, total, gpu_util, mem_util = gpu_status()
        print(f"\rGPU Memory: {used}/{total} GB | GPU Util: {gpu_util}% | Mem Util: {mem_util}%", end="")
        time.sleep(1)
    print()  # newline after stop

def run_with_live_monitor(func):
    stop_event = threading.Event()
    thread = threading.Thread(target=live_monitor, args=(stop_event,))
    thread.start()
    start = time.time()
    result = func()
    elapsed = time.time() - start
    stop_event.set()
    thread.join()
    used, total, gpu_util, mem_util = gpu_status()
    return result, elapsed, used, gpu_util, mem_util

# --------------------------
# Dataset (10GB scaled)
# --------------------------
shape = (240, 4000, 1500)  # Z,Y,X
dtype = np.uint16
chunk_strategies = [
    (10, 1024, 512),
    (20, 512, 256),
]

zarr_compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
hdf5_compression = "gzip"
hdf5_compression_opts = 4

# Lazy Dask array (CPU)
data = da.random.randint(0, 65535, size=shape, dtype=dtype, chunks=chunk_strategies[0])

# --------------------------
# Benchmark
# --------------------------
def benchmark(strategy, results):
    print(f"\n--- Benchmark chunks {strategy} ---")
    arr = data.rechunk(strategy)

    # --- Zarr (CPU write) ---
    zarr_store = f"microscopy_{strategy}.zarr"
    arr_cpu = arr
    _, write_time, mem, gpu_util, mem_util = run_with_live_monitor(
        lambda: arr_cpu.to_zarr(zarr_store, overwrite=True, compressor=zarr_compressor)
    )

    # Load into GPU for compute
    dask_gpu = da.from_zarr(zarr_store, chunks=strategy).map_blocks(cp.asarray)

    _, mean_time, mem, gpu_util, mem_util = run_with_live_monitor(lambda: dask_gpu.mean().compute())
    mean_val = dask_gpu.mean().compute()
    _, max_time, mem, gpu_util, mem_util = run_with_live_monitor(lambda: dask_gpu.max(axis=0).compute())

    results.append([
        "Zarr", strategy, write_time, mean_time, float(mean_val), max_time, mem, gpu_util, mem_util
    ])

    # --- HDF5 (CPU write) ---
    hdf5_store = f"microscopy_{strategy}.h5"
    def write_hdf5():
        with h5py.File(hdf5_store, "w") as f:
            ds = f.create_dataset(
                "data", shape=shape, chunks=strategy, dtype=dtype,
                compression=hdf5_compression, compression_opts=hdf5_compression_opts
            )
            for z in range(0, shape[0], strategy[0]):
                ds[z:z+strategy[0], :, :] = arr[z:z+strategy[0], :, :].compute()
    _, write_time, mem, gpu_util, mem_util = run_with_live_monitor(write_hdf5)

    f = h5py.File(hdf5_store, "r")
    dask_gpu = da.from_array(f["data"], chunks=strategy).map_blocks(cp.asarray)

    _, mean_time, mem, gpu_util, mem_util = run_with_live_monitor(lambda: dask_gpu.mean().compute())
    mean_val = dask_gpu.mean().compute()
    _, max_time, mem, gpu_util, mem_util = run_with_live_monitor(lambda: dask_gpu.max(axis=0).compute())
    f.close()

    results.append([
        "HDF5", strategy, write_time, mean_time, float(mean_val), max_time, mem, gpu_util, mem_util
    ])

# --------------------------
# Run benchmarks
# --------------------------
results = [["Backend","Chunks","WriteTime(s)","MeanTime(s)","MeanValue",
            "MaxTime(s)","GPU_Memory(GB)","GPU_Util(%)","GPU_Mem_Util(%)"]]

for chunks in chunk_strategies:
    benchmark(chunks, results)

# Print & save
print("\n=== Benchmark Results ===")
for row in results:
    print(row)

with open("gpu_benchmark_10GB_live.csv","w",newline="") as f:
    csv.writer(f).writerows(results)
print("\nResults saved to gpu_benchmark_10GB_live.csv")
