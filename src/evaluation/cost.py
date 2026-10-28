"""Measured cost on the test hardware (Part 12): model size, inference
latency (CUDA events on GPU, perf_counter on CPU), peak VRAM."""
import copy
import time

import numpy as np
import torch


def model_size(model):
    params = sum(p.numel() for p in model.parameters())
    nbytes = sum(t.numel() * t.element_size()
                 for t in list(model.parameters()) + list(model.buffers()))
    return {"params": int(params), "size_MB": nbytes / 2**20}


@torch.no_grad()
def measure_latency(model, X, batch_sizes=(1, 256), iters=50, warmup=10):
    """Inference latency per batch and per sample on each available device."""
    out = {"cpu_threads": torch.get_num_threads(), "iters": iters}
    targets = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    for target in targets:
        m = copy.deepcopy(model).to(target).eval()
        for b in batch_sizes:
            xb = torch.from_numpy(np.resize(X, (b, X.shape[1])).astype(np.float32)).to(target)
            for _ in range(warmup):
                m(xb)
            if target == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                for _ in range(iters):
                    m(xb)
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end) / iters
            else:
                t0 = time.perf_counter()
                for _ in range(iters):
                    m(xb)
                ms = (time.perf_counter() - t0) * 1000 / iters
            out[f"{target}_bs{b}"] = {"batch_ms": ms, "per_sample_ms": ms / b}
    return out


def peak_vram_MB():
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
