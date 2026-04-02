
# import threading
# import time
# from dataclasses import dataclass
# from typing import List, Optional, Tuple


# def get_gpu_utilization(gpu_index: int = 0) -> Tuple[Optional[float], Optional[float]]:
#     """
#     Backward-compatible instantaneous snapshot.
#     Returns (gpu_util_percent, gpu_mem_percent) for one GPU index.
#     If NVML isn't available or no GPU -> (None, None)
#     """
#     try:
#         from pynvml import (
#             nvmlInit, nvmlShutdown,
#             nvmlDeviceGetHandleByIndex,
#             nvmlDeviceGetUtilizationRates,
#             nvmlDeviceGetMemoryInfo,
#         )
#         nvmlInit()
#         h = nvmlDeviceGetHandleByIndex(gpu_index)
#         util = nvmlDeviceGetUtilizationRates(h)
#         mem = nvmlDeviceGetMemoryInfo(h)
#         gpu_util = float(util.gpu)
#         gpu_mem = float(mem.used / mem.total * 100.0) if mem.total else None
#         nvmlShutdown()
#         return gpu_util, gpu_mem
#     except Exception:
#         return None, None


# @dataclass
# class QueryGpuStats:
#     avg_gpu_util_percent: Optional[float]
#     max_gpu_util_percent: Optional[float]
#     avg_gpu_mem_percent: Optional[float]
#     max_gpu_mem_percent: Optional[float]
#     avg_gpu_mem_mb: Optional[float]
#     max_gpu_mem_mb: Optional[float]
#     torch_peak_mem_mb: Optional[float]
#     sample_count: int


# class QueryGpuMonitor:
#     """Lightweight per-query GPU monitor.

#     Samples NVML periodically while a single query runs and also records
#     torch peak allocated memory for the selected GPU.
#     """

#     def __init__(self, gpu_index: int = 0, poll_interval_s: float = 0.05):
#         self.gpu_index = int(gpu_index)
#         self.poll_interval_s = float(poll_interval_s)
#         self._stop = threading.Event()
#         self._thread = None
#         self._utils: List[float] = []
#         self._mem_percents: List[float] = []
#         self._mem_mbs: List[float] = []
#         self._started_nvml = False
#         self._handle = None
#         self._torch_peak_supported = False

#     def start(self):
#         try:
#             import torch
#             if torch.cuda.is_available() and self.gpu_index < torch.cuda.device_count():
#                 with torch.cuda.device(self.gpu_index):
#                     torch.cuda.synchronize()
#                     torch.cuda.reset_peak_memory_stats(self.gpu_index)
#                     self._torch_peak_supported = True
#         except Exception:
#             self._torch_peak_supported = False

#         try:
#             from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex
#             nvmlInit()
#             self._started_nvml = True
#             self._handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
#             self._thread = threading.Thread(target=self._run, daemon=True)
#             self._thread.start()
#         except Exception:
#             self._started_nvml = False
#             self._handle = None
#         return self

#     def _run(self):
#         try:
#             from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
#             while not self._stop.is_set():
#                 try:
#                     util = nvmlDeviceGetUtilizationRates(self._handle)
#                     mem = nvmlDeviceGetMemoryInfo(self._handle)
#                     self._utils.append(float(util.gpu))
#                     if mem.total:
#                         self._mem_percents.append(float(mem.used / mem.total * 100.0))
#                     self._mem_mbs.append(float(mem.used / (1024 ** 2)))
#                 except Exception:
#                     pass
#                 time.sleep(self.poll_interval_s)
#         except Exception:
#             return

#     def stop(self) -> QueryGpuStats:
#         self._stop.set()
#         if self._thread is not None:
#             self._thread.join(timeout=max(0.2, 2 * self.poll_interval_s))

#         torch_peak_mb = None
#         try:
#             import torch
#             if self._torch_peak_supported and torch.cuda.is_available() and self.gpu_index < torch.cuda.device_count():
#                 with torch.cuda.device(self.gpu_index):
#                     torch.cuda.synchronize()
#                     torch_peak_mb = float(torch.cuda.max_memory_allocated(self.gpu_index) / (1024 ** 2))
#         except Exception:
#             torch_peak_mb = None

#         if self._started_nvml:
#             try:
#                 from pynvml import nvmlShutdown
#                 nvmlShutdown()
#             except Exception:
#                 pass

#         def _avg(xs):
#             return float(sum(xs) / len(xs)) if xs else None
#         def _max(xs):
#             return float(max(xs)) if xs else None

#         return QueryGpuStats(
#             avg_gpu_util_percent=_avg(self._utils),
#             max_gpu_util_percent=_max(self._utils),
#             avg_gpu_mem_percent=_avg(self._mem_percents),
#             max_gpu_mem_percent=_max(self._mem_percents),
#             avg_gpu_mem_mb=_avg(self._mem_mbs),
#             max_gpu_mem_mb=_max(self._mem_mbs),
#             torch_peak_mem_mb=torch_peak_mb,
#             sample_count=len(self._utils),
#         )
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


def _resolve_nvml_index(torch_gpu_index: int) -> int:
    """
    Map a torch-visible GPU index to the physical NVML GPU index using
    CUDA_VISIBLE_DEVICES when present.

    Example:
      CUDA_VISIBLE_DEVICES=3
      torch index 0 -> physical index 3
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return int(torch_gpu_index)

    parts = [p.strip() for p in visible.split(",") if p.strip()]
    if not parts:
        return int(torch_gpu_index)

    if torch_gpu_index < 0 or torch_gpu_index >= len(parts):
        raise ValueError(
            f"torch gpu index {torch_gpu_index} out of range for "
            f"CUDA_VISIBLE_DEVICES={visible}"
        )

    mapped = parts[torch_gpu_index]
    if mapped.isdigit():
        return int(mapped)

    # If UUID-style values are ever used, fall back to torch index.
    return int(torch_gpu_index)


def get_gpu_utilization(gpu_index: int = 0) -> Tuple[Optional[float], Optional[float]]:
    """
    Backward-compatible instantaneous snapshot.
    Returns (gpu_util_percent, gpu_mem_percent) for one torch-visible GPU index.
    If NVML isn't available or no GPU -> (None, None)
    """
    try:
        from pynvml import (
            nvmlInit,
            nvmlShutdown,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetUtilizationRates,
            nvmlDeviceGetMemoryInfo,
        )
        nvmlInit()
        physical_index = _resolve_nvml_index(int(gpu_index))
        h = nvmlDeviceGetHandleByIndex(physical_index)
        util = nvmlDeviceGetUtilizationRates(h)
        mem = nvmlDeviceGetMemoryInfo(h)
        gpu_util = float(util.gpu)
        gpu_mem = float(mem.used / mem.total * 100.0) if mem.total else None
        nvmlShutdown()
        return gpu_util, gpu_mem
    except Exception:
        return None, None


@dataclass
class QueryGpuStats:
    avg_gpu_util_percent: Optional[float]
    max_gpu_util_percent: Optional[float]
    avg_gpu_mem_percent: Optional[float]
    max_gpu_mem_percent: Optional[float]
    avg_gpu_mem_mb: Optional[float]
    max_gpu_mem_mb: Optional[float]
    torch_peak_mem_mb: Optional[float]
    sample_count: int


class QueryGpuMonitor:
    """Lightweight per-query GPU monitor.

    Samples NVML periodically while a single query runs and also records
    torch peak allocated memory for the selected torch-visible GPU.
    """

    def __init__(self, gpu_index: int = 0, poll_interval_s: float = 0.05):
        self.gpu_index = int(gpu_index)
        self.poll_interval_s = float(poll_interval_s)
        self._stop = threading.Event()
        self._thread = None
        self._utils: List[float] = []
        self._mem_percents: List[float] = []
        self._mem_mbs: List[float] = []
        self._started_nvml = False
        self._handle = None
        self._torch_peak_supported = False
        self._physical_index = None

    def start(self):
        try:
            import torch
            if torch.cuda.is_available() and self.gpu_index < torch.cuda.device_count():
                with torch.cuda.device(self.gpu_index):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(self.gpu_index)
                    self._torch_peak_supported = True
                    print(
                        f"[GPU MONITOR] torch index={self.gpu_index}, "
                        f"torch current device={torch.cuda.current_device()}, "
                        f"torch device count={torch.cuda.device_count()}, "
                        f"torch device name={torch.cuda.get_device_name(self.gpu_index)}, "
                        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
                    )
        except Exception as e:
            print(f"[GPU MONITOR] torch setup failed: {e}")
            self._torch_peak_supported = False

        try:
            from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex
            nvmlInit()
            self._started_nvml = True
            self._physical_index = _resolve_nvml_index(self.gpu_index)
            print(
                f"[GPU MONITOR] nvml physical index={self._physical_index} "
                f"(from torch index={self.gpu_index})"
            )
            self._handle = nvmlDeviceGetHandleByIndex(self._physical_index)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception as e:
            print(f"[GPU MONITOR] NVML init failed: {e}")
            self._started_nvml = False
            self._handle = None

        return self

    def _run(self):
        try:
            from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
            while not self._stop.is_set():
                try:
                    util = nvmlDeviceGetUtilizationRates(self._handle)
                    mem = nvmlDeviceGetMemoryInfo(self._handle)
                    self._utils.append(float(util.gpu))
                    if mem.total:
                        self._mem_percents.append(float(mem.used / mem.total * 100.0))
                    self._mem_mbs.append(float(mem.used / (1024 ** 2)))
                except Exception:
                    pass
                time.sleep(self.poll_interval_s)
        except Exception:
            return

    def stop(self) -> QueryGpuStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, 2 * self.poll_interval_s))

        torch_peak_mb = None
        try:
            import torch
            if (
                self._torch_peak_supported
                and torch.cuda.is_available()
                and self.gpu_index < torch.cuda.device_count()
            ):
                with torch.cuda.device(self.gpu_index):
                    torch.cuda.synchronize()
                    torch_peak_mb = float(
                        torch.cuda.max_memory_allocated(self.gpu_index) / (1024 ** 2)
                    )
        except Exception:
            torch_peak_mb = None

        if self._started_nvml:
            try:
                from pynvml import nvmlShutdown
                nvmlShutdown()
            except Exception:
                pass

        def _avg(xs):
            return float(sum(xs) / len(xs)) if xs else None

        def _max(xs):
            return float(max(xs)) if xs else None

        stats = QueryGpuStats(
            avg_gpu_util_percent=_avg(self._utils),
            max_gpu_util_percent=_max(self._utils),
            avg_gpu_mem_percent=_avg(self._mem_percents),
            max_gpu_mem_percent=_max(self._mem_percents),
            avg_gpu_mem_mb=_avg(self._mem_mbs),
            max_gpu_mem_mb=_max(self._mem_mbs),
            torch_peak_mem_mb=torch_peak_mb,
            sample_count=len(self._utils),
        )

        print(
            f"[GPU MONITOR] samples={stats.sample_count}, "
            f"avg util={stats.avg_gpu_util_percent}, "
            f"max util={stats.max_gpu_util_percent}, "
            f"avg mem%={stats.avg_gpu_mem_percent}, "
            f"max mem%={stats.max_gpu_mem_percent}, "
            f"torch peak MB={stats.torch_peak_mem_mb}"
        )

        return stats