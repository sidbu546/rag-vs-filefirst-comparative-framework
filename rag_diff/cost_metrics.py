# cost_metrics.py
from typing import Optional

def usd_cost_from_time(
    gpu_seconds: float,
    cpu_seconds: float = 0.0,
    gpu_cost_per_hour: float = 0.0,
    cpu_cost_per_hour: float = 0.0,
    fixed_cost_usd: float = 0.0,
) -> float:
    """
    Simple deployment cost estimator.
    - gpu_seconds: time the model used GPU (use llm_latency_s)
    - cpu_seconds: retrieval/overhead time (use latency_s - llm_latency_s)
    - *_cost_per_hour: your assumed $/hour rates (defaults 0 to avoid guessing)
    - fixed_cost_usd: any flat fee per run (optional)
    """
    gpu_cost = (gpu_seconds / 3600.0) * gpu_cost_per_hour
    cpu_cost = (cpu_seconds / 3600.0) * cpu_cost_per_hour
    return float(gpu_cost + cpu_cost + fixed_cost_usd)