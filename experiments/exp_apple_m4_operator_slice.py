from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "results/apple_m4_operator_slice_results.json"
OUTPUT_MD = ROOT / "results/apple_m4_operator_slice_results.md"


@dataclass
class BenchmarkResult:
    operator: str
    device: str
    shape: str
    warmup_runs: int
    timed_runs: int
    mean_ms: float | None
    std_ms: float | None
    median_ms: float | None
    status: str
    notes: str = ""


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def measure(label: str, shape: str, device: torch.device, fn: Callable[[], None], warmup: int = 20, runs: int = 50) -> BenchmarkResult:
    try:
        for _ in range(warmup):
            fn()
        synchronize(device)

        samples: List[float] = []
        for _ in range(runs):
            synchronize(device)
            start = time.perf_counter()
            fn()
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)

        return BenchmarkResult(
            operator=label,
            device=device.type,
            shape=shape,
            warmup_runs=warmup,
            timed_runs=runs,
            mean_ms=statistics.mean(samples),
            std_ms=statistics.pstdev(samples),
            median_ms=statistics.median(samples),
            status="ok",
        )
    except Exception as exc:
        return BenchmarkResult(
            operator=label,
            device=device.type,
            shape=shape,
            warmup_runs=warmup,
            timed_runs=runs,
            mean_ms=None,
            std_ms=None,
            median_ms=None,
            status="failed",
            notes=str(exc),
        )


def run_for_device(device_name: str) -> List[BenchmarkResult]:
    device = torch.device(device_name)
    results: List[BenchmarkResult] = []

    linear = nn.Linear(1024, 1024).to(device).eval()
    linear_input = torch.randn(32, 1024, device=device)
    results.append(measure("Linear", "(32, 1024)", device, lambda: linear(linear_input)))

    conv = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device).eval()
    conv_input = torch.randn(1, 64, 224, 224, device=device)
    results.append(measure("Conv2d", "(1, 64, 224, 224)", device, lambda: conv(conv_input)))

    layernorm = nn.LayerNorm(768).to(device).eval()
    ln_input = torch.randn(32, 512, 768, device=device)
    results.append(measure("LayerNorm", "(32, 512, 768)", device, lambda: layernorm(ln_input)))

    softmax_input = torch.randn(8, 12, 512, 512, device=device)
    results.append(measure("Softmax", "(8, 12, 512, 512)", device, lambda: torch.softmax(softmax_input, dim=-1)))

    mha = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True).to(device).eval()
    mha_input = torch.randn(4, 512, 768, device=device)
    results.append(measure("MultiheadAttention", "(4, 512, 768)", device, lambda: mha(mha_input, mha_input, mha_input)))

    return results


def main() -> None:
    all_results: Dict[str, object] = {
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "results": [],
    }

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")

    rows: List[BenchmarkResult] = []
    for device_name in devices:
        rows.extend(run_for_device(device_name))

    all_results["results"] = [asdict(row) for row in rows]
    OUTPUT_JSON.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    lines = [
        "# Apple M4 Operator Slice Results",
        "",
        f"- Torch: {torch.__version__}",
        f"- MPS available: {torch.backends.mps.is_available()}",
        "",
        "| Operator | Device | Shape | Mean (ms) | Std (ms) | Median (ms) | Status | Notes |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row.operator} | {row.device} | {row.shape} | "
            f"{'' if row.mean_ms is None else f'{row.mean_ms:.4f}'} | "
            f"{'' if row.std_ms is None else f'{row.std_ms:.4f}'} | "
            f"{'' if row.median_ms is None else f'{row.median_ms:.4f}'} | "
            f"{row.status} | {row.notes.replace('|', '/')} |"
        )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_JSON)
    print(OUTPUT_MD)


if __name__ == "__main__":
    main()