from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List

import torch
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model
from torchvision.models import resnet50, vit_b_16


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "results/apple_m4_model_slice_results.json"
OUTPUT_MD = ROOT / "results/apple_m4_model_slice_results.md"


@dataclass
class ModelBenchmarkResult:
    model: str
    device: str
    input_desc: str
    warmup_runs: int
    timed_runs: int
    mean_ms: float | None
    std_ms: float | None
    median_ms: float | None
    status: str
    notes: str = ""


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def measure(model_name: str, device: torch.device, input_desc: str, fn: Callable[[], None], warmup: int = 10, runs: int = 20) -> ModelBenchmarkResult:
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                fn()
            sync(device)

            samples: List[float] = []
            for _ in range(runs):
                sync(device)
                start = time.perf_counter()
                fn()
                sync(device)
                samples.append((time.perf_counter() - start) * 1000.0)

        return ModelBenchmarkResult(
            model=model_name,
            device=device.type,
            input_desc=input_desc,
            warmup_runs=warmup,
            timed_runs=runs,
            mean_ms=statistics.mean(samples),
            std_ms=statistics.pstdev(samples),
            median_ms=statistics.median(samples),
            status="ok",
        )
    except Exception as exc:
        return ModelBenchmarkResult(
            model=model_name,
            device=device.type,
            input_desc=input_desc,
            warmup_runs=warmup,
            timed_runs=runs,
            mean_ms=None,
            std_ms=None,
            median_ms=None,
            status="failed",
            notes=str(exc),
        )


def build_tasks(device: torch.device):
    resnet = resnet50(weights=None).to(device).eval()
    resnet_x = torch.randn(1, 3, 224, 224, device=device)

    vit = vit_b_16(weights=None).to(device).eval()
    vit_x = torch.randn(1, 3, 224, 224, device=device)

    bert = BertModel(BertConfig()).to(device).eval()
    bert_ids = torch.randint(0, 1000, (1, 512), device=device)

    gpt2 = GPT2Model(GPT2Config()).to(device).eval()
    gpt2_ids = torch.randint(0, 1000, (1, 512), device=device)

    return [
        ("ResNet50", "(1, 3, 224, 224)", lambda: resnet(resnet_x)),
        ("ViT-Base", "(1, 3, 224, 224)", lambda: vit(vit_x)),
        ("BERT-Base", "(1, 512)", lambda: bert(input_ids=bert_ids).last_hidden_state),
        ("GPT2-Small", "(1, 512)", lambda: gpt2(input_ids=gpt2_ids).last_hidden_state),
    ]


def main() -> None:
    devices = ["mps"] if torch.backends.mps.is_available() else ["cpu"]
    results: List[ModelBenchmarkResult] = []

    for device_name in devices:
        device = torch.device(device_name)
        for model_name, input_desc, fn in build_tasks(device):
            results.append(measure(model_name, device, input_desc, fn))

    payload = {
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "results": [asdict(result) for result in results],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Apple M4 Model Slice Results",
        "",
        f"- Torch: {torch.__version__}",
        f"- MPS available: {torch.backends.mps.is_available()}",
        "- Protocol: 10 warm-up runs, 20 timed runs, random-weight models, inference mode.",
        "",
        "| Model | Device | Input | Mean (ms) | Std (ms) | Median (ms) | Status | Notes |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]

    for result in results:
        lines.append(
            f"| {result.model} | {result.device} | {result.input_desc} | "
            f"{'' if result.mean_ms is None else f'{result.mean_ms:.4f}'} | "
            f"{'' if result.std_ms is None else f'{result.std_ms:.4f}'} | "
            f"{'' if result.median_ms is None else f'{result.median_ms:.4f}'} | "
            f"{result.status} | {result.notes.replace('|', '/')} |"
        )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_JSON)
    print(OUTPUT_MD)


if __name__ == "__main__":
    main()