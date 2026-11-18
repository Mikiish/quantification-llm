"""Quantification uniforme des projections de Qwen2.5, au format GEMM AWQ."""

import json
import math
import re
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
GROUP = 128
PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
UNPACK_SHIFTS = (0, 16, 4, 20, 8, 24, 12, 28)
PROJECTION = re.compile(
    r"model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight"
)
CHECKPOINT = "model.safetensors"
CONFIG_FILES = (
    "generation_config.json", "merges.txt", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "LICENSE",
)


def pack(codes):
    """Range huit codes de 4 bits dans chaque mot de 32 bits."""
    ordered = codes.reshape(codes.shape[0], -1, 8)[..., PACK_ORDER]
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    return (ordered.to(torch.int64) << shifts).sum(dim=-1).to(torch.int32)


def unpack(words):
    shifts = torch.tensor(UNPACK_SHIFTS, dtype=torch.int32)
    return ((words.unsqueeze(-1) >> shifts) & 15).flatten(1)


def quantize(weights):
    """Une échelle et un zéro pour 128 entrées consécutives d'une ligne."""
    outputs, inputs = weights.shape
    if inputs % GROUP or outputs % 8:
        raise ValueError(f"Dimensions incompatibles avec GEMM AWQ : {weights.shape}")
    groups = weights.float().reshape(outputs, -1, GROUP)
    low = groups.amin(dim=-1).clamp_max(0)
    high = groups.amax(dim=-1).clamp_min(0)
    scales = ((high - low) / 15).clamp_min(2 ** -24).half()
    zeros = (-low / scales.float()).round().clamp(0, 15).to(torch.int32)
    codes = (groups / scales.float().unsqueeze(-1) + zeros.unsqueeze(-1))
    codes = codes.round().clamp(0, 15).to(torch.int32)
    return {
        "qweight": pack(codes.reshape(outputs, inputs).T),
        "qzeros": pack(zeros.T),
        "scales": scales.T.contiguous(),
    }


def dequantize(qweight, qzeros, scales):
    """Relit le format effectivement écrit et reconstruit les poids FP16."""
    codes = unpack(qweight).reshape(scales.shape[0], GROUP, scales.shape[1])
    zeros = unpack(qzeros)
    return ((codes - zeros[:, None, :]) * scales[:, None, :]).reshape(
        qweight.shape[0], -1
    ).T.contiguous()


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def prepare(original: Path, compressed: Path, threads: int):
    from huggingface_hub import snapshot_download

    stats_file = compressed / "mesures-poids.json"
    if stats_file.is_file() and (compressed / CHECKPOINT).is_file():
        if not (original / CHECKPOINT).is_file():
            raise FileNotFoundError(f"Checkpoint de référence absent : {original}")
        return json.loads(stats_file.read_text(encoding="utf-8"))
    if compressed.exists():
        raise FileExistsError(f"Dossier de sortie incomplet : {compressed}")

    snapshot_download(
        MODEL, revision=REVISION, local_dir=original,
        allow_patterns=["config.json", CHECKPOINT, *CONFIG_FILES],
    )
    torch.set_num_threads(threads)
    config = json.loads((original / "config.json").read_text(encoding="utf-8"))
    packed = {}
    dense_bytes = 0
    matrices = []
    for_copy = []
    with safe_open(original / CHECKPOINT, framework="pt") as source:
        for name in source.keys():
            weight = source.get_tensor(name).half()
            dense_bytes += tensor_bytes(weight)
            if not PROJECTION.fullmatch(name):
                packed[name] = weight
                for_copy.append(name)
                continue
            quantized = quantize(weight)
            restored = dequantize(**quantized).float()
            reference = weight.float()
            difference = restored - reference
            matrices.append({
                "name": name,
                "shape": list(weight.shape),
                "parameters": weight.numel(),
                "squared_error": difference.square().sum(dtype=torch.float64).item(),
                "squared_norm": reference.square().sum(dtype=torch.float64).item(),
                "max_error": difference.abs().max().item(),
            })
            for suffix, tensor in quantized.items():
                packed[name.removesuffix("weight") + suffix] = tensor
    if len(matrices) != 7 * config["num_hidden_layers"]:
        raise ValueError("Nombre de projections inattendu pour le checkpoint Qwen.")

    config["torch_dtype"] = "float16"
    config["quantization_config"] = {
        "quant_method": "awq", "version": "gemm", "bits": 4,
        "group_size": GROUP, "zero_point": True,
    }
    error = math.fsum(matrix["squared_error"] for matrix in matrices)
    norm = math.fsum(matrix["squared_norm"] for matrix in matrices)
    stats = {
        "model": MODEL, "revision": REVISION, "bits": 4, "group_size": GROUP,
        "fp16_bytes": dense_bytes,
        "int4_bytes": sum(tensor_bytes(tensor) for tensor in packed.values()),
        "fp16_file_bytes": (original / CHECKPOINT).stat().st_size,
        "relative_frobenius": math.sqrt(error / norm),
        "rmse": math.sqrt(error / sum(m["parameters"] for m in matrices)),
        "max_error": max(m["max_error"] for m in matrices),
        "matrices": matrices,
        "other_tensor_count": len(for_copy),
    }
    print(f"{len(matrices)} matrices quantifiées, écriture du checkpoint INT4.", flush=True)
    with tempfile.TemporaryDirectory(dir=compressed.parent) as staging:
        output = Path(staging) / "modele"
        output.mkdir()
        for filename in CONFIG_FILES:
            shutil.copyfile(original / filename, output / filename)
        (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        save_file(packed, output / CHECKPOINT, metadata={"format": "pt"})
        stats["int4_file_bytes"] = (output / CHECKPOINT).stat().st_size
        (output / stats_file.name).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        output.rename(compressed)
    return stats
