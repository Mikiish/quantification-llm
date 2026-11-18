"""Prépare le modèle, compare les réponses et produit un rapport en français."""

import argparse
import gc
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
EOS = [151645, 151643]
KV_BYTES = 128 * 2 ** 20
LABELS = "ABCD"
SYSTEM = "Tu réponds en français et respectes le format demandé."
OPEN_QUESTIONS = [
    ("Explique en deux phrases pourquoi le produit de deux nombres impairs est impair.",
     "La justification fait apparaître que le produit a la forme 2k + 1."),
    ("Paul affirme que toute matrice carrée non nulle est inversible. Donne un contre-exemple 2 × 2 et explique-le brièvement.",
     "La réponse donne une matrice non nulle avec un déterminant nul ou des lignes dépendantes."),
]


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def cases():
    closed = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    records = []
    for category, prompt, options, answer in closed:
        choices = "\n".join(f"{label}. {option}" for label, option in zip(LABELS, options, strict=True))
        records.append({
            "category": category, "question": prompt, "expected": answer, "options": options,
            "prompt": f"{prompt}\n{choices}\nRéponds uniquement par la lettre A, B, C ou D.",
            "max_tokens": 1,
        })
    for prompt, rubric in OPEN_QUESTIONS:
        records.append({"category": "Réponse libre", "question": prompt, "prompt": prompt,
                        "rubric": rubric, "max_tokens": 128})
    return records


def choice_tokens(tokenizer):
    choices = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if any(len(choice) != 1 for choice in choices):
        raise ValueError("Chaque lettre du QCM doit correspondre à un seul token.")
    return [choice[0] for choice in choices]


def cpu_responses(original, compressed, records, threads):
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from quantifier import CHECKPOINT, dequantize

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(original, padding_side="left")
    allowed = choice_tokens(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(original, dtype=torch.float16).float().eval()

    def generate():
        answers = []
        start = 0
        while start < len(records):
            end = start + 1
            budget = records[start]["max_tokens"]
            while end < min(start + 8, len(records)) and records[end]["max_tokens"] == budget:
                end += 1
            encoded = tokenizer.apply_chat_template(
                [messages(record["prompt"]) for record in records[start:end]],
                tokenize=True, add_generation_prompt=True, padding=True,
                return_tensors="pt", return_dict=True,
            )
            config = GenerationConfig(
                max_new_tokens=budget, do_sample=False,
                eos_token_id=EOS, pad_token_id=tokenizer.pad_token_id,
            )
            output = model.generate(
                **encoded, generation_config=config,
                prefix_allowed_tokens_fn=(lambda _index, _tokens: allowed) if budget == 1 else None,
            )
            prompt_length = encoded["input_ids"].shape[1]
            answers.extend(tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True))
            start = end
        return answers

    with torch.inference_mode():
        print("Évaluation CPU des poids de référence.", flush=True)
        dense = generate()
        print("Reconstruction depuis le checkpoint INT4 et évaluation CPU.", flush=True)
        with safe_open(compressed / CHECKPOINT, framework="pt") as file:
            for name in file.keys():
                if name.endswith(".qweight"):
                    prefix = name.removesuffix("qweight")
                    layer = model.get_submodule(prefix.rstrip("."))
                    layer.weight.copy_(dequantize(
                        file.get_tensor(name), file.get_tensor(prefix + "qzeros"),
                        file.get_tensor(prefix + "scales"),
                    ))
        quantized = generate()
    del model
    gc.collect()
    return dense, quantized


def server_command(model, variant, port):
    command = [
        "vllm", "serve", str(model), "--served-model-name", variant,
        "--host", "127.0.0.1", "--port", str(port), "--dtype", "half",
        "--max-model-len", "1024", "--max-num-seqs", "8",
        "--kv-cache-memory-bytes", str(KV_BYTES), "--enforce-eager",
        "--attention-backend", "TRITON_ATTN", "--generation-config", "vllm",
        "--no-enable-prefix-caching", "--seed", "0",
    ]
    if variant == "int4":
        command.extend(["--quantization", "awq"])
    return command


def server_environment():
    return {**os.environ, "VLLM_USE_TRITON_AWQ": "1", "VLLM_ROCM_USE_AITER": "0"}


def request(port, route, payload=None, timeout=120):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(f"http://127.0.0.1:{port}{route}", data=body,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return response.read()


def stop_server(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def gpu_responses(model, variant, records, output, torch, allowed):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    def used_vram():
        free, total = torch.cuda.mem_get_info()
        return total - free

    baseline = used_vram()
    log_path = output / f"vllm-{variant}.log"
    print(f"Lancement de vLLM ({variant}). Journal {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            server_command(model, variant, port), env=server_environment(),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"vLLM ({variant}) indisponible. Consulter {log_path}")
                try:
                    request(port, "/health", timeout=2)
                    break
                except (URLError, TimeoutError):
                    time.sleep(1)

            ready = used_vram()
            answers = []
            started = time.perf_counter()
            for record in records:
                payload = {
                    "model": variant, "messages": messages(record["prompt"]),
                    "temperature": 0, "seed": 0, "max_tokens": record["max_tokens"],
                    "stop_token_ids": EOS,
                }
                if "expected" in record:
                    payload["allowed_token_ids"] = allowed
                response = request(port, "/v1/chat/completions", payload)
                answers.append(json.loads(response)["choices"][0]["message"]["content"] or "")
            elapsed = time.perf_counter() - started
            after = used_vram()
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"vLLM ({variant}) HTTP {error.code}. {detail}") from error
        finally:
            stop_server(process)

    log = log_path.read_text(encoding="utf-8", errors="replace")
    weight_log = re.search(r"Model loading (?:took|used)\s+([\d.]+)\s+GiB", log)
    return answers, {
        "baseline_bytes": baseline, "ready_bytes": ready, "after_requests_bytes": after,
        "model_load_gib": float(weight_log[1]) if weight_log else None,
        "evaluation_seconds": elapsed,
    }


def answer_label(text):
    match = re.fullmatch(r"\s*([ABCD])[.)]?\s*", text, flags=re.IGNORECASE | re.ASCII)
    return match[1].upper() if match else None


def write_report(output, stats, records, dense, quantized, metadata):
    def gib(byte_count):
        return f"{byte_count / 2 ** 30:.3f} Gio"

    pairs = []
    for record, left, right in zip(records, dense, quantized, strict=True):
        pair = {**record, "fp16_response": left, "int4_response": right}
        if "expected" in record:
            pair.update(fp16_label=answer_label(left), int4_label=answer_label(right))
        pairs.append(pair)
    closed = [pair for pair in pairs if "expected" in pair]
    scores = {}
    for variant in ("fp16", "int4"):
        scores[variant] = sum(p[variant + "_label"] == p["expected"] for p in closed)
    wins = sum(p["int4_label"] == p["expected"] != p["fp16_label"] for p in closed)
    losses = sum(p["fp16_label"] == p["expected"] != p["int4_label"] for p in closed)
    unchanged = sum(p["int4_label"] == p["fp16_label"] is not None for p in closed)
    bad_format = {v: sum(p[v + "_label"] is None for p in closed) for v in scores}
    reduction = 1 - stats["int4_bytes"] / stats["fp16_bytes"]
    points = 100 * (scores["int4"] - scores["fp16"]) / len(closed)
    lines = [
        "# Quantifier les poids d’un modèle de langage", "",
        "**Question étudiée** — Combien de mémoire gagne-t-on en codant les poids sur "
        "4 bits, et quelles réponses changent ?", "",
        f"Modèle **{stats['model']}**, révision `{stats['revision']}`. "
        f"Exécution **{metadata['backend']}**. Le checkpoint source BF16 est chargé en FP16.", "",
        "## Mémoire et qualité", "",
        "| Mesure | Référence FP16 | Poids INT4 |",
        "| --- | ---: | ---: |",
        f"| Tenseurs stockés, échelles et zéros inclus | {gib(stats['fp16_bytes'])} | {gib(stats['int4_bytes'])} |",
        f"| Checkpoint Safetensors sur disque | {gib(stats['fp16_file_bytes'])} | {gib(stats['int4_file_bytes'])} |",
    ]
    gpu = metadata.get("gpu")
    if gpu:
        for label, key, relative in (
            ("VRAM totale quand le serveur est prêt", "ready_bytes", False),
            ("Hausse de VRAM quand le serveur est prêt", "ready_bytes", True),
            ("Hausse de VRAM après l’évaluation", "after_requests_bytes", True),
        ):
            values = [gib(gpu[v][key] - (gpu[v]["baseline_bytes"] if relative else 0)) for v in scores]
            lines.append(f"| {label} | {values[0]} | {values[1]} |")
        loads = [gpu[v]["model_load_gib"] for v in scores]
        if all(load is not None for load in loads):
            lines.append(f"| VRAM au chargement du modèle, d’après vLLM | {loads[0]:.3f} Gio | {loads[1]:.3f} Gio |")
    lines.extend([
        f"| Bonnes réponses au QCM | {scores['fp16']}/{len(closed)} | {scores['int4']}/{len(closed)} |",
    ])
    for category in dict.fromkeys(p["category"] for p in closed):
        group = [p for p in closed if p["category"] == category]
        counts = [sum(p[v + "_label"] == p["expected"] for p in group) for v in scores]
        lines.append(f"| ↳ {category} | {counts[0]}/{len(group)} | {counts[1]}/{len(group)} |")
    lines.extend([
        f"| Réponses avec un format invalide | {bad_format['fp16']} | {bad_format['int4']} |", "",
        f"Le stockage des tenseurs diminue de **{reduction:.1%}**. Le score du QCM varie de "
        f"**{points:+.1f} points**. La même lettre valide est donnée à {unchanged}/{len(closed)} questions. "
        f"L’INT4 gagne {wins} question(s) et en perd {losses}.", "",
    ])
    if gpu:
        lines.extend([
            f"Carte mesurée **{metadata['device']}**. Le cache KV demandé est fixé à "
            f"{KV_BYTES / 2 ** 20:.0f} Mio pour les deux serveurs. Les mesures HIP concernent "
            "la carte entière et sont relevées avant le démarrage, à la disponibilité "
            "du serveur et après les requêtes. Elles dépendent aussi des autres processus GPU.", "",
        ])
    else:
        lines.extend([
            "Sur CPU, les deux variantes calculent en FP32 avec des poids arrondis en FP16. "
            "Les poids INT4 sont décompressés pour calculer. La table donne la taille du stockage. "
            "Le parcours vLLM/ROCm mesure la consommation de VRAM.", "",
        ])
    lines.extend([
        "## Erreur numérique", "",
        f"Les {len(stats['matrices'])} projections linéaires sont quantifiées par groupes de "
        f"{stats['group_size']} coefficients dans chaque ligne. Pour un groupe, on choisit une "
        "échelle s à partir de l’intervalle des poids (étendu à zéro) et un point zéro entier z. "
        "On stocke q = clip(arrondi(w/s) + z, 0, 15) et on reconstruit ŵ = s(q − z). "
        "Chaque groupe stocke aussi son échelle en FP16 et son point zéro sur 4 bits.", "",
        "Par exemple, pour un groupe dont les bornes sont −1 et 2, on obtient "
        "s ≈ 0,2 et z = 5. Le coefficient 0,67 devient le code entier 8, "
        "qui reconstruit environ 0,60.", "",
        f"En relisant les poids compactés, l’erreur relative de Frobenius globale "
        f"‖Ŵ − W‖ / ‖W‖ vaut **{stats['relative_frobenius']:.2%}**. La racine de l’erreur "
        f"quadratique moyenne par coefficient vaut {stats['rmse']:.5g}, et l’erreur absolue "
        f"maximale vaut {stats['max_error']:.5g}. Le rapport de normes et le score du QCM "
        "mesurent deux effets différents de la même approximation.", "",
        "## Réponses qui diffèrent", "",
    ])
    for p in closed:
        if p["fp16_label"] != p["int4_label"] or p["fp16_label"] is None:
            lines.append(f"- {p['question']} Attendu **{p['expected']}**, FP16 "
                         f"**{p['fp16_label'] or 'invalide'}**, INT4 **{p['int4_label'] or 'invalide'}**.")
    if unchanged == len(closed):
        lines.append("Les deux modèles ont donné la même lettre à chaque question.")
    lines.extend(["", "## Deux réponses à lire", ""])
    for p in pairs[len(closed):]:
        lines.extend([
            f"### {p['question']}", "", f"Critère de lecture — {p['rubric']}", "",
            "**FP16**", "", *("> " + line for line in p["fp16_response"].strip().splitlines()), "",
            "**INT4**", "", *("> " + line for line in p["int4_response"].strip().splitlines()), "",
        ])
    lines.extend([
        f"Le QCM interne de {len(closed)} questions teste surtout le calcul, la logique et la compréhension "
        "de consignes courtes en français. Le modèle choisit la lettre de plus forte probabilité "
        "parmi A, B, C et D. Les réponses libres sont générées avec un budget de "
        "128 tokens et peuvent s’interrompre en cours de phrase. "
        "Le résultat décrit ce modèle, ce réglage et ces questions.", "",
    ])
    (output / "rapport.md").write_text("\n".join(lines), encoding="utf-8")
    results = {"execution": metadata, "scores": scores, "storage_reduction": reduction,
               "gain_int4": wins, "loss_int4": losses, "same_valid_answer": unchanged,
               "invalid_format": bad_format, "weights": stats, "cases": pairs}
    (output / "resultats.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Rapport écrit dans {output / 'rapport.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / ".work")
    parser.add_argument("--threads", type=int, default=8)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preparer", help="Télécharger et quantifier le modèle")
    compare = subparsers.add_parser("comparer", help="Évaluer les deux variantes et écrire le rapport")
    compare.add_argument("--backend", choices=("cpu", "rocm"), default="rocm")
    compare.add_argument("--sortie", type=Path)
    serve = subparsers.add_parser("servir", help="Lancer une variante sur le port local de vLLM")
    serve.add_argument("variant", choices=("fp16", "int4"))
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads doit être strictement positif.")
    work = args.work.resolve()
    original, compressed = work / "original", work / "int4"
    if args.command == "preparer":
        from quantifier import prepare

        stats = prepare(original, compressed, args.threads)
        print(f"INT4 prêt. Erreur relative de Frobenius {stats['relative_frobenius']:.2%}.")
        return

    if not all((location / "model.safetensors").is_file() for location in (original, compressed)):
        parser.error("Préparer les checkpoints avec la commande preparer.")
    if args.command == "servir":
        model = original if args.variant == "fp16" else compressed
        command = server_command(model, args.variant, args.port)
        os.execvpe(command[0], command, server_environment())

    import torch

    records = cases()
    output = (args.sortie or ROOT / "resultats" / args.backend).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stats = json.loads((compressed / "mesures-poids.json").read_text(encoding="utf-8"))
    metadata = {"platform": platform.platform(), "torch": str(torch.__version__)}
    if args.backend == "rocm":
        from transformers import AutoTokenizer

        if sys.platform != "linux" or not torch.version.hip or not torch.cuda.is_available():
            parser.error("Le parcours ROCm exige Linux ou WSL2 et une installation PyTorch ROCm fonctionnelle.")
        device = torch.cuda.get_device_properties(0)
        torch.ones(1, device="cuda").sum().item()
        metadata.update(backend="vLLM / ROCm", device=device.name, arch=device.gcnArchName,
                        hip=torch.version.hip, vllm=version("vllm"), kv_cache_bytes=KV_BYTES)
        allowed = choice_tokens(AutoTokenizer.from_pretrained(original))
        dense, dense_memory = gpu_responses(original, "fp16", records, output, torch, allowed)
        quantized, quant_memory = gpu_responses(compressed, "int4", records, output, torch, allowed)
        metadata["gpu"] = {"fp16": dense_memory, "int4": quant_memory}
    else:
        metadata.update(backend="Transformers / CPU", transformers=version("transformers"))
        dense, quantized = cpu_responses(original, compressed, records, args.threads)
    write_report(output, stats, records, dense, quantized, metadata)


if __name__ == "__main__":
    main()
