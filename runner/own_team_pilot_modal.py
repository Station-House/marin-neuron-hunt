"""Maintainer entry point for the own-team leniency pilot on the project H200."""

import hashlib
import json
import platform
import re
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import modal

from runner.modal_app import POLE_STATS, VOL, image, vol
from runner.own_team_pilot import (ANSWER_IDS, LAYERS, NEURONS, REQUEST_ID, analyze, answer_scores,
                                   encode_rows, frozen_prompts, json_text, jsonl_text, package_files, sha256)
from runner.prompts import MODEL_REPO, MODEL_SHA

app = modal.App("marin-own-team-leniency-pilot")


def verify_checkpoint(path, checkpoint):
    if (checkpoint["repo"], checkpoint["revision"]) != (MODEL_REPO, MODEL_SHA):
        raise ValueError("wrong checkpoint revision in manifest")
    extra = {file.name for file in path.iterdir()} - checkpoint["files"].keys()
    if extra - {"README.md", ".gitattributes", ".cache"}:
        raise ValueError("checkpoint directory has files outside the pinned manifest")
    for name, digest in checkpoint["files"].items():
        file = path / name
        if "sha256" in digest:
            actual, expected = sha256(file), digest["sha256"]
        else:
            data = file.read_bytes()
            actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            expected = digest["git_blob_sha1"]
        if actual != expected:
            raise ValueError(f"checkpoint file does not match the pinned revision: {name}")


def fixture_checks(m):
    import numpy as np
    import torch

    from runner.model import Row
    from runner.own_team_pilot import parity_error

    fixtures = [
        {"case_id": "fixture-1", "marked": "The jar has a sealed lid. Ruling[[:]]", "waive_label": "A"},
        {"case_id": "fixture-2", "marked": "A red notebook lies near an unused pencil. Ruling[[:]]",
         "waive_label": "B"},
    ]
    encoded = encode_rows(m.tok, fixtures)
    rows = [Row(row["token_ids"], row["cue_index"]) for row in encoded]
    modules = [layer.mlp.down_proj for layer in m.layers]
    hook_counts = [len(module._forward_pre_hooks) for module in modules]
    dense, hooked, _ = m.run(rows, tokens=ANSWER_IDS, dense=True)
    if dense.shape != (len(rows), LAYERS, NEURONS) or dense.dtype != np.float32 or not np.isfinite(dense).all():
        raise ValueError("fixture dense shape, dtype or values failed")

    layer = LAYERS // 2
    neuron = int(np.abs(dense[0, layer]).argmax())
    original = float(dense[0, layer, neuron])
    sparse, _, _ = m.run(rows, reads=[(layer, neuron), (LAYERS - 1, 0)], tokens=ANSWER_IDS)
    if original == 0 or not np.array_equal(sparse, dense[:, [layer, LAYERS - 1], [neuron, 0]]):
        raise ValueError("fixture coordinate does not match the sparse read")

    edit_inputs = []
    handle = modules[layer].register_forward_hook(lambda _, args, output: edit_inputs.append(args[0].clone()))
    try:
        _, base, _ = m.run(rows, tokens=ANSWER_IDS)
        noop_rows = [Row(rows[0].ids, rows[0].pos, [(layer, neuron, original)]), rows[1]]
        _, noop_scores, _ = m.run(noop_rows, tokens=ANSWER_IDS)
        zero_rows = [Row(rows[0].ids, rows[0].pos, [(layer, neuron, 0.0)]), rows[1]]
        m.run(zero_rows, tokens=ANSWER_IDS)
    finally:
        handle.remove()
    capture = parity_error(base, hooked, fixtures)
    noop = parity_error(base, noop_scores, fixtures)
    if not capture["passed"]:
        raise ValueError(f"fixture capture parity failed: {capture}")
    if not noop["passed"] or not torch.equal(edit_inputs[0], edit_inputs[1]):
        raise ValueError(f"fixture no-op writeback failed: {noop}")
    expected = edit_inputs[0].clone()
    if float(expected[0, rows[0].pos, neuron]) != original:
        raise ValueError("fixture writeback coordinate differs from the dense read")
    expected[0, rows[0].pos, neuron] = 0.0
    if not torch.equal(edit_inputs[2], expected):
        raise ValueError("fixture single-coordinate zeroing failed")
    if hook_counts != [len(module._forward_pre_hooks) for module in modules]:
        raise ValueError("fixture left a runner hook registered")
    return {"prompts": len(rows), "prompt_evaluations": 10, "capture_parity": capture, "noop_parity": noop,
            "sparse_indexing": True, "single_coordinate_zeroing": True, "hooks_removed": True,
            "edit_coordinate": [0, layer, neuron]}


@app.function(image=image, volumes={VOL: vol}, gpu="H200", cpu=8, memory=131072, timeout=6 * 3600)
def run_pilot(files, manifest_sha, code_revision, artifact_dir, checkpoint_dir):
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download

    from runner.model import BATCH, Marin, Row

    manifest_groups = frozen_prompts(files, manifest_sha)
    out = Path(artifact_dir).resolve()
    if not out.is_relative_to(VOL) or out == Path(VOL) or out.exists():
        raise ValueError("use a new directory under the approved project Volume /vol")
    checkpoint = json.loads(files["checkpoint.json"])
    if not checkpoint_dir:
        checkpoint_dir = snapshot_download(MODEL_REPO, revision=MODEL_SHA,
                                           allow_patterns=list(checkpoint["files"]))
    verify_checkpoint(Path(checkpoint_dir), checkpoint)
    m = Marin.load(checkpoint_dir, POLE_STATS)
    if any(param.dtype != torch.bfloat16 for param in m.model.parameters()):
        raise ValueError("the model must be loaded in bf16")
    encoded = encode_rows(m.tok, manifest_groups["pilot"])
    encode_rows(m.tok, manifest_groups["fresh"])
    smoke = fixture_checks(m)

    rows = [Row(row["token_ids"], row["cue_index"]) for row in encoded]
    evidence = [dict(row) for row in encoded]
    for name, dense in (("unhooked", False), ("hooked", True)):
        acts, scores, _ = m.run(rows, tokens=ANSWER_IDS, dense=dense)
        waiver, mass = answer_scores(scores, encoded)
        for row, lp, probability, label_mass in zip(evidence, scores, waiver, mass, strict=True):
            row[f"logprobs_{name}"] = lp.tolist()
            row[f"waiver_{name}"] = float(probability)
            row[f"label_mass_{name}"] = float(label_mass)
    metadata = dict(
        request=REQUEST_ID, code_revision=code_revision, manifest_sha256=manifest_sha,
        model_repo=MODEL_REPO, weight_revision=MODEL_SHA, tokenizer_revision=MODEL_SHA,
        checkpoint_files=checkpoint["files"], dtype="bfloat16", gpu=torch.cuda.get_device_name(),
        python=platform.python_version(), cuda=torch.version.cuda,
        packages={name: version(name) for name in ("torch", "transformers", "accelerate", "tokenizers",
                                                   "huggingface-hub", "safetensors", "numpy", "modal")},
        batch_size=BATCH, answer_ids=dict(zip(("A", "B"), ANSWER_IDS)),
        prompt_evaluations=2 * len(rows), fixture_checks=smoke,
    )

    # Assemble and verify the complete artifact set before writing to the shared Volume.
    with TemporaryDirectory() as directory:
        run_dir = Path(directory)
        (run_dir / "scores.jsonl").write_text(jsonl_text(evidence), encoding="utf-8")
        (run_dir / "run.json").write_text(json_text(metadata), encoding="utf-8")
        np.save(run_dir / "activations.npy", acts, allow_pickle=False)
        report = analyze(run_dir, encoded, run_dir)
        out.mkdir(parents=True, exist_ok=False)
        for file in sorted(run_dir.iterdir(), key=lambda file: file.name == "report.json"):
            shutil.copyfile(file, out / file.name)
        vol.commit()
    return {"artifact_dir": str(out), "status": report["status"], "sha256": report["sha256"]}


@app.local_entrypoint()
def main(approved_manifest_sha: str, code_revision: str, artifact_dir: str, checkpoint_dir: str = ""):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if not re.fullmatch(r"[a-f0-9]{40}", code_revision) or head != code_revision:
        raise ValueError("check out the full reviewed Git commit SHA")
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "runner", f"requests/{REQUEST_ID}"], check=True)
    files = package_files()
    frozen_prompts(files, approved_manifest_sha)
    result = run_pilot.remote(files, approved_manifest_sha, code_revision, artifact_dir, checkpoint_dir)
    print(json_text(result))
    if result["status"] == "invalid_parity":
        raise SystemExit("Study parity failed. Inspect the saved evidence before rerunning the same inputs.")
