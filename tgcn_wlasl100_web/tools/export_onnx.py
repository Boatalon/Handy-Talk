#!/usr/bin/env python3
"""Export the included WLASL100 Pose-TGCN checkpoint to ONNX."""

import argparse
import importlib.util
from pathlib import Path

import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("tgcn_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    module = load_module(args.architecture)
    model = module.GCN_muti_att(
        input_feature=100,
        hidden_feature=64,
        num_class=100,
        p_dropout=0.3,
        num_stage=20,
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 55, 100, dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        args.output,
        input_names=["keypoints"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
        dynamic_axes={
            "keypoints": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
