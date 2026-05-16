"""Export a trained AlphaZero policy network's `forward_policy` subgraph
to ONNX for use with the ONNX Runtime inference path.

Example:
    uv run python export_onnx.py \\
        --checkpoint models/model_best_combined_wdl.pth \\
        --architecture seresnetwdl \\
        --output models/policy_seresnetwdl.fp16.onnx \\
        --fp16

The output .onnx file can be loaded by alphazero.onnx_io.OnnxPolicyRunner
or by setting `backend: "onnx"` and `onnx_path: ...` in a policy_only
player config.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from alphazero.nn import ResNet, SEResNet, SEResNetWDL
from alphazero.onnx_io import export_policy_to_onnx


ARCHITECTURES = {
    "resnet": ResNet,
    "seresnet": SEResNet,
    "seresnetwdl": SEResNetWDL,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--architecture", required=True, choices=list(ARCHITECTURES),
                   help="Model architecture matching the checkpoint")
    p.add_argument("--output", required=True, help="Output .onnx path")
    p.add_argument("--fp16", action="store_true", help="Export with FP16 weights (recommended for CUDA EP)")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version (default 17)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device used during the export trace")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[export] device={device}, fp16={args.fp16}, opset={args.opset}")

    model = ARCHITECTURES[args.architecture]().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"[export] loaded {args.architecture} from {args.checkpoint}")

    out_path = export_policy_to_onnx(
        model,
        Path(args.output),
        fp16=args.fp16,
        opset=args.opset,
        device=device,
    )
    print(f"[export] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
