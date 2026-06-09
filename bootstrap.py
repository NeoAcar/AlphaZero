"""Create a randomly-initialised model checkpoint for tabula-rasa self-play.

Run once before the first self-play session. Drops `models/v00_seed.pth`
(by default), which becomes the seed checkpoint for the entire training
lineage. Every subsequent checkpoint is the output of a training run --
this script is never invoked again unless you start a brand-new lineage.

Usage:
    uv run python bootstrap.py \\
        --architecture seresnetwdl \\
        --in-channels 119 \\
        --output models/v00_seed.pth
"""
import argparse
from pathlib import Path

import torch

from alphazero.nn import (
    INPUT_PLANES_HISTORY,
    ResNet,
    SEResNet,
    SEResNetWDL,
)


ARCHITECTURES = {
    "resnet":      ResNet,
    "seresnet":    SEResNet,
    "seresnetwdl": SEResNetWDL,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--architecture", choices=list(ARCHITECTURES), default="seresnetwdl",
                   help="Network class to instantiate. seresnetwdl recommended for tabula-rasa runs.")
    p.add_argument("--in-channels", type=int, default=INPUT_PLANES_HISTORY,
                   help=f"Input plane count. 119 = AlphaZero 8-frame history (default); "
                        f"19 = legacy current-board-only.")
    p.add_argument("--output", default="models/v00_seed.pth",
                   help="Where to write the checkpoint.")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional torch RNG seed for fully reproducible weights.")
    cli = p.parse_args()

    if cli.seed is not None:
        torch.manual_seed(cli.seed)

    cls = ARCHITECTURES[cli.architecture]
    model = cls(in_channels=cli.in_channels)
    model.eval()

    out_path = Path(cli.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "meta": {
            "architecture":  cli.architecture,
            "in_channels":   cli.in_channels,
            "tabula_rasa":   True,
            "torch_seed":    cli.seed,
        },
    }
    torch.save(payload, out_path)

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path}  ({size_mb:.1f} MB, {n_params/1e6:.1f}M params)")
    print(f"  architecture: {cli.architecture}")
    print(f"  in_channels:  {cli.in_channels}")
    if cli.seed is not None:
        print(f"  torch_seed:   {cli.seed}")


if __name__ == "__main__":
    main()
