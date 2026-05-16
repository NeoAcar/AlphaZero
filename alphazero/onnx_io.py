"""ONNX export + inference helpers for the policy-only fast path.

We export the model's `forward_policy` subgraph (no value head, no MCTS) at
a fixed input shape (1, 19, 8, 8) and load it with ONNX Runtime's CUDA
Execution Provider for inference. This bypasses PyTorch's per-op dispatch
entirely and is typically ~15-30% faster than torch.compile(max-autotune)
at batch=1.

Output ONNX graph:
    input  'board'          : (1, 19, 8, 8), float16 or float32
    output 'policy_logits'  : (1, 4672), same dtype as input
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


class _PolicyOnly(torch.nn.Module):
    """Adapter that exposes `model.forward_policy` as the wrapper's forward.

    `torch.onnx.export` traces `__call__` -> `forward`, so to export only
    the policy subgraph we wrap the real model and route forward through
    `forward_policy`.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner.forward_policy(x)


def export_policy_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    fp16: bool = True,
    opset: int = 17,
    device: torch.device | str = "cpu",
) -> Path:
    """Export model.forward_policy to ONNX. Returns the written path.

    Set fp16=True to bake half-precision weights into the graph; this is
    what unlocks Tensor Core kernels under CUDAExecutionProvider on
    Ampere+ GPUs.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    if fp16:
        model = model.half()
    model = model.to(device)

    dummy = torch.zeros(
        1, 19, 8, 8,
        dtype=torch.float16 if fp16 else torch.float32,
        device=device,
    )

    wrapped = _PolicyOnly(model)
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(output_path),
        input_names=["board"],
        output_names=["policy_logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,           # fixed shape -- enables best kernel pick
    )
    return output_path


class OnnxPolicyRunner:
    """Wraps an ORT InferenceSession for the exported policy graph.

    Use `__call__(board_np)` to get policy logits as a numpy array.

    On CUDA, uses ORT IOBinding with a preallocated GPU input OrtValue
    (reused across calls) and a CPU-backed output OrtValue (so the
    device->host copy lands directly in our numpy buffer with no extra
    allocation per call). Falls back to the default feed-dict path if
    IOBinding setup fails for any reason.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        device: str = "cuda",
        fp16: bool = True,
    ):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if device == "cuda":
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "cudnn_conv_use_max_workspace": "1",
                }),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.fp16 = fp16
        self.np_dtype = np.float16 if fp16 else np.float32
        self.device = device

        # Host-side input buffer in the correct dtype/shape, refilled per call.
        self._host_buf = np.zeros((1, 19, 8, 8), dtype=self.np_dtype)
        # CPU-backed numpy buffer the output is written into directly.
        self._out_buf = np.zeros((1, 4672), dtype=self.np_dtype)

        self._io_binding = None
        if device == "cuda":
            try:
                # Input on GPU, persistent across calls; refilled via update_inplace.
                self._input_ortvalue = ort.OrtValue.ortvalue_from_numpy(self._host_buf, "cuda", 0)
                # Output bound to our CPU numpy buffer; ORT writes here directly.
                self._output_ortvalue = ort.OrtValue.ortvalue_from_numpy(self._out_buf)
                io = self.session.io_binding()
                io.bind_ortvalue_input(self.input_name, self._input_ortvalue)
                io.bind_ortvalue_output(self.output_name, self._output_ortvalue)
                self._io_binding = io
            except Exception:
                self._io_binding = None

        # Warm up so first real call doesn't pay the cuDNN tuner cost.
        _ = self.__call__(self._host_buf[0])

    def __call__(self, board_planes: np.ndarray) -> np.ndarray:
        """board_planes: (19, 8, 8) or (1, 19, 8, 8) numpy array, any float dtype.

        Returns (4672,) numpy array of policy logits in the runner's dtype.
        """
        if board_planes.ndim == 3:
            board_planes = board_planes[None]
        if board_planes.dtype != self.np_dtype or board_planes.shape != self._host_buf.shape:
            np.copyto(self._host_buf, board_planes.astype(self.np_dtype, copy=False))
        else:
            np.copyto(self._host_buf, board_planes)

        if self._io_binding is not None:
            # Refresh device-side input from the host buffer, then run.
            # Output is already bound to self._out_buf -- ORT writes it directly.
            self._input_ortvalue.update_inplace(self._host_buf)
            self.session.run_with_iobinding(self._io_binding)
            return self._out_buf.reshape(-1)

        # Fallback: default feed-dict path (CPU EP, or if IOBinding setup failed).
        out = self.session.run([self.output_name], {self.input_name: self._host_buf})[0]
        return out.reshape(-1)
