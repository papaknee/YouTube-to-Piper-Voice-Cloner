#!/usr/bin/env python3
"""Run Piper ONNX export with a torch.onnx.export compatibility shim."""

from __future__ import annotations

import runpy
import sys
import torch


def main() -> None:
    original_export = torch.onnx.export

    def _compat_export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return original_export(*args, **kwargs)

    torch.onnx.export = _compat_export
    sys.argv = ["piper.train.export_onnx", *sys.argv[1:]]
    runpy.run_module("piper.train.export_onnx", run_name="__main__")


if __name__ == "__main__":
    main()
