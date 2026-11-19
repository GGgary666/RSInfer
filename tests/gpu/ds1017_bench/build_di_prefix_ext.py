"""Build the ``ds_prefix_pages`` C++ extension against ``$DEEPINFER_ROOT``."""
# pyright: reportMissingImports=false

from __future__ import annotations

import os
from pathlib import Path
from torch.utils.cpp_extension import load


def main() -> None:
    here = Path(__file__).resolve().parent
    ext_dir = here / "extension"
    ext_dir.mkdir(exist_ok=True)
    load(
        name="ds_prefix_pages",
        sources=[str(here / "di_prefix_pages.cpp")],
        build_directory=str(ext_dir),
        extra_include_paths=[os.environ["DEEPINFER_ROOT"]],
        verbose=True,
    )


if __name__ == "__main__":
    main()
