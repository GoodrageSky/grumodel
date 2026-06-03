"""Write <output>.meta.json sidecar files for stage outputs.

Captures: stage name, timestamp, output path, input file mtimes/sizes,
config snapshot, and (if available) git HEAD. Used by every stage script.
"""
from __future__ import annotations
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _git_hash() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=3)
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _input_info(paths: Iterable[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if path.exists():
            st = path.stat()
            out.append({"path": str(path), "mtime": st.st_mtime, "size": st.st_size})
        else:
            out.append({"path": str(path), "mtime": None, "size": None})
    return out


def write_meta(output_path: Path | str, *, stage: str,
               config: dict, inputs: list[str]) -> Path:
    """Write `<output_path>.meta.json` next to `output_path`. Returns the meta path."""
    output_path = Path(output_path)
    meta_path = output_path.with_name(output_path.name + ".meta.json")
    meta = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_path": str(output_path),
        "input_paths": _input_info(inputs),
        "config": config,
        "git_hash": _git_hash(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta_path


if __name__ == "__main__":
    # smoke test — `python _meta.py`
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "x.parquet"
        out.write_bytes(b"hi")
        m = write_meta(out, stage="test", config={"a": 1, "b": "ok"}, inputs=[str(out)])
        loaded = json.loads(m.read_text(encoding="utf-8"))
        assert loaded["stage"] == "test"
        assert loaded["config"] == {"a": 1, "b": "ok"}
        assert loaded["input_paths"][0]["size"] == 2
        print("[_meta.py] smoke ok ->", m)
