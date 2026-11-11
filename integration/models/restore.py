"""Restore verified Release assets into the training server directory layout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

REPO = "prummn/face-age-transformation-course-project"
TAG = "models-2026-11-11"
DEMO = {
    "face-age/models/sam/sam_ffhq_aging.pt",
    "face-age/out/ft_l1/encoder_step1000.pt",
    "face-age/models/faceattr/convnext_t_fairface.pt",
    "face-age/models/yunet/face_detection_yunet_2023mar.onnx",
}


def download(name, directory):
    subprocess.run(["gh", "release", "download", TAG, "--repo", REPO, "--pattern",
                    name, "--dir", str(directory)], check=True)
    return directory / name


def restore(row, root):
    target = (root / row["path"]).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Unexpected destination")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with target.open("rb") as existing:
            if hashlib.file_digest(existing, "sha256").hexdigest() == row["sha256"]:
                return
        raise ValueError(f"Existing file has a different digest: {target}")
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        folder = Path(tmp)
        complete = hashlib.sha256()
        with (folder / "restored").open("wb") as output:
            for asset in row["assets"]:
                part = download(asset["name"], folder)
                digest = hashlib.sha256()
                with part.open("rb") as source:
                    while data := source.read(1024 * 1024):
                        output.write(data)
                        complete.update(data)
                        digest.update(data)
                if digest.hexdigest() != asset["sha256"]:
                    raise ValueError(f"Invalid asset: {part.name}")
                part.unlink()
        if complete.hexdigest() != row["sha256"]:
            raise ValueError(f"Invalid restored file: {target}")
        os.replace(folder / "restored", target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Parent directory of face-age")
    parser.add_argument("--all", action="store_true", help="Include training and evaluation weights")
    args = parser.parse_args()
    rows = json.loads(Path(__file__).with_name("weights.json").read_text())
    for row in rows:
        if args.all or row["path"] in DEMO:
            print(row["path"], flush=True)
            restore(row, args.root.resolve())
