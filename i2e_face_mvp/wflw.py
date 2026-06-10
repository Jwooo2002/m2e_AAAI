from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class WFLWSample:
    annotation_file: Path
    line_index: int
    image_path: Path
    landmarks: list[dict]
    rect: list[float]
    attributes: list[int]
    relative_image_path: str


def search_wflw_annotations(search_roots: Iterable[Path] | None = None) -> tuple[Path, Path]:
    roots = [Path(os.path.expanduser(str(root))) for root in list(search_roots or default_search_roots())]
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for train, test in common_annotation_candidates(root):
            if train.exists() and test.exists():
                return train, test
    for expanded in roots:
        if not expanded.exists():
            continue
        for name in ("list_98pt_rect_attr_train.txt", "list_98pt_rect_attr_test.txt"):
            for path in expanded.rglob(name):
                found[name] = path
                if len(found) == 2:
                    return found["list_98pt_rect_attr_train.txt"], found["list_98pt_rect_attr_test.txt"]
    missing = {"list_98pt_rect_attr_train.txt", "list_98pt_rect_attr_test.txt"} - set(found)
    raise FileNotFoundError(f"Could not find WFLW annotation files under {', '.join(map(str, roots))}; missing {sorted(missing)}")


def default_search_roots() -> list[Path]:
    roots: list[Path] = []
    for prefix in ("/mnt/hdd*", "/mnt/ssd*", "/mnt/nas*"):
        roots.extend(sorted(Path("/").glob(prefix.lstrip("/"))))
    roots.extend([Path("~/datasets"), Path("~/data"), Path("./data")])
    return roots


def common_annotation_candidates(root: Path) -> list[tuple[Path, Path]]:
    suffix = Path("WFLW_annotations/list_98pt_rect_attr_train_test")
    bases = [
        root / suffix,
        root / "WFLW" / suffix,
        root / "WFLW_V_release" / suffix,
        root / "data" / "WFLW" / suffix,
    ]
    return [
        (base / "list_98pt_rect_attr_train.txt", base / "list_98pt_rect_attr_test.txt")
        for base in bases
    ]


def load_first_valid_sample(annotation_file: Path, max_lines: int = 5000) -> WFLWSample:
    with annotation_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            sample = parse_wflw_line(annotation_file, line, i)
            if sample.image_path.exists():
                return sample
    raise FileNotFoundError(f"No valid image referenced by the first {max_lines} rows of {annotation_file}")


def parse_wflw_line(annotation_file: Path, line: str, line_index: int = 0) -> WFLWSample:
    parts = line.strip().split()
    if len(parts) < 207:
        raise ValueError(f"WFLW row has {len(parts)} tokens, expected at least 207")

    coords = [float(v) for v in parts[:196]]
    landmarks = [
        {"id": i, "x": coords[2 * i], "y": coords[2 * i + 1], "visibility": True}
        for i in range(98)
    ]
    rect = [float(v) for v in parts[196:200]]
    attributes = [int(float(v)) for v in parts[200:206]]
    rel_path = parts[-1]
    image_path = resolve_image_path(annotation_file, rel_path)
    return WFLWSample(annotation_file, line_index, image_path, landmarks, rect, attributes, rel_path)


def resolve_image_path(annotation_file: Path, rel_path: str) -> Path:
    direct = Path(rel_path).expanduser()
    if direct.is_absolute() and direct.exists():
        return direct

    parents = [annotation_file.parent, *annotation_file.parents]
    candidates: list[Path] = []
    for parent in parents[:8]:
        candidates.extend(
            [
                parent / rel_path,
                parent / "WFLW_images" / rel_path,
                parent / "images" / rel_path,
                parent.parent / "WFLW_images" / rel_path,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    filename = Path(rel_path).name
    for parent in parents[:6]:
        try:
            for candidate in parent.rglob(filename):
                if candidate.is_file():
                    return candidate
        except PermissionError:
            continue
    return candidates[0]


def save_landmarks_json(path: Path, sample: WFLWSample, image_shape: tuple[int, int]) -> None:
    payload = {
        "dataset_name": "WFLW",
        "annotation_file": str(sample.annotation_file),
        "annotation_line_index": sample.line_index,
        "image_path": str(sample.image_path),
        "image_height": image_shape[0],
        "image_width": image_shape[1],
        "coordinate_format": "xy",
        "num_keypoints": len(sample.landmarks),
        "rect": sample.rect,
        "attributes": sample.attributes,
        "landmarks": sample.landmarks,
        "semantic_anchors": wflw_semantic_anchors(sample.landmarks),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def wflw_semantic_anchors(landmarks: list[dict]) -> dict:
    groups = {
        "face_contour": range(0, 33),
        "jawline": range(0, 33),
        "left_eyebrow": range(33, 42),
        "right_eyebrow": range(42, 51),
        "nose": range(51, 60),
        "left_eye": range(60, 68),
        "right_eye": range(68, 76),
        "mouth": range(76, 96),
    }
    return {
        name: [{"id": i, "x": landmarks[i]["x"], "y": landmarks[i]["y"]} for i in idxs]
        for name, idxs in groups.items()
    }
