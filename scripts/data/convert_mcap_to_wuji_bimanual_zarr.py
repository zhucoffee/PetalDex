
# Add the repository root for local package imports.
import sys as _sys
from pathlib import Path as _Path
_entry = _Path(__file__)
_project_root = _entry.resolve().parents[2]
_sys.path.insert(0, str(_project_root))

import argparse
import bisect
import glob
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from omegaconf import OmegaConf


try:
    from PIL import Image

    try:
        _RESAMPLE = Image.Resampling.LANCZOS
    except AttributeError:
        _RESAMPLE = Image.LANCZOS
except Exception:
    Image = None
    _RESAMPLE = None


@dataclass
class StreamSpec:
    topic: str
    kind: str
    expr: Optional[str] = None
    image_transport: str = "auto"
    # Optional crop in source-image pixels: [left, top, right, bottom].
    # Apply the crop before resizing image streams.
    crop: Optional[Sequence[int]] = None


DEFAULT_REQUIRED_STREAMS = (
    "cam_head_color",
    "cam_right_wrist_color",
    "left_arm_states",
    "right_arm_states",
    "left_hand_states",
    "right_hand_states",
    "left_arm_commands",
    "right_arm_commands",
    "left_hand_commands",
    "right_hand_commands",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ROS2 MCAP/rosbag data into the Wuji bimanual zarr format."
    )
    parser.add_argument("--config", required=True, help="YAML config for stream mappings.")
    parser.add_argument("--output", required=True, help="Target zarr path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output zarr.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"config must resolve to a dict, got {type(data)}")
    return data


def resolve_input_paths(cfg: Dict[str, Any]) -> List[str]:
    paths = []
    explicit = cfg.get("input_paths") or []
    if explicit:
        paths.extend([os.path.expanduser(str(x)) for x in explicit])

    pattern = cfg.get("input_glob")
    if pattern:
        paths.extend(sorted(glob.glob(os.path.expanduser(str(pattern)))))

    deduped = []
    seen = set()
    for path in paths:
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)

    if not deduped:
        raise ValueError("no input_paths or input_glob matched any files")
    return deduped


def build_stream_specs(cfg: Dict[str, Any]) -> Dict[str, StreamSpec]:
    streams = cfg.get("streams")
    if not isinstance(streams, dict) or not streams:
        raise ValueError("config.streams must be a non-empty mapping")

    specs: Dict[str, StreamSpec] = {}
    for name, raw in streams.items():
        if not isinstance(raw, dict):
            raise ValueError(f"stream {name!r} must be a mapping")
        topic = raw.get("topic")
        kind = raw.get("kind")
        if not topic or not kind:
            raise ValueError(f"stream {name!r} must provide both topic and kind")
        crop = raw.get("crop")
        if crop is not None:
            crop = [int(x) for x in crop]
            if len(crop) != 4:
                raise ValueError(f"stream {name!r} crop must be [left, top, right, bottom]")
            if crop[2] <= crop[0] or crop[3] <= crop[1]:
                raise ValueError(f"stream {name!r} crop must satisfy right>left and bottom>top, got {crop}")
        specs[name] = StreamSpec(
            topic=str(topic),
            kind=str(kind),
            expr=raw.get("expr"),
            image_transport=str(raw.get("image_transport", "auto")),
            crop=crop,
        )
    return specs


def ensure_runtime_dependencies() -> None:
    missing = []
    try:
        import zarr  # noqa: F401
    except Exception:
        missing.append("zarr")
    try:
        import rosbags  # noqa: F401
    except Exception:
        missing.append("rosbags")
    if Image is None:
        missing.append("Pillow")

    if missing:
        raise RuntimeError(
            "missing runtime dependencies: "
            + ", ".join(missing)
            + ". Install them in the training environment before running this converter."
        )


def iter_rosbag_messages(source_path: str, topics: Sequence[str]) -> Iterable[Tuple[str, int, Any]]:
    from rosbags.highlevel import AnyReader

    source = Path(source_path)
    with AnyReader([source]) as reader:
        connections = [x for x in reader.connections if x.topic in topics]
        topic_set = {x.topic for x in connections}
        missing = sorted(set(topics) - topic_set)
        if missing:
            raise ValueError(f"{source_path} is missing configured topics: {missing}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            yield connection.topic, int(timestamp), msg


def decode_compressed_image(encoded: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required to decode compressed images") from exc

    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cv2.imdecode returned None")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.uint8, copy=False)


def decode_raw_image(msg: Any) -> np.ndarray:
    encoding = getattr(msg, "encoding", "rgb8").lower()
    height = int(msg.height)
    width = int(msg.width)
    data = bytes(msg.data)

    if encoding in ("rgb8", "bgr8"):
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        if encoding == "bgr8":
            arr = arr[..., ::-1]
        return arr.copy()

    if encoding == "mono8":
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return np.repeat(arr[..., None], 3, axis=2)

    raise ValueError(f"unsupported raw image encoding: {encoding}")


def decode_image_message(msg: Any) -> np.ndarray:
    if hasattr(msg, "format") and hasattr(msg, "data"):
        return decode_compressed_image(np.asarray(msg.data, dtype=np.uint8))
    if hasattr(msg, "height") and hasattr(msg, "width") and hasattr(msg, "data"):
        return decode_raw_image(msg)
    raise ValueError(f"unsupported image message type: {type(msg)}")


def crop_rgb(image: np.ndarray, crop: Optional[Sequence[int]]) -> np.ndarray:
    """Crop to [left, top, right, bottom] before resizing."""
    if crop is None:
        return image
    left, top, right, bottom = [int(x) for x in crop]
    h, w = image.shape[:2]
    if right > w or bottom > h or left < 0 or top < 0:
        raise ValueError(
            f"crop {list(crop)} exceeds image bounds {w}x{h} (WxH); set crop coordinates for this image resolution"
        )
    return image[top:bottom, left:right]


def resize_rgb(image: np.ndarray, resize_hw: Optional[Sequence[int]]) -> np.ndarray:
    if resize_hw is None:
        return image
    if Image is None:
        raise RuntimeError("Pillow is required for image resize")

    target_h, target_w = [int(x) for x in resize_hw]
    pil = Image.fromarray(image)
    if pil.size != (target_w, target_h):
        pil = pil.resize((target_w, target_h), _RESAMPLE)
    return np.asarray(pil, dtype=np.uint8)


def eval_vector_expr(expr: str, msg: Any) -> np.ndarray:
    value = eval(expr, {"__builtins__": {}}, {"np": np, "msg": msg})
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"vector expr must produce a 1D array, got shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("vector expr produced non-finite joint values")
    return arr


def extract_stream_value(spec: StreamSpec, msg: Any, resize_hw: Optional[Sequence[int]]) -> np.ndarray:
    if spec.kind == "image":
        image = decode_image_message(msg)
        image = crop_rgb(image, spec.crop)
        image = resize_rgb(image, resize_hw)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"decoded image must be HWC RGB, got shape={image.shape}")
        return image.astype(np.uint8, copy=False)

    if spec.kind == "vector":
        if not spec.expr:
            raise ValueError(f"vector stream {spec.topic} requires expr")
        return eval_vector_expr(spec.expr, msg)

    raise ValueError(f"unsupported stream kind: {spec.kind}")


def collect_stream_samples(
    source_path: str,
    stream_specs: Dict[str, StreamSpec],
    resize_hw: Optional[Sequence[int]],
) -> Dict[str, List[Tuple[int, np.ndarray]]]:
    by_topic = {spec.topic: name for name, spec in stream_specs.items()}
    topics = list(by_topic.keys())
    samples: Dict[str, List[Tuple[int, np.ndarray]]] = {name: [] for name in stream_specs}

    for topic, timestamp, msg in iter_rosbag_messages(source_path, topics):
        name = by_topic[topic]
        spec = stream_specs[name]
        try:
            value = extract_stream_value(spec, msg, resize_hw)
        except Exception as exc:
            raise ValueError(f"stream {name!r} ({topic}) at {timestamp}: {exc}") from exc
        samples[name].append((timestamp, value))
    return samples


def nearest_index(timestamps: Sequence[int], target: int, tolerance_ns: int, name: str = '?') -> int:
    pos = bisect.bisect_left(timestamps, target)
    candidates = []
    if pos < len(timestamps):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    if not candidates:
        raise ValueError("no timestamps available for alignment")

    best_idx = min(candidates, key=lambda idx: abs(timestamps[idx] - target))
    if abs(timestamps[best_idx] - target) > tolerance_ns:
        delta = abs(timestamps[best_idx] - target)
        raise ValueError(
            f"stream {name!r}: no sample within tolerance for target={target}, "
            f"nearest delta={delta} ns ({delta/1e6:.2f} ms), tolerance={tolerance_ns/1e6:.0f} ms"
        )
    return best_idx


def align_episode_samples(
    samples: Dict[str, List[Tuple[int, np.ndarray]]],
    reference_stream: str,
    tolerance_sec: float,
    required_streams: Sequence[str],
) -> Dict[str, np.ndarray]:
    tolerance_ns = int(float(tolerance_sec) * 1e9)
    reference_samples = samples[reference_stream]
    if not reference_samples:
        raise ValueError(f"reference stream {reference_stream!r} has no samples")

    aligned: Dict[str, List[np.ndarray]] = {name: [] for name in samples}
    stream_timestamps = {name: [ts for ts, _ in values] for name, values in samples.items()}

    # Skip frames missing a synchronized input; reject episodes with excessive drops.
    drop_counts: Dict[str, int] = {}
    skipped_frames = 0
    total_frames = len(reference_samples)

    for ref_ts, ref_value in reference_samples:
        frame_buf: Dict[str, np.ndarray] = {reference_stream: ref_value}
        skip = False
        for name, values in samples.items():
            if name == reference_stream:
                continue
            timestamps = stream_timestamps[name]
            if not timestamps:
                raise ValueError(f"stream {name!r} has no samples")
            try:
                idx = nearest_index(timestamps, ref_ts, tolerance_ns, name=name)
            except ValueError:
                drop_counts[name] = drop_counts.get(name, 0) + 1
                skip = True
                break
            frame_buf[name] = values[idx][1]

        if skip:
            skipped_frames += 1
            continue

        for name, val in frame_buf.items():
            aligned[name].append(val)

    kept = total_frames - skipped_frames
    drop_pct = (skipped_frames / max(total_frames, 1)) * 100.0
    print(f"[align] kept={kept}/{total_frames} frames ({drop_pct:.2f}% dropped)")
    if drop_counts:
        for name, n in sorted(drop_counts.items(), key=lambda kv: -kv[1]):
            print(f"        drops by stream: {name} = {n}")

    if drop_pct > 5.0:
        raise ValueError(
            f"too many frames dropped during alignment: {skipped_frames}/{total_frames} "
            f"({drop_pct:.1f}%, threshold 5%). Check stream timestamps and alignment tolerance."
        )

    result: Dict[str, np.ndarray] = {}
    for name, values in aligned.items():
        if name in required_streams and not values:
            raise ValueError(f"required stream {name!r} produced no aligned samples")
        result[name] = np.stack(values, axis=0)
    return result


COMMAND_STREAMS = (
    "left_arm_commands",
    "right_arm_commands",
    "left_hand_commands",
    "right_hand_commands",
)


def build_action(aligned: Dict[str, np.ndarray]) -> np.ndarray:
    if "action" in aligned:
        return aligned["action"].astype(np.float32)

    missing = [name for name in COMMAND_STREAMS if name not in aligned]
    if missing:
        raise ValueError(
            "cannot build action because the aligned data is missing command streams: "
            + ", ".join(missing)
        )

    return np.concatenate(
        [
            aligned["left_arm_commands"],
            aligned["right_arm_commands"],
            aligned["left_hand_commands"],
            aligned["right_hand_commands"],
        ],
        axis=1,
    ).astype(np.float32)


def ensure_required_streams(stream_specs: Dict[str, StreamSpec], required_streams: Sequence[str]) -> None:
    missing = sorted(set(required_streams) - set(stream_specs.keys()))
    if missing:
        raise ValueError(f"missing required configured streams: {missing}")


def create_appendable_array(group: Any, name: str, sample_array: np.ndarray, chunk_len: int = 64) -> Any:
    shape = (0,) + sample_array.shape[1:]
    chunks = (min(chunk_len, max(1, len(sample_array))),) + sample_array.shape[1:]
    return group.create_dataset(name=name, shape=shape, chunks=chunks, dtype=sample_array.dtype)


def append_to_zarr(arr: Any, data: np.ndarray) -> None:
    old_n = arr.shape[0]
    new_n = old_n + data.shape[0]
    arr.resize(new_n, *arr.shape[1:])
    arr[old_n:new_n] = data


def _write_zarr(output_path: str, episodes: Iterable[Dict[str, np.ndarray]]) -> None:
    import zarr

    root = zarr.open(output_path, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    arrays = None
    episode_ends = []
    total_steps = 0

    for episode_idx, episode in enumerate(episodes):
        if arrays is None:
            arrays = {}
            for key, value in episode.items():
                arrays[key] = create_appendable_array(data_group, key, value)

        for key, value in episode.items():
            append_to_zarr(arrays[key], value)

        total_steps += int(next(iter(episode.values())).shape[0])
        episode_ends.append(total_steps)
        print(f"[write] episode {episode_idx + 1}: {total_steps} total steps")

    if arrays is None:
        raise RuntimeError("no episodes were converted")
    meta_group.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64))


def write_zarr(output_path: str, episodes: Iterable[Dict[str, np.ndarray]], overwrite: bool) -> None:
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_zarr(str(temporary_path), episodes)
        # Keep any existing dataset intact until conversion has completed.
        if output.exists():
            if not overwrite:
                raise FileExistsError(output)
            shutil.rmtree(output)
        temporary_path.rename(output)
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_runtime_dependencies()

    stream_specs = build_stream_specs(cfg)
    required_streams = cfg.get("required_streams") or list(DEFAULT_REQUIRED_STREAMS)
    ensure_required_streams(stream_specs, required_streams)

    input_paths = resolve_input_paths(cfg)
    output_path = Path(args.output).expanduser().resolve()
    for source_path in input_paths:
        source = Path(source_path).expanduser().resolve()
        if (source == output_path or source in output_path.parents
                or output_path in source.parents):
            raise ValueError(f"Input and output paths must not overlap: {source}")
    resize_hw = cfg.get("resize_hw")
    reference_stream = str(cfg.get("reference_stream", "left_arm_commands"))
    tolerance_sec = float(cfg.get("timestamp_tolerance_sec", 0.1))
    min_frames = int(cfg.get("min_frames", 1))

    def convert_episodes():
        skipped_bags = []
        converted_count = 0
        for source_path in input_paths:
            print(f"[read] {source_path}")
            try:
                samples = collect_stream_samples(source_path, stream_specs, resize_hw)
                aligned = align_episode_samples(samples, reference_stream, tolerance_sec, required_streams)
                # Release decoded source frames once aligned arrays are available.
                del samples

                episode = {}
                for name, array in aligned.items():
                    if name in COMMAND_STREAMS:
                        continue
                    episode[name] = array.astype(np.float32) if array.dtype.kind in "iu" and array.ndim == 2 else array
                episode["action"] = build_action(aligned)
                del aligned
            except Exception as exc:
                print(f"[skip] {source_path}: {type(exc).__name__}: {exc}")
                skipped_bags.append((source_path, str(exc)))
                continue

            episode_len = int(episode["action"].shape[0])
            if episode_len < min_frames:
                print(f"[skip] {source_path}: only {episode_len} frames (< {min_frames})")
                skipped_bags.append((source_path, f"only {episode_len} frames"))
                continue

            print(f"[ok] {source_path}: {episode_len} aligned steps")
            for key, value in episode.items():
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
            converted_count += 1
            yield episode
            del episode

        if skipped_bags:
            print(f"\n[summary] skipped {len(skipped_bags)} bag(s):")
            for path, reason in skipped_bags:
                print(f"  - {path}: {reason}")
        print(f"[summary] kept {converted_count}/{len(input_paths)} bags")

    write_zarr(str(output_path), convert_episodes(), overwrite=args.overwrite)
    print(f"[done] wrote zarr to {args.output}")


if __name__ == "__main__":
    main()
