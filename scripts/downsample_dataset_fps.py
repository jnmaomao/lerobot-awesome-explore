#!/usr/bin/env python
"""Downsample a LeRobot dataset from a higher FPS to a lower FPS.

Converts a dataset (e.g. 30fps) to a lower frame rate (e.g. 15fps) by
subsampling every Nth frame.  Handles parquet data, video re-encoding,
and all metadata updates.

Usage:
    # Convert 30fps dataset to 15fps
    python scripts/downsample_dataset_fps.py \
        --repo_id local/nexarm_smolvla_train \
        --target_fps 15 \
        --output_repo_id local/nexarm_smolvla_train_15fps

    # Dry run to preview changes
    python scripts/downsample_dataset_fps.py \
        --repo_id local/nexarm_smolvla_train \
        --target_fps 15 \
        --dry_run

    # Custom output directory
    python scripts/downsample_dataset_fps.py \
        --repo_id local/nexarm_smolvla_train \
        --target_fps 15 \
        --output_dir /data/datasets/nexarm_15fps
"""

import argparse
import logging
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import load_episodes, write_info, write_stats, write_tasks
from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_PATH,
)
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.video_utils import get_video_info
from lerobot.utils.constants import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
logger = logging.getLogger(__name__)

# Also log to stdout for CLI usage
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def downsample_dataset_fps(
    repo_id: str,
    target_fps: int,
    output_repo_id: str | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    use_gpu: bool = True,
) -> LeRobotDatasetMetadata | None:
    """Downsample a LeRobot dataset to a lower FPS.

    Args:
        repo_id: Source dataset repo_id (e.g. "local/nexarm_smolvla_train").
        target_fps: Target FPS (must be an integer divisor of source FPS).
        output_repo_id: Repo ID for the output dataset. Defaults to
            ``{repo_id}_{target_fps}fps``.
        output_dir: Custom output directory. Defaults to
            ``$HF_LEROBOT_HOME/{output_repo_id}``.
        dry_run: If True, only print what would be done without writing files.
    """
    # Load source metadata
    src_meta = LeRobotDatasetMetadata(repo_id)
    src_fps = src_meta.fps

    if target_fps >= src_fps:
        raise ValueError(
            f"target_fps ({target_fps}) must be less than source fps ({src_fps})"
        )

    if src_fps % target_fps != 0:
        raise ValueError(
            f"source fps ({src_fps}) must be evenly divisible by target_fps ({target_fps}). "
            f"Valid targets: {[f for f in range(1, src_fps) if src_fps % f == 0]}"
        )

    subsample_factor = src_fps // target_fps

    if output_repo_id is None:
        output_repo_id = f"{repo_id}_{target_fps}fps"
    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        output_dir = HF_LEROBOT_HOME / output_repo_id

    # Load source episodes
    src_episodes = load_episodes(src_meta.root)
    if src_episodes is None or len(src_episodes) == 0:
        raise ValueError(f"No episodes found in dataset {repo_id}")

    total_src_frames = sum(ep["length"] for ep in src_episodes)
    total_dst_frames = total_src_frames // subsample_factor

    logger.info(
        f"Downsampling: {src_fps}fps → {target_fps}fps "
        f"(subsample_factor={subsample_factor})"
    )
    logger.info(
        f"Source: {len(src_episodes)} episodes, {total_src_frames} frames"
    )
    logger.info(
        f"Output: {len(src_episodes)} episodes, ~{total_dst_frames} frames"
    )
    logger.info(f"Output path: {output_dir}")

    if use_gpu:
        logger.info("GPU-accelerated video encoding: enabled (NVENC)")
    else:
        logger.info("GPU-accelerated video encoding: disabled (software only)")

    if dry_run:
        logger.info("[DRY RUN] No files will be written.")
        # Print per-episode info
        for ep_idx, ep in enumerate(src_episodes):
            new_length = ep["length"] // subsample_factor
            logger.info(
                f"  Episode {ep_idx}: {ep['length']} frames → {new_length} frames"
            )
        return None

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory {output_dir} already exists. "
            "Remove it or use a different output_dir."
        )

    # ── Step 1: Create destination metadata ──────────────────────
    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=output_repo_id,
        fps=target_fps,
        features=src_meta.features,
        robot_type=src_meta.robot_type,
        root=output_dir,
        use_videos=len(src_meta.video_keys) > 0,
    )

    # Copy tasks
    task_names = [src_meta.tasks.iloc[i].name for i in range(len(src_meta.tasks))]
    dst_meta.save_episode_tasks(task_names)

    # ── Step 2: Process parquet data (subsample frames) ─────────
    logger.info("Processing parquet data...")
    episode_data_metadata = _subsample_and_reindex_data(
        src_meta, dst_meta, src_episodes, subsample_factor
    )

    # ── Step 3: Re-encode videos at new FPS ──────────────────────
    episode_video_metadata = {}
    if src_meta.video_keys:
        logger.info("Re-encoding videos...")
        episode_video_metadata = _subsample_videos(
            src_meta, dst_meta, src_episodes, subsample_factor, target_fps,
            use_gpu=use_gpu,
        )

    # ── Step 4: Save episode metadata ────────────────────────────
    logger.info("Saving episode metadata...")
    _save_subsampled_episodes_metadata(
        src_meta, dst_meta, src_episodes, episode_data_metadata,
        episode_video_metadata, subsample_factor, target_fps
    )

    # ── Step 5: Update video info in info.json ───────────────────
    if dst_meta.video_keys:
        dst_meta.update_video_info()

    # ── Step 6: Recompute stats ──────────────────────────────────
    logger.info("Recomputing dataset statistics...")
    _recompute_stats(dst_meta, src_meta)

    logger.info(
        f"Done! Created {target_fps}fps dataset at {output_dir} "
        f"({dst_meta.total_episodes} episodes, {dst_meta.total_frames} frames)"
    )

    return dst_meta


def _subsample_and_reindex_data(
    src_meta: LeRobotDatasetMetadata,
    dst_meta: LeRobotDatasetMetadata,
    src_episodes: list[dict],
    subsample_factor: int,
) -> dict[int, dict]:
    """Subsample parquet data files by keeping every Nth frame per episode.

    Returns dict mapping new episode index to data file metadata.
    """
    # Group source episodes by their data parquet file
    file_to_episodes: dict[Path, list[int]] = {}
    for ep_idx, ep in enumerate(src_episodes):
        chunk_idx = ep["data/chunk_index"]
        file_idx = ep["data/file_index"]
        file_path = Path(
            DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        )
        if file_path not in file_to_episodes:
            file_to_episodes[file_path] = []
        file_to_episodes[file_path].append(ep_idx)

    # Build task mapping
    task_mapping = {}
    for old_task_idx in range(len(src_meta.tasks)):
        task_name = src_meta.tasks.iloc[old_task_idx].name
        new_task_idx = dst_meta.get_task_index(task_name)
        if new_task_idx is not None:
            task_mapping[old_task_idx] = new_task_idx

    global_index = 0
    episode_data_metadata: dict[int, dict] = {}

    for src_path in tqdm(sorted(file_to_episodes.keys()), desc="Subsampling data"):
        full_path = src_meta.root / src_path
        df = pd.read_parquet(full_path)

        # Process each episode in this file
        new_frames = []
        for ep_idx in file_to_episodes[src_path]:
            ep_df = df[df["episode_index"] == ep_idx]
            # Subsample: keep every Nth frame (0, subsample_factor, 2*subsample_factor, ...)
            subsampled = ep_df.iloc[::subsample_factor].copy()
            if len(subsampled) == 0:
                logger.warning(
                    f"Episode {ep_idx} has no frames after subsampling "
                    f"(length={len(ep_df)}, factor={subsample_factor})"
                )
                continue
            new_frames.append(subsampled)

        if not new_frames:
            continue

        combined = pd.concat(new_frames, ignore_index=True)

        # Reindex
        num_rows = len(combined)
        combined["index"] = range(global_index, global_index + num_rows)

        # Reindex frame_index and timestamp per episode
        for ep_idx in file_to_episodes[src_path]:
            mask = combined["episode_index"] == ep_idx
            ep_rows = combined[mask]
            new_frame_indices = range(len(ep_rows))
            combined.loc[mask, "frame_index"] = list(new_frame_indices)
            combined.loc[mask, "timestamp"] = [
                fi / dst_meta.fps for fi in new_frame_indices
            ]

        # Remap task indices
        combined["task_index"] = combined["task_index"].replace(task_mapping)

        # Write to destination (use same chunk/file structure)
        first_ep = file_to_episodes[src_path][0]
        src_ep = src_episodes[first_ep]
        chunk_idx = src_ep["data/chunk_index"]
        file_idx = src_ep["data/file_index"]

        dst_path = dst_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=chunk_idx, file_index=file_idx
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Use same schema as source
        table = pa.Table.from_pandas(combined, preserve_index=False)
        pq.write_table(table, dst_path, compression="snappy")

        # Record metadata per episode
        for ep_idx in file_to_episodes[src_path]:
            ep_rows = combined[combined["episode_index"] == ep_idx]
            episode_data_metadata[ep_idx] = {
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": int(ep_rows["index"].min()),
                "dataset_to_index": int(ep_rows["index"].max() + 1),
            }

        global_index += num_rows

    return episode_data_metadata


def _subsample_videos(
    src_meta: LeRobotDatasetMetadata,
    dst_meta: LeRobotDatasetMetadata,
    src_episodes: list[dict],
    subsample_factor: int,
    target_fps: int,
    use_gpu: bool = True,
) -> dict[int, dict]:
    """Re-encode videos at the new FPS by subsampling frames.

    Returns dict mapping episode index to video metadata.
    """
    episode_video_metadata: dict[int, dict] = {}

    for vid_key in src_meta.video_keys:
        logger.info(f"  Processing video key: {vid_key}")

        # Get source video encoder info from features
        src_feature_info = src_meta.features[vid_key].get("info", {})
        vcodec = src_feature_info.get("video.codec", "h264")
        pix_fmt = src_feature_info.get("video.pix_fmt", "yuv420p")

        # Group episodes by video file
        file_to_episodes: dict[Path, list[int]] = {}
        for ep_idx, ep in enumerate(src_episodes):
            if f"videos/{vid_key}/chunk_index" not in ep:
                continue
            chunk_idx = ep[f"videos/{vid_key}/chunk_index"]
            file_idx = ep[f"videos/{vid_key}/file_index"]
            file_path = Path(
                DEFAULT_VIDEO_PATH.format(
                    video_key=vid_key, chunk_index=chunk_idx, file_index=file_idx
                )
            )
            if file_path not in file_to_episodes:
                file_to_episodes[file_path] = []
            file_to_episodes[file_path].append(ep_idx)

        for src_path in tqdm(sorted(file_to_episodes.keys()), desc=f"  Re-encoding {vid_key}"):
            full_src_path = src_meta.root / src_path

            # Determine output path (same chunk/file structure)
            first_ep_idx = file_to_episodes[src_path][0]
            src_ep = src_episodes[first_ep_idx]
            chunk_idx = src_ep[f"videos/{vid_key}/chunk_index"]
            file_idx = src_ep[f"videos/{vid_key}/file_index"]

            dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=vid_key, chunk_index=chunk_idx, file_index=file_idx
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Re-encode video with subsampled frames
            _subsample_video_with_av(
                input_path=full_src_path,
                output_path=dst_path,
                subsample_factor=subsample_factor,
                target_fps=target_fps,
                vcodec=vcodec,
                pix_fmt=pix_fmt,
                episodes=file_to_episodes[src_path],
                src_episodes=src_episodes,
                use_gpu=use_gpu,
            )

            # Record video metadata per episode
            for ep_idx in file_to_episodes[src_path]:
                if ep_idx not in episode_video_metadata:
                    episode_video_metadata[ep_idx] = {}
                src_ep = src_episodes[ep_idx]
                new_length = src_ep["length"] // subsample_factor

                # Compute new from_timestamp for this episode
                # Original from_timestamp + offset adjusted for subsampling
                orig_from_ts = src_ep.get(f"videos/{vid_key}/from_timestamp", 0.0)
                # The first frame of the episode in the original video
                # After subsampling, timestamp = frame_index / target_fps
                episode_video_metadata[ep_idx][f"videos/{vid_key}/chunk_index"] = chunk_idx
                episode_video_metadata[ep_idx][f"videos/{vid_key}/file_index"] = file_idx
                # from_timestamp stays at 0 for the new video (each episode starts at t=0 in its segment)
                episode_video_metadata[ep_idx][f"videos/{vid_key}/from_timestamp"] = 0.0

    return episode_video_metadata


def _get_nvenc_codec(vcodec: str) -> tuple[str, dict]:
    """Try to find an NVENC (GPU) codec for the given source codec.

    Returns (codec_name, codec_options). Falls back to software encoding
    if NVENC is not available.
    """
    # Map source codec → NVENC equivalent
    nvenc_map = {
        "h264": "h264_nvenc",
        "hevc": "hevc_nvenc",
        "av1": "av1_nvenc",
    }

    # Normalize codec name (e.g. "av1" or "libdav1d" → "av1")
    base_codec = vcodec
    for prefix in ("lib",):
        if base_codec.startswith(prefix):
            base_codec = base_codec[len(prefix):]

    nvenc_codec = nvenc_map.get(base_codec)

    if nvenc_codec:
        try:
            codec = av.codec.Codec(nvenc_codec, "w")
            logger.info(f"  Using GPU-accelerated encoder: {nvenc_codec}")
            # NVENC options: use 'p1' preset (fastest) + 'hq' tune
            # 'preset' for nvenc uses different values than libx264
            codec_options = {
                "preset": "p1",       # Fastest NVENC preset
                "tune": "hq",         # High quality
                "rc": "vbr",          # Variable bitrate
                "cq": "23",           # Quality target (like CRF)
            }
            return nvenc_codec, codec_options
        except Exception:
            logger.info(f"  NVENC codec {nvenc_codec} not available, using software encoding")

    # Fallback: software encoding
    codec_options = {
        "crf": "23",
        "preset": "ultrafast",
    }
    return vcodec, codec_options


def _subsample_video_with_av(
    input_path: Path,
    output_path: Path,
    subsample_factor: int,
    target_fps: int,
    vcodec: str,
    pix_fmt: str,
    episodes: list[int],
    src_episodes: list[dict],
    use_gpu: bool = True,
) -> None:
    """Re-encode a video file with subsampled frame rate using PyAV.

    Keeps every Nth frame and re-encodes at the new FPS.
    Uses GPU-accelerated NVENC encoding when available.
    """
    fps_fraction = Fraction(target_fps).limit_denominator(1000)

    in_container = av.open(str(input_path))
    if not in_container.streams.video:
        logger.warning(f"No video streams in {input_path}, skipping")
        in_container.close()
        return

    v_in = in_container.streams.video[0]

    # Select codec: GPU (NVENC) if available, else software
    if use_gpu:
        out_codec, codec_options = _get_nvenc_codec(vcodec)
    else:
        out_codec = vcodec
        codec_options = {
            "crf": "23",
            "preset": "ultrafast",
        }

    out = av.open(str(output_path), mode="w")
    v_out = out.add_stream(out_codec, rate=fps_fraction, options=codec_options)
    v_out.width = v_in.codec_context.width
    v_out.height = v_in.codec_context.height
    # NVENC requires nv12 format (yuv420p equivalent for GPU)
    if "_nvenc" in out_codec:
        v_out.pix_fmt = "nv12"
    else:
        v_out.pix_fmt = pix_fmt
    v_out.time_base = Fraction(1, target_fps)

    out.start_encoding()

    src_frame_count = 0
    dst_frame_count = 0

    for packet in in_container.demux(v_in):
        for frame in packet.decode():
            if frame is None:
                continue

            # Keep every Nth frame
            if src_frame_count % subsample_factor == 0:
                new_frame = frame.reformat(
                    width=v_out.width, height=v_out.height, format=v_out.pix_fmt
                )
                new_frame.pts = dst_frame_count
                new_frame.time_base = Fraction(1, target_fps)

                for pkt in v_out.encode(new_frame):
                    out.mux(pkt)

                dst_frame_count += 1

            src_frame_count += 1

    # Flush encoder
    for pkt in v_out.encode():
        out.mux(pkt)

    out.close()
    in_container.close()


def _save_subsampled_episodes_metadata(
    src_meta: LeRobotDatasetMetadata,
    dst_meta: LeRobotDatasetMetadata,
    src_episodes: list[dict],
    episode_data_metadata: dict[int, dict],
    episode_video_metadata: dict[int, dict],
    subsample_factor: int,
    target_fps: int,
) -> None:
    """Save per-episode metadata for the downsampled dataset."""
    for ep_idx, src_ep in enumerate(src_episodes):
        new_length = src_ep["length"] // subsample_factor

        # Build episode metadata dict
        ep_meta = {
            "episode_index": ep_idx,
            "length": new_length,
        }

        # Add data metadata
        if ep_idx in episode_data_metadata:
            ep_meta.update(episode_data_metadata[ep_idx])

        # Add video metadata
        if ep_idx in episode_video_metadata:
            ep_meta.update(episode_video_metadata[ep_idx])

        # Compute episode stats from the downsampled data
        # (We need to load the actual downsampled frames)
        episode_stats = _compute_subsampled_episode_stats(
            dst_meta, ep_idx, new_length
        )

        # Get task for this episode
        task_name = src_meta.tasks.iloc[src_ep.get("task_index", 0)].name
        episode_tasks = [task_name]

        dst_meta.save_episode(
            episode_index=ep_idx,
            episode_length=new_length,
            episode_tasks=episode_tasks,
            episode_stats=episode_stats,
            episode_metadata=ep_meta,
        )

    dst_meta.finalize()


def _compute_subsampled_episode_stats(
    dst_meta: LeRobotDatasetMetadata,
    episode_index: int,
    episode_length: int,
) -> dict[str, dict]:
    """Compute stats for a downsampled episode from its parquet data."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Temporarily load just this episode's data
    try:
        # Read the parquet data for this episode
        ep_info = dst_meta.episodes[episode_index] if dst_meta.episodes else None
        if ep_info is None:
            # Fallback: return empty stats
            return {}

        chunk_idx = ep_info.get("data/chunk_index", 0)
        file_idx = ep_info.get("data/file_index", 0)
        parquet_path = dst_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=chunk_idx, file_index=file_idx
        )

        if not parquet_path.exists():
            return {}

        df = pd.read_parquet(parquet_path)
        ep_df = df[df["episode_index"] == episode_index]

        if len(ep_df) == 0:
            return {}

        # Compute stats for each non-video feature
        stats = {}
        for key in dst_meta.features:
            feat = dst_meta.features[key]
            if feat.get("dtype") in ("video", "image"):
                continue
            if key not in ep_df.columns:
                continue

            values = np.stack(ep_df[key].values)
            stats[key] = {
                "mean": np.atleast_1d(values.mean(axis=0)),
                "std": np.atleast_1d(values.std(axis=0)),
                "min": np.atleast_1d(values.min(axis=0)),
                "max": np.atleast_1d(values.max(axis=0)),
                "count": np.array([len(values)]),
            }

        return stats

    except Exception as e:
        logger.warning(f"Failed to compute stats for episode {episode_index}: {e}")
        return {}


def _recompute_stats(dst_meta: LeRobotDatasetMetadata, src_meta: LeRobotDatasetMetadata | None = None) -> None:
    """Recompute dataset statistics from the downsampled data.

    Uses LeRobot's built-in recompute_stats to ensure correct types,
    then copies image/video stats from source (pixel stats don't change with fps).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import recompute_stats as _recompute_stats_tool

    dst_meta.ensure_readable()

    # Load the dataset and use LeRobot's recompute_stats (ensures correct types/percentiles)
    try:
        ds = LeRobotDataset(
            repo_id=dst_meta.repo_id,
            root=dst_meta.root,
        )
        _recompute_stats_tool(ds, skip_image_video=True)
        logger.info("Stats recomputed using LeRobot's built-in tool")
    except Exception as e:
        logger.warning(f"Built-in stats recomputation failed ({e}), using manual computation")

        # Manual fallback
        all_stats = []
        for ep_idx in range(dst_meta.total_episodes):
            ep = dst_meta.episodes[ep_idx]
            chunk_idx = ep["data/chunk_index"]
            file_idx = ep["data/file_index"]
            parquet_path = dst_meta.root / DEFAULT_DATA_PATH.format(
                chunk_index=chunk_idx, file_index=file_idx
            )

            if not parquet_path.exists():
                continue

            df = pd.read_parquet(parquet_path)
            ep_df = df[df["episode_index"] == ep_idx]

            if len(ep_df) == 0:
                continue

            ep_stats = {}
            for key in dst_meta.features:
                feat = dst_meta.features[key]
                if feat.get("dtype") in ("video", "image"):
                    continue
                if key not in ep_df.columns:
                    continue

                try:
                    values = np.stack(ep_df[key].values)
                    ep_stats[key] = {
                        "mean": np.atleast_1d(values.mean(axis=0)),
                        "std": np.atleast_1d(values.std(axis=0)),
                        "min": np.atleast_1d(values.min(axis=0)),
                        "max": np.atleast_1d(values.max(axis=0)),
                        "count": np.array([len(values)]),
                    }
                except Exception as e:
                    logger.warning(f"Failed to compute stats for {key} in episode {ep_idx}: {e}")

            if ep_stats:
                all_stats.append(ep_stats)

        if all_stats:
            final_stats = aggregate_stats(all_stats)
            write_stats(final_stats, dst_meta.root)
            dst_meta.stats = final_stats
        else:
            logger.warning("No episode stats could be computed")

    # Copy image/video stats from source dataset (pixel stats don't change with fps)
    if src_meta is not None:
        import json
        from copy import deepcopy
        from lerobot.datasets.io_utils import load_stats

        src_stats = load_stats(src_meta.root)
        dst_stats = load_stats(dst_meta.root)

        updated = False
        for key in src_meta.video_keys + src_meta.image_keys:
            if key in src_stats and key not in dst_stats:
                dst_stats[key] = deepcopy(src_stats[key])
                logger.info(f"  Copied image/video stats for '{key}' from source")
                updated = True

        # Also copy percentile stats (q01, q10, q50, q90, q99) if missing
        percentile_keys = ["q01", "q10", "q50", "q90", "q99"]
        for key in dst_stats:
            if key in src_stats:
                for pk in percentile_keys:
                    if pk in src_stats[key] and pk not in dst_stats[key]:
                        dst_stats[key][pk] = deepcopy(src_stats[key][pk])
                        updated = True

        if updated:
            write_stats(dst_stats, dst_meta.root)
            dst_meta.stats = dst_stats


def main():
    parser = argparse.ArgumentParser(
        description="Downsample a LeRobot dataset to a lower FPS"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Source dataset repo_id (e.g. local/nexarm_smolvla_train)",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        required=True,
        help="Target FPS (must be an integer divisor of source FPS, e.g. 15 for 30→15)",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        default=None,
        help="Output dataset repo_id (default: {repo_id}_{target_fps}fps)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory (default: $HF_LEROBOT_HOME/{output_repo_id})",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview changes without writing files",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU-accelerated video encoding (use software encoding instead)",
    )

    args = parser.parse_args()

    downsample_dataset_fps(
        repo_id=args.repo_id,
        target_fps=args.target_fps,
        output_repo_id=args.output_repo_id,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        use_gpu=not args.no_gpu,
    )


if __name__ == "__main__":
    main()
