"""
Auto-discover video and trajectory file pairs from directory

Usage:
    python discover_and_run.py --dir /path/to/files --output batch_config.yaml
    python discover_and_run.py --dir /path/to/files --run --workers 4
"""

import os
import re
import argparse
import yaml
from pathlib import Path
from typing import List, Tuple


def extract_time_prefix(filename: str) -> str:
    """
    Extract time prefix from filename
    Examples:
        12_26_1907outputraw.mp4 -> 12_26_1907
        12_26_1907mousePosRec.txt -> 12_26_1907
        01_17_1842outputraw.mp4 -> 01_17_1842
    """
    # Match pattern: digits_digits_digits (like 12_26_1907)
    match = re.search(r'(\d{2}_\d{2}_\d{4})', filename)
    if match:
        return match.group(1)
    return None


def discover_pairs(directory: str, video_pattern: str = None, traj_pattern: str = None) -> List[Tuple[str, str]]:
    """
    Discover video and trajectory file pairs based on time prefix

    Args:
        directory: directory to search
        video_pattern: optional pattern to filter videos (e.g., "*.mp4")
        traj_pattern: optional pattern to filter trajectories (default: "*mousePosRec.txt")

    Returns:
        List of (video_path, trajectory_path) tuples
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    # Default patterns
    if video_pattern is None:
        video_pattern = "*.mp4"
    if traj_pattern is None:
        traj_pattern = "*mousePosRec.txt"

    # Find all trajectory files
    traj_files = list(dir_path.glob(traj_pattern))
    if not traj_files:
        # Try alternative pattern
        traj_files = list(dir_path.glob("*mousePos*.txt"))
        if not traj_files:
            traj_files = list(dir_path.glob("*.txt"))

    # Group by time prefix
    traj_dict = {}  # prefix -> trajectory path
    for traj_file in traj_files:
        prefix = extract_time_prefix(traj_file.name)
        if prefix:
            traj_dict[prefix] = str(traj_file)

    # Find matching video files
    video_files = list(dir_path.glob(video_pattern))
    video_dict = {}  # prefix -> video path
    for video_file in video_files:
        prefix = extract_time_prefix(video_file.name)
        if prefix:
            video_dict[prefix] = str(video_file)

    # Match pairs
    pairs = []
    for prefix, traj_path in traj_dict.items():
        if prefix in video_dict:
            pairs.append((video_dict[prefix], traj_path))
        else:
            print(f"Warning: No video found for trajectory: {traj_path}")

    # Also check for videos without trajectories
    for prefix, video_path in video_dict.items():
        if prefix not in traj_dict:
            print(f"Warning: No trajectory found for video: {video_path}")

    # Sort by prefix
    pairs.sort(key=lambda x: extract_time_prefix(Path(x[0]).name))

    return pairs


def generate_config(pairs: List[Tuple[str, str]], output_path: str,
                    clip_size: int = 180,
                    pad_mode: str = 'edge',
                    output_fps: int = 50,
                    output_codec: str = 'mp4v',
                    output_dir: str = './output_clips',
                    yolo_model: str = './models/TopViewMiniscopeBodyBestWithAddition.pt',
                    enable_auto_align: bool = True,
                    enable_preview: bool = True) -> None:
    """Generate YAML config file from discovered pairs"""

    config_data = {
        'config': {
            'clip_size': clip_size,
            'pad_mode': pad_mode,
            'pad_value': 0,
            'output_fps': output_fps,
            'output_suffix': '_clip',
            'output_codec': output_codec,
            'output_dir': output_dir,
            'yolo_model_path': yolo_model,
            'yolo_conf': 0.5,
            'yolo_fill_max_frames': 100,
            'enable_auto_align': enable_auto_align,
            'auto_align_frames': 20,
            'auto_align_max_offset': 50,
            'enable_preview': enable_preview,
            'preview_frames': 10
        },
        'videos': []
    }

    for video_path, traj_path in pairs:
        config_data['videos'].append({
            'video': video_path,
            'trajectory': traj_path,
            'offset': 0  # Use auto-detect
        })

    with open(output_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    print(f"Generated config: {output_path} ({len(pairs)} videos)")


def run_batch_direct(pairs: List[Tuple[str, str]], workers: int = None,
                     clip_size: int = 180,
                     pad_mode: str = 'edge',
                     output_dir: str = './output_clips',
                     yolo_model: str = './models/TopViewMiniscopeBodyBestWithAddition.pt',
                     enable_auto_align: bool = True,
                     enable_preview: bool = True) -> None:
    """Run batch processing directly without config file"""
    import subprocess
    import sys

    videos_str = ','.join(p[0] for p in pairs)
    trajs_str = ','.join(p[1] for p in pairs)
    offsets_str = ','.join(['0'] * len(pairs))

    cmd = [
        sys.executable, 'mouseExtractBatch.py',
        '--videos', videos_str,
        '--traj', trajs_str,
        '--offsets', offsets_str,
        '--clip-size', str(clip_size),
        '--pad-mode', pad_mode,
        '--output-dir', output_dir,
        '--yolo-model', yolo_model
    ]

    if enable_auto_align:
        cmd.append('--auto-align-frames')
        cmd.append('20')
    else:
        cmd.append('--no-auto-align')

    if enable_preview:
        cmd.append('--preview')
        cmd.append('--preview-frames')
        cmd.append('10')

    if workers:
        cmd.extend(['--workers', str(workers)])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description='Discover and batch process video/trajectory pairs')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing files')
    parser.add_argument('--output', type=str, default='batch_config.yaml', help='Output config file')
    parser.add_argument('--run', action='store_true', help='Run batch processing after discovery')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers for batch processing')

    # Filter options
    parser.add_argument('--video-pattern', type=str, default='*.mp4', help='Video file pattern')
    parser.add_argument('--traj-pattern', type=str, default='*mousePosRec.txt', help='Trajectory file pattern')

    # Config options
    parser.add_argument('--clip-size', type=int, default=180, help='Crop size')
    parser.add_argument('--pad-mode', type=str, default='edge', help='Padding mode')
    parser.add_argument('--output-fps', type=int, default=50, help='Output FPS')
    parser.add_argument('--output-dir', type=str, default='./output_clips', help='Output directory')
    parser.add_argument('--yolo-model', type=str, default='./models/TopViewMiniscopeBodyBestWithAddition.pt', help='YOLO model path')
    parser.add_argument('--no-auto-align', action='store_true', help='Disable auto-alignment')
    parser.add_argument('--no-preview', action='store_true', help='Disable preview generation')

    args = parser.parse_args()

    # Discover pairs
    print(f"Searching in: {args.dir}")
    pairs = discover_pairs(args.dir, args.video_pattern, args.traj_pattern)

    if not pairs:
        print("No video/trajectory pairs found!")
        return

    print(f"\nFound {len(pairs)} pairs:")
    for video, traj in pairs:
        print(f"  {Path(video).name} <-> {Path(traj).name}")

    # Generate config
    generate_config(
        pairs, args.output,
        clip_size=args.clip_size,
        pad_mode=args.pad_mode,
        output_fps=args.output_fps,
        output_dir=args.output_dir,
        yolo_model=args.yolo_model,
        enable_auto_align=not args.no_auto_align,
        enable_preview=not args.no_preview
    )

    # Run if requested
    if args.run:
        print("\nRunning batch processing...")
        run_batch_direct(
            pairs, args.workers,
            clip_size=args.clip_size,
            pad_mode=args.pad_mode,
            output_dir=args.output_dir,
            yolo_model=args.yolo_model,
            enable_auto_align=not args.no_auto_align,
            enable_preview=not args.no_preview
        )


if __name__ == "__main__":
    main()
