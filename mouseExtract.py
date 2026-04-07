"""
Video Trajectory Extraction Tool

Features:
1. Load video and trajectory data
2. Find continuous frame start point (video recording start)
3. Use YOLO to fill missing frames when offset is negative
4. Manual alignment (A/D to adjust offset)
5. Extract fixed-size clips around trajectory
6. Edge padding support
7. Stream processing for long videos
8. Output to specified folder

Frame mapping: trajectory_points[i] corresponds to video frame i
"""

import os
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Optional
from ultralytics import YOLO


# ==================== Config ====================
@dataclass
class ExtractConfig:
    """Video extraction config"""
    clip_size: int = 256
    pad_mode: str = 'constant'    # 'constant', 'edge', 'reflect', 'wrap'
    pad_value: int = 0
    output_fps: int = 50
    output_suffix: str = '_clip'
    output_codec: str = 'mp4v'    # 'avc1', 'mp4v', 'XVID', 'MJPG'
    output_dir: Optional[str] = None
    yolo_model_path: Optional[str] = None
    yolo_conf: float = 0.5
    yolo_fill_max_frames: int = 100


# ==================== Data Loading ====================
def load_and_find_trajectory_start(trajectory_path, consecutive_threshold=10):
    """
    Load trajectory and find continuous frame start point

    Returns:
        trajectory_points: interpolated trajectory
        camera_start_frame: camera exposure frame number when video recording starts
    """
    df = pd.read_csv(trajectory_path, sep=',', names=['x','y','syncInd','time','frameInd'], dtype=int)
    df = df.drop_duplicates(subset='frameInd', keep='first').sort_values('frameInd')

    frame_inds = df['frameInd'].values
    x_vals = df['x'].values.astype(float)
    y_vals = df['y'].values.astype(float)

    # Find continuous increasing start point
    video_start_idx = 0
    for i in range(consecutive_threshold, len(frame_inds)):
        consecutive_count = 0
        for j in range(1, consecutive_threshold + 1):
            if frame_inds[i - j + 1] - frame_inds[i - j] == 1:
                consecutive_count += 1

        if consecutive_count >= consecutive_threshold:
            video_start_idx = i - consecutive_threshold
            break

    frame_inds = frame_inds[video_start_idx:]
    x_vals = x_vals[video_start_idx:]
    y_vals = y_vals[video_start_idx:]

    camera_start_frame = frame_inds[0]

    f_x = interp1d(frame_inds, x_vals, kind='linear', fill_value="extrapolate")
    f_y = interp1d(frame_inds, y_vals, kind='linear', fill_value="extrapolate")

    interpolated_frames = np.arange(frame_inds[0], frame_inds[-1] + 1)
    x_interp = f_x(interpolated_frames)
    y_interp = f_y(interpolated_frames)

    trajectory_points = np.vstack((x_interp, y_interp)).T

    print(f"Camera exposure frame range: {camera_start_frame} - {frame_inds[-1]}")
    print(f"Trajectory points: {len(trajectory_points)}")

    return trajectory_points, camera_start_frame


def yolo_fill_negative_offset(video_path: str, trajectory_points: np.ndarray,
                               offset: int, config: ExtractConfig) -> np.ndarray:
    """Fill missing frames at the beginning using YOLO"""
    if offset >= 0:
        return trajectory_points

    if config.yolo_model_path is None or not os.path.exists(config.yolo_model_path):
        print(f"Negative offset ({offset}) needs YOLO, but model not provided")
        return trajectory_points

    fill_count = -offset
    if fill_count > config.yolo_fill_max_frames:
        print(f"Offset too large, max fill {config.yolo_fill_max_frames}")
        fill_count = config.yolo_fill_max_frames

    print(f"Using YOLO to fill first {fill_count} frames...")
    model = YOLO(config.yolo_model_path)

    cap = cv2.VideoCapture(video_path)
    start_positions = []

    for _ in range(fill_count):
        ret, frame = cap.read()
        if not ret:
            start_positions.append([np.nan, np.nan])
            continue

        results = model(frame, verbose=False, conf=config.yolo_conf, task="detect")

        detected = False
        for result in results:
            if len(result.boxes):
                box = result.boxes[0]
                xyxy = np.array(box.xyxy[0].tolist(), int)
                x_center = (xyxy[0] + xyxy[2]) * 0.5
                y_center = (xyxy[1] + xyxy[3]) * 0.5
                start_positions.append([x_center, y_center])
                detected = True
                break

        if not detected:
            if len(start_positions) > 0 and not np.isnan(start_positions[-1]).all():
                start_positions.append(start_positions[-1].copy())
            elif len(trajectory_points) > 0:
                start_positions.append(trajectory_points[0].copy())
            else:
                start_positions.append([0, 0])

    cap.release()

    # Interpolate to fill NaN
    start_positions = np.array(start_positions)
    valid_indices = ~np.isnan(start_positions).any(axis=1)

    if np.sum(valid_indices) > 1:
        valid_x = start_positions[valid_indices, 0]
        valid_y = start_positions[valid_indices, 1]
        valid_idx = np.where(valid_indices)[0]

        f_x = interp1d(valid_idx, valid_x, kind='linear', fill_value="extrapolate")
        f_y = interp1d(valid_idx, valid_y, kind='linear', fill_value="extrapolate")

        all_idx = np.arange(len(start_positions))
        start_positions[:, 0] = f_x(all_idx)
        start_positions[:, 1] = f_y(all_idx)
    elif len(trajectory_points) > 0:
        for i in range(len(start_positions)):
            if np.isnan(start_positions[i]).any():
                start_positions[i] = trajectory_points[0]

    filled_trajectory = np.vstack([start_positions, trajectory_points])
    print(f"Filled trajectory length: {len(filled_trajectory)}")

    return filled_trajectory


def align_trajectory_to_video_frames(trajectory_points: np.ndarray,
                                      video_frame_offset: int,
                                      total_video_frames: int) -> np.ndarray:
    """
    Align trajectory to video frames

    Args:
        trajectory_points: trajectory array
        video_frame_offset: trajectory_points[0] corresponds to video frame at this offset
        total_video_frames: total video frames
    """
    aligned_length = total_video_frames
    aligned_trajectory = np.zeros((aligned_length, 2))

    for j in range(aligned_length):
        traj_idx = j - video_frame_offset

        if 0 <= traj_idx < len(trajectory_points):
            aligned_trajectory[j] = trajectory_points[traj_idx]
        elif traj_idx < 0:
            aligned_trajectory[j] = trajectory_points[0]
        else:
            aligned_trajectory[j] = trajectory_points[-1]

    print(f"Aligned trajectory length: {len(aligned_trajectory)}")

    return aligned_trajectory


# ==================== Video Extraction ====================
def extract_crop_from_frame(frame: np.ndarray, center_x: float, center_y: float,
                            clip_size: int, pad_mode: str = 'constant',
                            pad_value: int = 0) -> np.ndarray:
    half_size = clip_size // 2
    x1 = int(center_x) - half_size
    y1 = int(center_y) - half_size
    x2 = x1 + clip_size
    y2 = y1 + clip_size

    h, w = frame.shape[:2]

    # Clamp crop region to frame bounds
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    # Extract valid region
    src_crop = frame[src_y1:src_y2, src_x1:src_x2].copy()

    # Calculate padding needed
    pad_top = src_y1 - y1
    pad_bottom = y2 - src_y2
    pad_left = src_x1 - x1
    pad_right = x2 - src_x2

    # Add border to get exactly clip_size x clip_size
    crop = cv2.copyMakeBorder(
        src_crop, pad_top, pad_bottom, pad_left, pad_right,
        borderType={
            'constant': cv2.BORDER_CONSTANT,
            'edge': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT_101,
            'wrap': cv2.BORDER_WRAP
        }[pad_mode],
        value=pad_value if pad_mode == 'constant' else 0
    )

    return crop


def extract_clips_streaming(video_path: str, trajectory_points: np.ndarray,
                             config: ExtractConfig,
                             start_frame: int = 0,
                             end_frame: Optional[int] = None,
                             show_progress: bool = True) -> str:
    """Stream processing video extraction"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
        output_path = os.path.join(config.output_dir, f"{base_name}{config.output_suffix}.mp4")
    else:
        output_path = os.path.join(os.path.dirname(video_path), f"{base_name}{config.output_suffix}.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return ""

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.output_fps

    if end_frame is None:
        end_frame = total_frames

    process_frame_count = end_frame - start_frame

    print(f"Video: {total_frames} frames, {fps:.2f} fps")
    print(f"Processing frames {start_frame} - {end_frame}")
    print(f"Output: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*config.output_codec)
    out = cv2.VideoWriter(output_path, fourcc, config.output_fps, (config.clip_size, config.clip_size))

    if not out.isOpened():
        print(f"Error: Cannot open video writer for {output_path}")
        cap.release()
        return ""

    print(f"Video writer: {config.clip_size}x{config.clip_size}, {config.output_fps} fps, codec: {config.output_codec}")
    cv2.setNumThreads(1)

    processed_count = 0
    next_print = 500

    while processed_count < process_frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        video_frame_idx = start_frame + processed_count

        if 0 <= video_frame_idx < len(trajectory_points):
            center = trajectory_points[video_frame_idx]
        else:
            valid_idx = max(0, min(video_frame_idx, len(trajectory_points) - 1))
            center = trajectory_points[valid_idx]

        crop = extract_crop_from_frame(
            frame, center[0], center[1],
            config.clip_size, config.pad_mode, config.pad_value
        )

        if crop is None or crop.size == 0:
            print(f"Warning: Empty crop at frame {video_frame_idx}")
            continue

        if crop.shape != (config.clip_size, config.clip_size, 3):
            print(f"Warning: Invalid crop shape {crop.shape} at frame {video_frame_idx}")
            continue

        out.write(crop)

        processed_count += 1
        if show_progress and processed_count >= next_print:
            print(f"Progress: {processed_count}/{process_frame_count}")
            next_print += 500

    cap.release()
    out.release()

    print(f"Done! Processed {processed_count} frames")

    return output_path


# ==================== Manual Alignment ====================
def visualize_and_align(video_path: str, trajectory_points: np.ndarray,
                       camera_start_frame: int,
                       max_frame_count: int = 30, interval: int = 5,
                       initial_offset: int = 0,
                       clip_size: int = 256, pad_mode: str = 'edge') -> int:
    """
    Manual alignment interface

    Args:
        video_path: video file path
        trajectory_points: trajectory[i] corresponds to camera frame (camera_start_frame + i)
        camera_start_frame: camera exposure frame when trajectory starts
        initial_offset: trajectory_points[0] corresponds to video frame at this offset

    Returns:
        final_offset: trajectory_points[0] corresponds to video frame at final_offset
    """
    # Read video frames from START (frame 0), sampling by interval
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    frame_count = 0

    # Read frames with interval sampling
    max_frames_to_read = (max_frame_count + abs(initial_offset) + 100) * interval

    while frame_count < max_frames_to_read:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()

    if not frames:
        print("No frames read")
        return initial_offset

    frame_count_limit = max_frame_count
    current_offset = initial_offset

    print("A/D: adjust offset | Space: preview | Enter: confirm | Q: exit")
    print(f"Camera start frame: {camera_start_frame}, offset: {current_offset}")

    while True:
        # FIXED: Always display first N frames (0 to frame_count_limit * interval)
        # so trajectory shifts relative to video as offset changes
        start_idx = 0
        end_idx = min(len(frames), frame_count_limit)

        # Get frames for display
        display_frames = frames[start_idx:end_idx]

        # Create motion trail effect (average)
        if len(display_frames) > 0:
            base_image = np.mean(display_frames, axis=0).astype(np.uint8)
        else:
            base_image = frames[0].copy()

        # Calculate trajectory point range to display
        # Displayed sampled frames: start_idx to end_idx-1 (FIXED range)
        # Corresponding video frames: start_idx*interval to end_idx*interval
        # Video frame j corresponds to trajectory index: (j - current_offset)
        traj_start = (start_idx * interval) - current_offset
        traj_end = (end_idx * interval) - current_offset
        traj_start = max(0, traj_start)
        traj_end = min(len(trajectory_points), traj_end)

        # Draw trajectory points
        for i in range(traj_start, traj_end):
            if 0 <= i < len(trajectory_points):
                point = trajectory_points[i]
                x, y = int(point[0]), int(point[1])
                cv2.circle(base_image, (x, y), 3, (255, 0, 0), -1)
                cv2.circle(base_image, (x, y), 3, (0, 0, 0), 1)

                if i > traj_start:
                    prev_point = trajectory_points[i - 1]
                    px, py = int(prev_point[0]), int(prev_point[1])
                    cv2.line(base_image, (px, py), (x, y), (0, 255, 0), 2)

        # Display only offset
        cv2.putText(base_image, f"offset: {current_offset}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Alignment", base_image)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            print("Alignment cancelled")
            quit()
        elif key == ord('q'):
            break
        elif key == ord('a'):
            current_offset -= 1
        elif key == ord('d'):
            current_offset += 1
        elif key == 13:  # Enter
            print(f"Confirmed offset: {current_offset}")
            break
        elif key == ord(' '):  # Preview cropped view
            cap = cv2.VideoCapture(video_path)
            preview_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for preview_frame_idx in range(0, preview_total, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                traj_idx = preview_frame_idx - current_offset

                # Get valid trajectory index
                if 0 <= traj_idx < len(trajectory_points):
                    center = trajectory_points[traj_idx]
                else:
                    valid_idx = max(0, min(traj_idx, len(trajectory_points) - 1))
                    center = trajectory_points[valid_idx]

                # Extract crop using configured parameters
                crop = extract_crop_from_frame(
                    frame, center[0], center[1],
                    clip_size, pad_mode, 0
                )

                cv2.putText(crop, f"F: {preview_frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(crop, f"T: {traj_idx}", (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(crop, f"offset: {current_offset}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.imshow("Preview Crop (Q=exit)", crop)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyWindow("Preview Crop (Q=exit)")

    cv2.destroyAllWindows()
    return current_offset


# ==================== Main ====================
def main():
    # File paths
    video_path = r"E:\pythonFiles\YoloV8\tempScreen\12_26_1907outputraw.mp4"
    trajectory_path = r"E:\pythonFiles\YoloV8\tempScreen\12_26_1907mousePosRec.txt"
    yolo_model_path = r"E:\pythonFiles\YoloV8\models\TopViewMiniscopeBodyBestWithAddition.pt"

    # Config
    config = ExtractConfig(
        clip_size=180,
        pad_mode='edge',
        output_fps=50,
        output_suffix='_clip',
        output_codec='mp4v',  # Use mp4v, XVID, or MJPG
        output_dir=r"E:\pythonFiles\YoloV8\output_clips",
        yolo_model_path=yolo_model_path,
        yolo_conf=0.5,
        yolo_fill_max_frames=100
    )

    initial_offset = 23
    enable_manual_align = False
    process_start_frame = 0
    process_end_frame = None

    # Load data
    print("=" * 50)
    print("Loading data...")
    print("=" * 50)

    trajectory_points, camera_start_frame = load_and_find_trajectory_start(
        trajectory_path, consecutive_threshold=10
    )

    # Manual alignment
    if enable_manual_align:
        print("\n" + "=" * 50)
        print("Manual alignment")
        print("=" * 50)

        final_offset = visualize_and_align(
            video_path=video_path,
            trajectory_points=trajectory_points,
            camera_start_frame=camera_start_frame,
            max_frame_count=20,
            interval=5,
            initial_offset=initial_offset,
            clip_size=config.clip_size,
            pad_mode=config.pad_mode
        )
    else:
        final_offset = initial_offset

    # Process alignment
    print("\n" + "=" * 50)
    print("Processing alignment...")
    print("=" * 50)

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Negative offset: fill with YOLO
    if final_offset < 0:
        print(f"Negative offset ({final_offset}), filling with YOLO...")
        trajectory_points = yolo_fill_negative_offset(
            video_path=video_path,
            trajectory_points=trajectory_points,
            offset=final_offset,
            config=config
        )
        final_offset = 0

    # Align trajectory to video frames
    trajectory_points = align_trajectory_to_video_frames(
        trajectory_points=trajectory_points,
        video_frame_offset=final_offset,
        total_video_frames=total_video_frames
    )

    # Extract video
    print("\n" + "=" * 50)
    print("Extracting video...")
    print("=" * 50)

    output_path = extract_clips_streaming(
        video_path=video_path,
        trajectory_points=trajectory_points,
        config=config,
        start_frame=process_start_frame,
        end_frame=process_end_frame,
        show_progress=True
    )

    print("\n" + "=" * 50)
    print(f"Done! Output: {output_path}")
    print(f"Frames: {total_video_frames}")
    print("=" * 50)


if __name__ == "__main__":
    main()
