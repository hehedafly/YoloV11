import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from ultralytics import YOLO

def visualize_motion_trail_with_alignment(video_path, trajectory_points, head_orientations,
                                          max_frame_count=30, interval=5, offset=0, head_offset=0,
                                          model_path=None, yolo_conf=0.7, yaw_offset=0,
                                          original_trajectory_points=None):
    """
    可视化运动轨迹并与头部方向对齐
    
    Args:
        video_path: 视频文件路径
        trajectory_points: 轨迹点列表 [(x1, y1), (x2, y2), ...]
        head_orientations: 头部方向欧拉角列表 [(yaw1, pitch1, roll1), ...]
        max_frame_count: 最大帧数
        interval: 帧采样间隔
        offset: 轨迹偏移初始值
        head_offset: 头部方向偏移初始值
        model_path: YOLO模型路径（可选，用于补充缺失轨迹点）
        yolo_conf: YOLO检测置信度阈值
    """
    
    # 读取视频帧
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            frames.append(frame)
        
        frame_count += 1
        if frame_count > max_frame_count * 3:
            break
    
    cap.release()
    
    if not frames:
        print("没有读取到视频帧")
        return None
    
    if head_orientations is None:
        print("没有头部方向数据")
        return None

    # 处理轨迹点：如果offset为负数，可能需要用YOLO补充前几帧
    if original_trajectory_points is None:
        original_trajectory_points = trajectory_points.copy()
    yolo_positions = []
    
    if offset < 0 and model_path and os.path.exists(model_path):
        print(f"检测到负偏移量({offset})，尝试使用YOLO模型补充前{-offset}帧...")
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        
        for i in range(-offset):
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model(frame, verbose=False, conf=yolo_conf, task="detect")
            detected = False
            
            for result in results:
                if len(result.boxes):
                    box = result.boxes[0]
                    xyxy = np.array(box.xyxy[0].tolist(), int)
                    x_center = int((xyxy[0] + xyxy[2]) * 0.5)
                    y_center = int((xyxy[1] + xyxy[3]) * 0.5)
                    yolo_positions.append([x_center, y_center])
                    detected = True
                    break
            
            if not detected:
                # 如果没有检测到，使用第一个轨迹点或默认值
                if len(trajectory_points) > 0:
                    yolo_positions.append(trajectory_points[0])
                else:
                    yolo_positions.append([0, 0])
        
        cap.release()
        model = None
        
        # 将YOLO检测的位置插入到轨迹点前面
        if yolo_positions:
            trajectory_points = np.vstack([np.array(yolo_positions), trajectory_points])
            print(f"已用YOLO补充了{len(yolo_positions)}个轨迹点")
    
    frame_count_limit = max_frame_count // interval
    # 创建残影效果（简单平均）
    base_image = np.mean(frames[:frame_count_limit], axis=0).astype(np.uint8)
    
    # 复制一份原始残影用于重置
    # original_base = base_image.copy()
    
    current_offset = offset
    current_head_offset = head_offset

    while True:

        start_frame = min(max(0, -(current_offset // interval)), len(frames) - 1)
        end_frame = max(min(len(frames) - 1, start_frame + frame_count_limit), start_frame + 1)

        base_image = np.mean(frames[start_frame: end_frame], axis=0).astype(np.uint8)
        
        # 计算轨迹点的显示范围
        start_idx = max(0, -current_offset) if current_offset < 0 else 0
        # 修正：使用轨迹点的实际长度来计算结束索引
        max_display_points = min(frame_count_limit * interval, len(trajectory_points) - max(0, current_offset))
        end_idx = min(start_idx + max_display_points, len(trajectory_points))
        
        y_position = 30
        # 绘制轨迹点（带线段连接）
        for _i in range(end_idx -start_idx):
            # 计算当前帧对应的轨迹点索引
            i = _i + start_idx
            point_idx = i + max(0, current_offset)
            point = None
            if 0 <= point_idx < len(trajectory_points):
                point = trajectory_points[point_idx]
                x, y = int(point[0]), int(point[1])
                
                # 绘制点
                cv2.circle(base_image, (x, y), 3, (255, 0, 0), -1)
                cv2.circle(base_image, (x, y), 3, (0, 0, 0), 1)
                
                # 绘制连接线（如果不是第一个点）
                if i > start_idx:
                    prev_point_idx = i - 1 + max(0, current_offset)
                    if 0 <= prev_point_idx < len(trajectory_points):
                        prev_point = trajectory_points[prev_point_idx]
                        px, py = int(prev_point[0]), int(prev_point[1])
                        cv2.line(base_image, (px, py), (x, y), (0, 255, 0), 2)

            # 获取当前帧的头部方向信息（偏航角，即yaw）
            # 修正：确保头部方向索引有效
            head_index = _i + current_head_offset
            current_yaw = None
            
            if 0 <= head_index < len(head_orientations):
                # 假设头部方向欧拉角格式为 [yaw, pitch, roll]
                current_yaw = head_orientations[head_index]  # yaw是第一个元素
            
            # 添加偏移量信息文本

            if current_yaw is not None and point is not None:
                # 应用自动计算的角度偏差
                yaw_deg = np.degrees(current_yaw) + yaw_offset
                # 将角度归一化到[-180, 180]范围
                yaw_deg_normalized = ((yaw_deg + 180) % 360) - 180
                # 转换回弧度用于绘制箭头
                yaw_corrected = np.radians(yaw_deg_normalized)

                if i == end_idx - 1:
                    cv2.putText(base_image, f"yaw: {yaw_deg_normalized:.2f}", (10, y_position + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # 在最后一个轨迹点上添加方向指示器
                if end_idx > start_idx:
                    x, y = int(point[0]), int(point[1])

                    # 绘制方向箭头（使用校正后的角度）
                    arrow_length = 50
                    direction_x = x + int(arrow_length * np.sin(yaw_corrected))
                    direction_y = y + int(arrow_length * np.cos(yaw_corrected))

                    cv2.arrowedLine(base_image, (x, y), (direction_x, direction_y),
                                    (0, 255, 0), 1, tipLength=0.3)
            else:
                if i == end_idx - 1:
                    cv2.putText(base_image, "yaw: N/A", (10, y_position + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
        

        cv2.putText(base_image, f"track offset (A/D): {current_offset}", (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(base_image, f"head orientation offset (W/S): {current_head_offset}", (10, y_position + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(base_image, f"yaw offset: {yaw_offset:.1f}°", (10, y_position + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        # 添加使用说明
        cv2.putText(base_image, "A/D: track | W/S: head | Space: video | Enter: auto-align",
                    (10, y_position + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 显示图片
        cv2.imshow("Motion Trail Alignment with Head Yaw", base_image)
        
        # 等待按键
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC键
            break
        elif key == ord('a') or key == ord('d'):  # 调整轨迹偏移
            if key == ord('a'):  # 减少偏移（轨迹延后）
                current_offset -= 1
            elif key == ord('d'):  # 增加偏移（轨迹提前）
                current_offset += 1
            print(f"轨迹偏移量调整为: {current_offset}")
        elif key == ord('w') or key == ord('s'):  # 调整头部方向偏移
            if key == ord('w'):  # 减少头部偏移（头部方向延后）
                current_head_offset -= 1
            elif key == ord('s'):  # 增加头部偏移（头部方向提前）
                current_head_offset += 1
            print(f"头部方向偏移量调整为: {current_head_offset}")
        elif key == 13:  # Enter键自动对齐
            print("\n开始自动对齐...")
            best_offset, best_correlation, calculated_yaw_offset, _ = auto_align_head_offset_by_trajectory(
                original_trajectory_points,
                head_orientations,
                initial_head_offset=current_head_offset,
                search_range=200,
                match_window=100
            )
            current_head_offset = best_offset
            yaw_offset = calculated_yaw_offset  # 自动应用计算得到的角度偏差
            print(f"自动对齐完成，偏移量已更新为: {current_head_offset} (相关性: {best_correlation:.4f})")
            print(f"角度偏差已自动校正: {yaw_offset:.2f}°\n")

        elif key == ord(' '):  # 空格键预览视频
            cap = cv2.VideoCapture(video_path)
            if current_offset < 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, -current_offset)
            
            t_offset = max(0, current_offset)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                end_point = [-1, -1]
                # 绘制最近10个轨迹点
                for i in range(10):
                    if i + t_offset - 10 >= 0:
                        if i + t_offset - 10 < len(trajectory_points) and i + t_offset - 9 < len(trajectory_points):
                            start_point = tuple(np.array(trajectory_points[i - 10 + t_offset], dtype=int))
                            end_point = tuple(np.array(trajectory_points[i - 9 + t_offset], dtype=int))
                            cv2.line(frame, start_point, end_point, (0, 255, 0), thickness=2)
                
                # 显示当前头部方向（偏航角）
                head_idx = t_offset + current_head_offset
                if 0 <= head_idx < len(head_orientations):
                    yaw_raw = head_orientations[head_idx]
                    # 应用自动计算的角度偏差
                    yaw_deg = np.degrees(yaw_raw) + yaw_offset
                    yaw_deg_normalized = ((yaw_deg + 180) % 360) - 180
                    # 转换回弧度用于绘制箭头
                    yaw = np.radians(yaw_deg_normalized)

                    cv2.putText(frame, f"Yaw: {yaw_deg_normalized:.2f}°", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Offset: {yaw_offset:.1f}°", (10, 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

                    # 在视频帧上绘制方向指示器
                    if end_point[0] != -1:  # 如果有轨迹点
                        center_x, center_y = end_point
                        radius = 50

                        # 计算方向线（使用校正后的yaw）
                        direction_x = center_x + int(radius * np.sin(yaw))
                        direction_y = center_y - int(radius * np.cos(yaw))  # 注意：Y轴向下为正

                        # 绘制方向线
                        cv2.line(frame, (center_x, center_y), (direction_x, direction_y), (0, 255, 255), 3)

                        # 绘制方向箭头
                        arrow_length = 20
                        arrow_angle = 30 * np.pi / 180  # 30度
                        angle1 = yaw + np.pi - arrow_angle
                        angle2 = yaw + np.pi + arrow_angle

                        x1 = direction_x + int(arrow_length * np.sin(angle1))
                        y1 = direction_y - int(arrow_length * np.cos(angle1))
                        x2 = direction_x + int(arrow_length * np.sin(angle2))
                        y2 = direction_y - int(arrow_length * np.cos(angle2))

                        cv2.line(frame, (direction_x, direction_y), (x1, y1), (0, 255, 255), 2)
                        cv2.line(frame, (direction_x, direction_y), (x2, y2), (0, 255, 255), 2)
                
                t_offset += 1
                cv2.imshow("Video Preview with Yaw Indicator (Press Q to exit)", frame)
                key_preview = cv2.waitKey(10) & 0xFF
                if key_preview == ord('q'):
                    break
                elif key_preview == ord('e'):
                    yaw_offset += 1
                elif key_preview == ord('r'):
                    yaw_offset -= 1
            
            cap.release()
            cv2.destroyWindow("Video Preview with Yaw Indicator (Press Q to exit)")
    
    # 保存最终结果
    cv2.imwrite("aligned_motion_trail_with_yaw.png", base_image)
    print(f"已保存结果到 aligned_motion_trail_with_yaw.png")
    print(f"最终轨迹偏移量: {current_offset}")
    print(f"最终头部方向偏移量: {current_head_offset}")
    
    cv2.destroyAllWindows()
    
    return current_offset, current_head_offset


def load_head_orientation_csv(csv_path):
    """
    从CSV文件加载头部方向数据并转换为欧拉角
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        euler_angles: yaw
    """
    df = pd.read_csv(csv_path)
    
    # 自动识别四元数列
    quat_cols = []
    for col in df.columns:
        lower_col = col.lower()
        if 'x' in lower_col and 'y' not in lower_col and 'z' not in lower_col and 'w' not in lower_col:
            quat_cols.append(('x', col))
        elif 'y' in lower_col and 'x' not in lower_col and 'z' not in lower_col and 'w' not in lower_col:
            quat_cols.append(('y', col))
        elif 'z' in lower_col and 'x' not in lower_col and 'y' not in lower_col and 'w' not in lower_col:
            quat_cols.append(('z', col))
        elif 'w' in lower_col or 'scalar' in lower_col or 'real' in lower_col:
            quat_cols.append(('w', col))
    
    # 按xyzw顺序排序
    quat_cols.sort(key=lambda x: ['x','y','z','w'].index(x[0]))
    
    if len(quat_cols) < 4:
        print(f"警告: 只找到 {len(quat_cols)} 个四元数列，需要4个")
        # 尝试其他列名模式
        for col in df.columns:
            if 'quat' in col.lower() or 'rotation' in col.lower():
                print(f"可能的四元数列: {col}")
        return None
    
    quaternion_data = df[[col for _, col in quat_cols]].values
    
    # 将四元数转换为欧拉角
    # 使用'ZYX'顺序，这样第一个元素就是绕Z轴的旋转（偏航角yaw）
    rotations = R.from_quat(quaternion_data)
    euler_angles = rotations.as_euler('ZYX', degrees=False)  # 返回弧度值
    total_frames = df["Time Stamp (ms)"].values
    f_frame = interp1d(np.arange(len(total_frames)), total_frames, kind='linear', fill_value="extrapolate")
    scale = np.round((np.mean(total_frames[1:] - total_frames[:-1])) / 5) * 5 / (1000 / 50)
    interpolated_frames = f_frame(np.arange(len(total_frames) * scale) / scale)
    f_yaw = interp1d(total_frames, euler_angles[:, 0], kind='linear', fill_value="extrapolate")

    yaws = f_yaw(interpolated_frames)
    
    print(f"已加载 {len(euler_angles)} 个头部方向数据点")
    print(f"偏航角范围 (Yaw): [{np.degrees(euler_angles[:, 0].min()):.2f}, {np.degrees(euler_angles[:, 0].max()):.2f}]°")
    print(f"俯仰角范围 (Pitch): [{np.degrees(euler_angles[:, 1].min()):.2f}, {np.degrees(euler_angles[:, 1].max()):.2f}]°")
    print(f"横滚角范围 (Roll): [{np.degrees(euler_angles[:, 2].min()):.2f}, {np.degrees(euler_angles[:, 2].max()):.2f}]°")
    
    return yaws


def calculate_trajectory_direction(trajectory_points, window_size=5, min_speed_threshold=1.0):
    """
    计算轨迹的运动方向（以角度表示，弧度制）

    Args:
        trajectory_points: 轨迹点数组 (n, 2)
        window_size: 用于计算方向的滑动窗口大小
        min_speed_threshold: 最小速度阈值，低于此值时不计算方向（静止或转向时刻）

    Returns:
        directions: 运动方向数组（弧度），与轨迹点长度相同
        speeds: 速度数组
    """
    n = len(trajectory_points)
    directions = np.full(n, np.nan)
    speeds = np.zeros(n)

    for i in range(n):
        # 计算窗口范围
        start_idx = max(0, i - window_size // 2)
        end_idx = min(n, i + window_size // 2 + 1)

        # 如果窗口太小，跳过
        if end_idx - start_idx < 2:
            continue

        # 计算窗口内的位移
        window_points = trajectory_points[start_idx:end_idx]
        dx = window_points[-1, 0] - window_points[0, 0]
        dy = window_points[-1, 1] - window_points[0, 1]

        # 计算速度（每帧移动的距离）
        speed = np.sqrt(dx**2 + dy**2) / (end_idx - start_idx)
        speeds[i] = speed

        # 只有速度足够大时才计算方向（过滤静止或转向时刻）
        if speed >= min_speed_threshold:
            # 计算运动方向（弧度），注意Y轴向下为正
            direction = np.arctan2(dx, dy)  # 返回范围 [-pi, pi]
            directions[i] = direction

    return directions, speeds


def calculate_yaw_offset_from_alignment(trajectory_points, head_orientations, head_offset,
                                        match_window=100, high_speed_percentile=50):
    """
    计算头部方向与轨迹运动方向之间的固定角度偏差

    Args:
        trajectory_points: 轨迹点数组 (n, 2)
        head_orientations: 头部方向数组（yaw，弧度）
        head_offset: 头部方向数据的偏移量
        match_window: 用于计算偏差的窗口大小
        high_speed_percentile: 用于筛选高速运动帧的百分位数

    Returns:
        yaw_offset_deg: 角度偏差（度）
        confidence: 置信度（0-1）
    """
    # 计算轨迹运动方向
    traj_directions, traj_speeds = calculate_trajectory_direction(
        trajectory_points, window_size=5, min_speed_threshold=0.1
    )

    # 使用轨迹的前 match_window 帧
    traj_end = min(match_window, len(traj_directions))
    traj_dir_window = traj_directions[:traj_end]
    traj_speeds_window = traj_speeds[:traj_end]

    # 获取对应的头部方向数据
    head_start = head_offset
    head_end = min(head_offset + match_window, len(head_orientations))

    if head_end - head_start < 10:
        print("警告：有效数据不足，无法计算角度偏差")
        return 0.0, 0.0

    head_yaw_window = head_orientations[head_start:head_end]

    # 归一化长度
    window_len = min(traj_end, head_end - head_start)
    traj_dir_window = traj_dir_window[:window_len]
    head_yaw_window = head_yaw_window[:window_len]
    traj_speeds_window = traj_speeds_window[:window_len]

    # 归一化头部方向到 [-pi, pi]
    head_yaw_normalized = np.angle(np.exp(1j * head_yaw_window))

    # 只使用高速运动帧计算偏差
    speed_threshold = np.percentile(traj_speeds_window[traj_speeds_window > 0.1], high_speed_percentile)
    valid_mask = ~np.isnan(traj_dir_window) & (traj_speeds_window >= speed_threshold)

    traj_valid = traj_dir_window[valid_mask]
    head_valid = head_yaw_normalized[valid_mask]

    if len(traj_valid) < 10:
        print("警告：高速运动帧不足，无法计算角度偏差")
        return 0.0, 0.0

    # 计算角度差（使用圆周统计方法）
    # 注意：head_valid 和 traj_valid 都在 [-pi, pi] 范围内
    # 我们要找的是：head_valid - offset = traj_valid，即 offset = head_valid - traj_valid

    # 尝试不同的角度差计算方式，找出最匹配的
    test_configs = [
        lambda h, t: h - t,           # 直接相减
        lambda h, t: t - h,           # 反向相减
        lambda h, t: h - t + np.pi,   # +180度
        lambda h, t: h - t - np.pi,   # -180度
    ]

    config_names = ["直接相减", "反向相减", "+180度", "-180度"]

    best_concentration = -1
    best_mean_diff = 0
    best_config_name = ""

    for config_fn, name in zip(test_configs, config_names):
        diff = config_fn(head_valid, traj_valid)
        diff_wrapped = np.angle(np.exp(1j * diff))  # 归一化到 [-pi, pi]
        diff_deg = np.degrees(diff_wrapped)

        sin_mean = np.mean(np.sin(np.radians(diff_deg)))
        cos_mean = np.mean(np.cos(np.radians(diff_deg)))
        concentration = np.sqrt(sin_mean**2 + cos_mean**2)

        if concentration > best_concentration:
            best_concentration = concentration
            best_mean_diff = ((np.degrees(np.arctan2(sin_mean, cos_mean)) + 180) % 360) - 180
            best_config_name = name

    print(f"\n角度偏差分析:")
    print(f"  有效帧数: {len(traj_valid)}")
    print(f"  使用配置: {best_config_name}")
    print(f"  平均角度偏差: {best_mean_diff:.2f}°")
    print(f"  置信度: {best_concentration:.4f} (0=分散, 1=集中)")

    return best_mean_diff, best_concentration


def auto_align_head_offset_by_trajectory(trajectory_points, head_orientations,
                                         initial_head_offset=0,
                                         search_range=500,
                                         match_window=50,
                                         correlation_method='pearson'):
    """
    通过轨迹运动方向自动对齐头部方向数据

    原理：当小鼠奔跑时，其运动方向与头部方向高度相关。
    通过计算轨迹方向与头部方向序列的相关性，找到最佳偏移量。

    Args:
        trajectory_points: 轨迹点数组 (n, 2)
        head_orientations: 头部方向数组（yaw，弧度）
        initial_head_offset: 初始偏移量估计，用于缩小搜索范围
        search_range: 在初始偏移量附近的搜索范围（+/-）
        match_window: 用于计算相关性的窗口大小
        correlation_method: 相关性计算方法 ('pearson' 或 'spearman')

    Returns:
        best_offset: 最佳偏移量
        best_correlation: 最佳相关性值
        yaw_offset_deg: 角度偏差（度）
        correlation_curve: 相关性曲线（用于调试）
    """
    from scipy.stats import pearsonr, spearmanr

    # 计算轨迹运动方向，使用较低的速度阈值确保有足够的数据
    traj_directions, traj_speeds = calculate_trajectory_direction(trajectory_points, window_size=5, min_speed_threshold=0.1)

    # 只使用速度较高的帧进行匹配（奔跑时刻）
    # 使用较低的百分位数阈值（25%），确保有足够的数据点
    valid_speeds = traj_speeds[traj_speeds > 0.1]  # 只考虑有速度的帧
    if len(valid_speeds) > 0:
        speed_threshold = np.percentile(valid_speeds, 50)  # 使用25分位数
    else:
        speed_threshold = 0.1  # 默认阈值
    high_speed_mask = traj_speeds >= speed_threshold

    print(f"轨迹点数量: {len(trajectory_points)}")
    print(f"有效速度帧数: {np.sum(traj_speeds > 0)}")
    print(f"高速运动帧数: {np.sum(high_speed_mask)} (阈值: {speed_threshold:.2f})")
    print(f"速度范围: [{traj_speeds.min():.2f}, {traj_speeds.max():.2f}]")

    # 归一化头部方向到 [-pi, pi]
    head_yaw_normalized = np.angle(np.exp(1j * head_orientations))

    # 计算相关性函数
    def compute_correlation(offset):
        # 使用轨迹的前 match_window 帧作为参考
        traj_start = 0
        traj_end = min(match_window, len(traj_directions))

        # 头部方向数据从 offset 开始，取同样长度的窗口
        head_start = offset
        head_end = min(offset + match_window, len(head_yaw_normalized))

        # 确保窗口有效
        if traj_end <= traj_start or head_end <= head_start:
            return -np.inf
        if head_start < 0 or head_start >= len(head_yaw_normalized):
            return -np.inf

        # 获取匹配窗口内的数据（长度以较短的为准）
        window_len = min(traj_end - traj_start, head_end - head_start)
        if window_len < 10:
            return -np.inf

        traj_dir_window = traj_directions[traj_start:traj_start + window_len]
        head_dir_window = head_yaw_normalized[head_start:head_start + window_len]

        # 只使用有效数据（非NaN且高速运动）
        valid_mask = ~np.isnan(traj_dir_window) & high_speed_mask[traj_start:traj_start + window_len]
        traj_valid = traj_dir_window[valid_mask]
        head_valid = head_dir_window[valid_mask]

        if len(traj_valid) < 10:  # 至少需要10个有效点
            return -np.inf

        # 计算相关性
        if correlation_method == 'pearson':
            # 对于角度数据，使用 circular correlation
            # 将角度转换为复数单位向量
            traj_complex = np.exp(1j * traj_valid)
            head_complex = np.exp(1j * head_valid)

            # 计算复数相关性
            correlation = np.abs(np.mean(traj_complex * np.conj(head_complex)))
        else:
            corr, _ = spearmanr(traj_valid, head_valid)
            correlation = corr if not np.isnan(corr) else -np.inf

        return correlation

    # 在搜索范围内寻找最佳偏移量
    print(f"搜索范围: {initial_head_offset - search_range} 到 {initial_head_offset + search_range}")
    print(f"头部方向数据长度: {len(head_yaw_normalized)}")

    best_offset = initial_head_offset
    best_correlation = -np.inf
    correlation_curve = []
    valid_count = 0  # 有效计数

    for offset in range(initial_head_offset - search_range, initial_head_offset + search_range + 1):
        corr = compute_correlation(offset)
        correlation_curve.append((offset, corr))

        if corr > best_correlation and corr > -np.inf:
            best_correlation = corr
            best_offset = offset

        if corr > -np.inf:
            valid_count += 1

        # 打印进度和调试信息
        if (offset - initial_head_offset) % 100 == 0:
            print(f"偏移量 {offset}: 相关性 = {corr:.4f}")

    print(f"\n有效偏移量数量: {valid_count}/{search_range * 2 + 1}")

    print(f"\n最佳偏移量: {best_offset}")
    print(f"最佳相关性: {best_correlation:.4f}")
    print(f"偏移量变化: {best_offset - initial_head_offset}")

    # 计算角度偏差
    yaw_offset_deg, _ = calculate_yaw_offset_from_alignment(
        trajectory_points, head_orientations, best_offset,
        match_window=match_window
    )

    return best_offset, best_correlation, yaw_offset_deg, correlation_curve


def load_and_interpolate_trajectory_data(trajectory_path, skip_threshold=3):
    """
    加载轨迹点数据并进行插值处理

    Args:
        trajectory_path: 轨迹文件路径
        skip_threshold: 帧索引跳变阈值

    Returns:
        trajectory_points: 插值后的轨迹点列表
        frame_inds: 原始帧索引
    """
    # 读取轨迹数据
    df = pd.read_csv(trajectory_path, sep=',', names=['x','y','syncInd','time','frameInd'], dtype=int)
    df = df.drop_duplicates(subset='frameInd', keep='first').sort_values('frameInd')
    
    frame_inds = df['frameInd'].values
    x_vals = df['x'].values.astype(float)
    y_vals = df['y'].values.astype(float)
    
    # 检测并跳过起始部分的不连续帧
    skip = 0
    no_skip_count = 0
    
    for i in range(1, len(frame_inds)):
        if frame_inds[i] - frame_inds[i-1] > skip_threshold:
            skip = i
            no_skip_count = 0
        else:
            no_skip_count += 1
        
        if no_skip_count >= 100:
            break
    
    if skip > 0:
        print(f"检测到帧索引跳变，跳过前{skip}个轨迹点")
        frame_inds = frame_inds[skip:]
        x_vals = x_vals[skip:]
        y_vals = y_vals[skip:]
    
    # 对轨迹点进行线性插值
    total_frames = frame_inds[-1] - frame_inds[0] + 1
    f_x = interp1d(frame_inds, x_vals, kind='linear', fill_value="extrapolate")
    f_y = interp1d(frame_inds, y_vals, kind='linear', fill_value="extrapolate")
    
    # 生成插值后的轨迹点
    interpolated_frames = np.arange(frame_inds[0], frame_inds[-1] + 1)
    x_interp = f_x(interpolated_frames)
    y_interp = f_y(interpolated_frames)
    
    trajectory_points = np.vstack((x_interp, y_interp)).T
    
    print(f"已加载 {len(trajectory_points)} 个插值后的轨迹点")
    print(f"原始帧索引范围: {frame_inds[0]} 到 {frame_inds[-1]}")
    print(f"插值后帧数: {total_frames}")
    
    return trajectory_points, frame_inds


# 使用示例
if __name__ == "__main__":
    # 视频路径
    video_path = r"E:\pythonFiles\YoloV8\tempScreen\12_26_1907outputraw.mp4"
    
    # 轨迹点数据路径
    trajectory_path = r"E:\pythonFiles\YoloV8\tempScreen\12_26_1907mousePosRec.txt"
    
    # 头部方向数据路径
    head_orientation_csv = r"E:\pythonFiles\YoloV8\tempScreen\headOrientation.csv"
    
    # YOLO模型路径（可选）
    model_path = r"E:\pythonFiles\YoloV8\models\TopViewMiniscopeBodyBestWithAddition.pt"
    
    # 加载轨迹点（带插值）
    trajectory_points, frame_inds = load_and_interpolate_trajectory_data(trajectory_path)
    
    # 加载头部方向数据并转换为欧拉角
    head_orientations = load_head_orientation_csv(head_orientation_csv)
    
    if head_orientations is not None and len(trajectory_points) > 0:
        # 调用可视化函数
        offsets = visualize_motion_trail_with_alignment(
            video_path=video_path,
            trajectory_points=trajectory_points,
            head_orientations=head_orientations,
            max_frame_count=100,
            interval=5,
            offset=0,
            head_offset=1708,
            model_path=model_path,
            yolo_conf=0.7
        )
        
        if offsets:
            trajectory_offset, head_offset = offsets
            print(f"最终偏移量 - 轨迹: {trajectory_offset}, 头部方向: {head_offset}")