import cv2
import numpy as np

def parse_annotation_line(line):
    """
    解析 txt 文件中的一行数据。
    返回边界框信息和关键点信息。
    """
    parts = line.strip().split()
    # 忽略第一个值 "0"
    bbox_info = list(map(float, parts[1:5]))  # 中心点坐标和宽高
    keypoints = list(map(float, parts[5:]))   # 关键点坐标及可见性
    return bbox_info, keypoints

def draw_bbox_and_keypoints(image_path, annotation_path):
    """
    根据注释文件绘制边界框和关键点。
    """
    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"图片文件 {image_path} 未找到！")
    
    height, width = image.shape[:2]
    
    # 打开注释文件
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        # 解析每一行数据
        bbox_info, keypoints = parse_annotation_line(line)
        
        # 提取边界框信息
        cx, cy, bw, bh = bbox_info
        x_min = int((cx - bw / 2) * width)
        y_min = int((cy - bh / 2) * height)
        x_max = int((cx + bw / 2) * width)
        y_max = int((cy + bh / 2) * height)
        
        # 绘制边界框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # 绘制关键点
        for i in range(0, len(keypoints), 3):
            px, py, visible = keypoints[i:i+3]
            px = int(px * width)
            py = int(py * height)
            
            if visible == 2:  # 可见，绘制实心圆
                cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
            elif visible == 1:  # 不可见，绘制空心圆
                cv2.circle(image, (px, py), 5, (255, 0, 0), 2)
    
    # 保存或显示结果
    cv2.imshow("result", image)
    cv2.waitKey()
    # cv2.imwrite(output_path, image)
    # print(f"结果已保存到 {output_path}")

# 示例调用
containFolder = 'PoseExtract0407095629'
image_path = containFolder + '/'+ "03_20_1550outputraw0.jpg"
annotation_path = containFolder + '/'+ "03_20_1550outputraw0.txt"
# output_path = "path/to/your/output.jpg"

draw_bbox_and_keypoints(image_path, annotation_path)