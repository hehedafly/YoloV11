import cv2
import math
import numpy as np

def calculate_angle(P1, P2):
    x1, y1 = P1
    x2, y2 = P2
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return -1  # 表示两个点重合，无方向
    
    # 计算标准角度，以x轴正方向为0度，逆时针为正
    std_angle_rad = math.atan2(dy, dx)
    std_angle_deg = math.degrees(std_angle_rad)
    
    # 转换到新的坐标系，以y轴反方向为0度，顺时针为正
    angle_new = (90 + std_angle_deg) % 360
    # print(angle_new)
    return angle_new

def draw_arrow(img, start_point, length, angle, color, thickness, arrow_size = -1):
    # 将角度转换为弧度
    if angle < 0:
        return
    angle_rad = math.radians(angle)
    if arrow_size <= 0:
        arrow_size = min(20, max(8, length * 0.1))
    
    # 计算终点坐标
    end_x = start_point[0] + length * math.sin(angle_rad)
    end_y = start_point[1] - length * math.cos(angle_rad)
    
    # 绘制主箭头
    cv2.line(img, (int(start_point[0]), int(start_point[1])), (int(end_x), int(end_y)), color, thickness)
    
    # 定义箭头的参数
    phi = math.radians(30)  # 箭头张开的角度
    a = arrow_size  # 箭头的长度
    
    # 计算第一个箭头边的终点
    arrow1_angle = angle_rad + phi
    arrow1_x = end_x - a * math.sin(arrow1_angle)
    arrow1_y = end_y + a * math.cos(arrow1_angle)
    
    # 计算第二个箭头边的终点
    arrow2_angle = angle_rad - phi
    arrow2_x = end_x - a * math.sin(arrow2_angle)
    arrow2_y = end_y + a * math.cos(arrow2_angle)
    
    # 绘制箭头的两个边
    cv2.line(img, (int(end_x), int(end_y)), (int(arrow1_x), int(arrow1_y)), color, thickness)
    cv2.line(img, (int(end_x), int(end_y)), (int(arrow2_x), int(arrow2_y)), color, thickness)

    return img

def calculate_circle(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    A = x1*(y2 - y3) - y1*(x2 - x3) + x2*y3 - x3*y2
    if A == 0:
        return None, None  # 共线，无法确定圆

    C = (x1**2 + y1**2)*(y3 - y2) + (x2**2 + y2**2)*(y1 - y3) + (x3**2 + y3**2)*(y2 - y1)
    D = (x1**2 + y1**2)*(x3 - x2) + (x2**2 + y2**2)*(x1 - x3) + (x3**2 + y3**2)*(x2 - x1)
    
    x = -C / (2 * A)
    y = D / (2 * A)
    r = np.sqrt((x - x1)**2 + (y - y1)**2)
    return (int(x), int(y)), int(r)

def Distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def CheckInCircle(pos, center, radius):
    return Distance(pos, center) < radius


class DefineCircle(object):
    def __init__(self):
        super(DefineCircle, self).__init__()
        self.mousePos:list[int] = []

    def define_circle_by_three_points(self, image):
        data = {
            'points': [],
            'preview_circle': None,
            'preview_arrow':[],
            'image_display': image.copy(),
            'angle' : -1
        }
        cv2.namedWindow('Define Circle by Three Points')
        cv2.setMouseCallback('Define Circle by Three Points', self.on_mouse_three_points, param=(data, self))

        while True:
            # print(self.mousePos)
            img = data['image_display'].copy()
            for point in data['points']:
                cv2.circle(img, point, 5, (0,0,255), 2)
            if data['preview_circle']:
                center, radius = data['preview_circle']
                data['angle'] = calculate_angle(center, data['points'][0])
                # 确保圆心在图像范围内
                # if 0 <= center[0] < image.shape[1] and 0 <= center[1] < image.shape[0]:
                cv2.circle(img, center, radius, (0,255,0), 2)
                if len(data['points']) == 3:
                    draw_arrow(img, center, radius, calculate_angle(center, data['points'][0]), (255, 0, 0), 2, 10)
                    draw_arrow(img, center, radius, calculate_angle(center, self.mousePos), (0, 255, 0), 2, 10)
                    if len(self.mousePos):
                        cv2.putText(img, str(calculate_angle(center, self.mousePos)), [self.mousePos[0] - 40, self.mousePos[1]], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                elif len(data['points']) == 4:
                    data['angle'] = calculate_angle(center, data['points'][-1])
                    draw_arrow(img, center, radius, data['angle'], (0, 255, 0), 2, 4)
            if len(self.mousePos) and len(data['points']) < 4:
                cv2.circle(img, self.mousePos, 5, (0,0,255), 2)
                annotate:list[str] = ["1st/3 point defines the circle and direction if needed", "2nd/3 point defines the circle", "3rd/3 point defines the circle", "circle direction"]
                cv2.putText(img, annotate[len(data['points'])], [self.mousePos[0] - 40, self.mousePos[1] + 40], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow('Define Circle by Three Points', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return None, None, None
            elif key == 13:  # Enter
                if len(data['points']) >= 3:
                    center, radius = calculate_circle(data['points'][0], data['points'][1], data['points'][2])
                    if center and radius:
                        cv2.destroyAllWindows()
                        return center, radius, data['angle']
                    else:
                        print("三点共线，无法确定圆，请重新选择点。")
                        data['points'] = []
                else:
                    print("请选择至少三个点。")
            elif key == 8:  # Backspace
                if data['points']:
                    data['points'].pop()
                    data['preview_circle'] = None

    @staticmethod
    def on_mouse_three_points(event, x, y, flags, param) -> tuple[tuple, tuple, bool]:
        self = param[1]
        mousePos:list[int] = self.mousePos
        if len(mousePos):
            mousePos[0], mousePos[1] = x, y
        else:
            mousePos.append(x)
            mousePos.append(y)

        data = param[0]
        points = data['points']
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x,y))
            if len(points) == 3:
                center, radius = calculate_circle(points[0], points[1], points[2])
                if center and radius:
                    data['preview_circle'] = (center, radius)
                else:
                    points.pop()
                    print("三点共线，无法确定圆，请重新选择点。")
            elif len(points) == 2:
                # 计算预览圆
                p1, p2 = points
                center, radius = calculate_circle(p1, p2, (x,y))
                if center and radius:
                    data['preview_circle'] = (center, radius)
                else:
                    data['preview_circle'] = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if len(points) == 2:
                # 根据p1, p2和当前鼠标位置计算预览圆
                p1, p2 = points
                center, radius = calculate_circle(p1, p2, (x,y))
                if center and radius:
                    data['preview_circle'] = (center, radius)
                else:
                    data['preview_circle'] = None

    def define_circle_by_center_and_point(self, image):
        data = {
            'points': [],
            'preview_circle': None,
            'preview_arrow': [],
            'image_display': image.copy()
        }
        cv2.namedWindow('Define Circle by Center and Point')
        cv2.setMouseCallback('Define Circle by Center and Point', self.on_mouse_center_and_point, param=[data, self])
        while True:
            img = data['image_display'].copy()
            if data['points']:
                center = data['points'][0]
                cv2.circle(img, center, 2, (0,0,255), -1)
                if data['preview_circle']:
                    cv2.circle(img, data['preview_circle'][0], data['preview_circle'][1], (0,255,0), 2)
                    cv2.putText(img, str(data['preview_circle'][1]), data['preview_circle'][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

            if len(data['points']) >= 2:
                center = data['points'][0]
                radius = int(np.linalg.norm(np.array(center) - np.array(data['points'][1])))
                cv2.circle(img, center, radius, (0,255,0), 2)
                cv2.putText(img, str(radius), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                if len(data['preview_arrow']):
                    inner:bool = CheckInCircle(center, data['preview_arrow'], radius)
                    angle = calculate_angle(center, data['preview_arrow'])
                    if inner: 
                        draw_arrow(img, center, Distance(center, data['preview_arrow']), angle, (200, 200, 0), 2)
                        cv2.putText(img, "inner", data['preview_arrow'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

                    else:
                        angleRad = math.radians(angle)
                        tempRadius = Distance(center, data['preview_arrow'])
                        cv2.putText(img, "outter", data['preview_arrow'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                        draw_arrow(img, np.add(center , (tempRadius * np.sin(angleRad), -1 *tempRadius * np.cos(angleRad))), tempRadius - radius, angle + 180, (200, 200, 0), 2)
            cv2.imshow('Define Circle by Center and Point', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow('Define Circle by Center and Point')
                return None, None, None
            elif key == 13:  # Enter
                if len(data['points']) >= 2:
                    center = data['points'][0]
                    radius = int(np.linalg.norm(np.array(center) - np.array(data['points'][1])))
                    inner = True
                    if len(data['preview_arrow']):
                        inner:bool = CheckInCircle(center, data['preview_arrow'], radius)

                    cv2.destroyWindow('Define Circle by Center and Point')
                    data['preview_arrow'] = None

                    return center, radius, inner
                else:
                    print("请选择圆心和一个点。")
            elif key == 8:  # Backspace
                if data['points']:
                    data['points'].pop()
                    data['preview_circle'] = None

    @staticmethod
    def on_mouse_center_and_point(event, x, y, flags, param):
        data = param[0]
        points = data['points']
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) == 0:
                points.append((x,y))
            elif len(points) == 1:
                points.append((x,y))
                # 计算半径
                center = points[0]
                radius = int(np.linalg.norm(np.array(center) - np.array((x,y))))
                data['preview_circle'] = (center, radius)
            else:
                # 重新开始
                points = [ (x,y) ]
                data['preview_circle'] = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if len(points) == 1:
                # 计算预览圆
                center = points[0]
                radius = int(np.linalg.norm(np.array(center) - np.array((x,y))))
                data['preview_circle'] = (center, radius)
            elif len(points) == 2:
                #内外箭头
                data['preview_arrow'] = (x, y)


if __name__ == '__main__':
    pass
    # image = cv2.imread('bottomVision.jpg')

    # circleSelect = DefineCircle()
    # center, radius, angle = circleSelect.define_circle_by_three_points(image)
    # if center and radius:
    #     print(f'圆心: {center}, 半径: {radius}, 角度: {angle}')
    # else:
    #     print("未定义圆。")

    # center, radius, inner = circleSelect.define_circle_by_center_and_point(image)
    # if center and radius:
    #     print(f'圆心: {center}, 半径: {radius}')
    # else:
    #     print("未定义圆。")