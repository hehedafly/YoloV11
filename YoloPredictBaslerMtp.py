import datetime
import math
import os
import socket
import threading
import time
import copy
import queue
# Third-party imports
import torch
import cv2
import argparse
import keyboard
import numpy as np
from typing import Literal
from collections import defaultdict, deque
from ultralytics import YOLO
from pypylon import pylon
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from IPC import IPCTest
from CircleSelect import CircleSelect
from MessageBox import PyWinMessageBox

from pyinstrument import Profiler

# region ------------------------------------------------argparse-------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--useargs', action="store_true", help='按argparse参数运行')
parser.add_argument('-l', '--load', action="store_true", help='加载sceneInfo')
parser.add_argument('-c', '--camera', type=str, default='basler', help='图像来源basler, common, video')
parser.add_argument('--ccare', action="store_true", help='与其他程序通讯')
parser.add_argument('-t', '--type', type=str, default='unity', help='通讯对象, unity或processing')
parser.add_argument('-s', '--disablebaslersyncsignal', action="store_true", help='true: always ExposureActive, false: ExposureActive after connection')
parser.add_argument('-f', '--fps', type=int, default=50, help='')
parser.add_argument('-r', '--rec', action="store_true", help='record video file')
parser.add_argument('-m', '--multiThread', action="store_true", help='enable multi thread for camera grab and predict')
parser.add_argument('-i', '--ignoreKeyboard', action="store_true", help='ignore keyboard event')
parser.add_argument('-d', '--detectMethod', type=str, help='yolo or blob')
args = parser.parse_args()
useargs = args.useargs
#endregion ------------------------------------------------argparse end------------------------------

# region ------------------------------------------------meta Info-------------------------------------------
CameraTypes = ["basler", "common", "video"]
CameraType = "video"                                             if not useargs else args.camera
videoPath = "01_17_1842outputraw.mp4"
modelNmae = "models/TopViewDifferentiateLickSpout.pt"
# modelNmae = "models/TopViewMiniscopeBodyBestWithAddition.engine"
confidenceCoefficient = 0.7
CCare:bool = True                                                    if not useargs else args.ccare
CType:Literal['unity', 'processing'] = "unity"                   if not useargs else args.type
CPort = 2333 # for processing
ConnectRetryInterval = 2
BaslerSyncSignalControl = True                                   if not useargs else not args.disablebaslersyncsignal
#true: ExposureActive after connection, false: always ExposureActive
resolution = [1440,1080]
FPS = 50                                                         if not useargs else args.fps
#50fps max, grap need ~15ms, others need ~18.5ms in total
baslerWaitTime = int(2000/FPS) + 1
recordResult = False                                             if not useargs else args.rec
recordPredictResult = False
recordOnlyMouse = True
recordOnlyMouseResolution = [100, 100]
finalResolution = resolution
ignoreKeyboardEvent = False                                      if not useargs else args.ignoreKeyboard
# finalResolution = (resolution[0], resolution[1]) if not recordOnlyMouse else (recordOnlyMouseResolution[0], recordOnlyMouseResolution[1])

recordMissframe = False
videoSaveFolder = r"E:\pythonFiles\YoloV8\outputVideo"
missedFrameSaveFolder = r"E:\pythonFiles\YoloV8\missedFrames"
multiThread = True                                              if not useargs else args.multiThread 
Task: Literal['detect', 'track'] = 'detect'
detectMethod: Literal['yolo', 'blob'] = 'yolo'                  if not useargs else args.detectMethod
blobDetector = None
frame_rate_divider = 1  # 设置帧率除数
missed_frame_rate_divider = 10
frame_count = 0  # 初始化帧计数器
grabbedFrameCount = 0
missed_frame_count = 0
hide = False
simulate = False
selectAreas:list[list[int]] = []#[mark, type(0:circle, 1:rect), centerx(fpx), centery(fpy), radius(spx), angle(spy)]
simulateMousePos = [-1, -1, -1, -1, -1]
selectChanged = False
FontSize = 0.8
FontThick = 2
costTime = 0
useCuda = torch.cuda.is_available()
device = "cuda:0" if useCuda else "cpu"
if not useCuda:
    print("Cuda not available, using cpu")
performanceAnalysis = False
task_queue = queue.Queue()
result_queue = queue.Queue()
message_receive_queue = queue.Queue()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if not os.path.exists(videoSaveFolder):
    os.mkdir(videoSaveFolder)

processingcConnectRetryTime = 0


# endregion ------------------------------------------------meta Info end-------------------------------------------

# region ------------------------------------------------camera and log Info-------------------------------------------
# 连接Basler相机列表的第一个相机
def setBaslerCamera(_camera:pylon.InstantCamera):
    if _camera is None:
        return
    
    _camera.Open()
    _camera.Width.Value = resolution[0]
    _camera.Height.Value = resolution[1]
    # camera.PixelFormat = "BGR8"
    _camera.Gain.Value = 7.5
    _camera.ExposureTime.Value = 10000
    _camera.LineSelector.Value = "Line3"
    _camera.LineMode.SetValue("Output")
    # _camera.LineSource.Value = "UserOutput2"
    # _camera.UserOutputSelector.Value = "UserOutput2"
    # _camera.UserOutputValue.Value = True
    _camera.TriggerSelector.Value = "FrameStart"
    _camera.TriggerMode.Value = "On"
    _camera.TriggerSource.Value = "Software"
    # _camera.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(), pylon.RegistrationMode_ReplaceAll,
    #                          pylon.Cleanup_Delete)
    # _camera.StartGrabbing()
    if not BaslerSyncSignalControl:
        _camera.LineSource.Value = "ExposureActive"
        print("basler set to ExposureActive")
    else:
        _camera.LineSource.Value = "UserOutput2"
        _camera.UserOutputSelector.Value = "UserOutput2"
        _camera.UserOutputValue.Value = True
    _camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

def BaslerSyncEnable(_enable:bool = True):
    _enable = (not BaslerSyncSignalControl) | _enable
    if CameraType == "basler":
        if grabber is not None:
            _camera = grabber.baslerCamera
        else:
            _camera = camera

        if _enable:
            _camera.LineSource.Value = "ExposureActive"
            print("basler set to ExposureActive")
        else:
            _camera.LineSource.Value = "UserOutput2"
            _camera.UserOutputSelector.Value = "UserOutput2"
            _camera.UserOutputValue.Value = True
    if _enable and grabber is not None:
        grabber.clearPreviousStatus()


if multiThread:
    camera = None
else:
    if CameraType == "basler":
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        setBaslerCamera(camera)

        # mediaNamePure = "Basler" + datetime.datetime.now().strftime("%m_%d_%H%M")
    elif CameraType == "video":
        video_path = videoPath
        camera = cv2.VideoCapture(video_path)

        # mediaNamePure = ".".join(video_path.split(".")[0:-1])
    elif CameraType == "common":
        camera = cv2.VideoCapture(0)
        # mediaNamePure = "camera" + datetime.datetime.now().strftime("%m_%d_%H%M")
    else:
        print(f"wrong camera type:{CameraType}")
        exit()

timestr = datetime.datetime.now().strftime("%m_%d_%H%M")
if not os.path.exists(missedFrameSaveFolder):
    os.mkdir(missedFrameSaveFolder)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
if recordResult:
    pre_fname = os.path.join(videoSaveFolder, timestr)
    posRecFile = open(pre_fname + "mousePosRec.txt", 'w')
    if recordPredictResult:
        out = cv2.VideoWriter(os.path.join(videoSaveFolder, timestr+'output.mp4'), fourcc, 50.0, finalResolution)
    else:
        out = None
    if not multiThread:
        outRaw = cv2.VideoWriter(pre_fname +'outputraw.mp4', fourcc, 50.0, finalResolution)
    else:
        outRaw = None
else:
    out = None
    outRaw = None
    posRecFile = None
defineCircle = CircleSelect.DefineCircle()

def CareMessageReceive():
    event = threading.Event()
    while True:
        event.wait(0.02)
        for msg in CInstance.ReadToStr(CInstance.careindex):
            message_receive_queue.put(msg)

class FrameGrabber(threading.Thread):
    def __init__(self, _cameraType, fps=50):
        super().__init__()

        if CameraType == "basler":
            try:
                self.tempCamera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                setBaslerCamera(self.tempCamera)

            except Exception as e:
                print(f"failed to connect to basler camera:{e}")
                exit()

        elif CameraType == "video":
            video_path = videoPath
            camera = cv2.VideoCapture(video_path)

        elif CameraType == "common":
            camera = cv2.VideoCapture(0)

        else:
            print(f"wrong camera type:{CameraType}")
            exit()

        self.cameraType = _cameraType
        self.baslerCamera = self.tempCamera if _cameraType == "basler" else None
        self.camera = camera if _cameraType == "common" or _cameraType == "video" else None

        self.timestr = datetime.datetime.now().strftime("%m_%d_%H%M")
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = None
        self.writerRelease = False

        self.record = False
        self.fps = fps
        self.exposuredFrames = 0
        self.frame_buffer = deque(maxlen=int(fps * 0.4))  # 帧缓存队列
        self.lock = threading.Lock()
        self.running = False
        
        # 动态调整参数
        self.interval = 1.0 / fps
        self.adjustment_factor = 0.1         # 延迟补偿系数
        self.last_delay = 0.0

    def releaseCamera(self):
        if self.baslerCamera is not None:
            self.baslerCamera.StopGrabbing()
            self.baslerCamera.Close()
        elif self.camera is not None:
            self.camera.release()

    def getFrame(self) -> tuple[bool, np.ndarray]:
        try:
            if self.cameraType == "basler":
                # _t = time.time()
                if self.baslerCamera.WaitForFrameTriggerReady(baslerWaitTime, pylon.TimeoutHandling_ThrowException):
                    self.baslerCamera.ExecuteSoftwareTrigger()
                    # print("this step?")
                # grabResult = self.baslerCamera.GrabOne(100, pylon.TimeoutHandling_ThrowException)#
                grabResult = self.baslerCamera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                # print(f"grab time:{time.time()-_t}")
                ret =  grabResult is not None
                # ret =  grabResult.GrabSucceeded()
                if not ret:
                    for i in range(3):
                        print(f"failed to get frame {i+1} times")
                        grabResult = self.baslerCamera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                        frame = cv2.cvtColor(np.array(grabResult.Array, np.uint8), cv2.COLOR_GRAY2RGB)
                        ret =  grabResult.GrabSucceeded()
                        if ret:
                            break
                    print("lost connection to basler")
                frame = cv2.cvtColor(np.array(grabResult.Array, np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                ret, frame = self.camera.read()
                if not ret:
                    print("no camera connected")
            return ret, frame
        except Exception as e:
            print(e)
            return False, None
    
    def clearPreviousStatus(self):
        self.exposuredFrames = 0
        self.frame_buffer.clear()

    def run(self):
        self.running = True
        expect_start_time = time.time()
        
        while self.running:
            # 动态调整间隔
            # adjusted_interval = max(0.0001, self.interval - self.last_delay * self.adjustment_factor)
            
            # start_time = time.time()
            ret, frame = self.getFrame()
            if not ret:
                self.running = False
                self.clear_buffer()
                print("failed to get frame in child thread")
                break
            
            # 写入视频
            if self.writer != None and self.record:
                # with self.lock:
                if self.record:
                    self.writer.write(frame)
            self.exposuredFrames += 1 
            
            if self.writerRelease:
                self.VideoClear()
                self.writerRelease = False

            # 更新缓存
            # with self.lock:
            self.frame_buffer.append(frame)
            
            # 计算延迟补偿
            # process_time = time.time() - start_time
            expect_start_time += self.interval
            
            # 控制帧率
            sleep_time = expect_start_time - time.time()
            # print(f"sleep_time:{sleep_time}")
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.releaseCamera()
        if self.writer != None:
            self.writer.release()

    # def getFrameCount(self):
    #     return len(self.frame_buffer)

    def get_last_frame(self) -> tuple[bool, int, np.ndarray]:
        with self.lock:
            if len(self.frame_buffer):
                # 返回最新帧
                return True, self.exposuredFrames, self.frame_buffer[-1]
            return False, -1, None
        
    def returnCameraStatus(self):
        if self.baslerCamera is not None:
            return self.baslerCamera.IsGrabbing()
        elif self.camera is not None:
            return self.camera.isOpened()

    def clear_buffer(self):
        with self.lock:
            self.frame_buffer.clear()

    def startRecord(self):
        if recordResult:
            timestr = datetime.datetime.now().strftime("%m_%d_%H%M")
            self.writer = cv2.VideoWriter(os.path.join(videoSaveFolder, timestr+'outputraw.mp4'), fourcc, 50.0, finalResolution)
            self.record = True

    def stop(self):
        print("thread stop")
        self.running = False

    def VideoClearPublic(self):
        self.writerRelease = True

    def VideoClear(self):
        with self.lock:
            if type(self.writer) == cv2.VideoWriter:
                self.writer.release()
                self.writer = None
                self.record = False

    def Isrecording(self):
        return self.record

def startRecord():
    global outRaw, out
    timestr = datetime.datetime.now().strftime("%m_%d_%H%M")
    if recordResult:
        outRaw = cv2.VideoWriter(videoSaveFolder + timestr+'outputraw.mp4', fourcc, 50.0, (resolution[0], resolution[1]))
        if recordPredictResult:
            out = cv2.VideoWriter(videoSaveFolder + timestr+'output.mp4', fourcc, 50.0, (resolution[0], resolution[1]))

def WriteFrame(frame, isRaw:bool):
    if isRaw and outRaw is not None:
        outRaw.write(frame)
    else:
        if recordPredictResult and out is not None:
            out.write(frame)

def VideoClear():
    global outRaw, out
    if outRaw is not None:
        outRaw.release()
        outRaw = None
    if out is not None:
        out.release()
        out = None

#endregion------------------------------------------------camera and log Info end-------------------------------------------

# region ----------------------------------------model load and predict function-----------------------------------
# model = YOLO("TopViewbest.pt")
class Model():
    def __init__(self, modelName:str):
        self.modelName = modelName
        self.modelType = ""
        self.model = None
        # self.openVINOModel:CompiledModel = None

        if modelName.endswith(".pt") or modelName.endswith(".engine"):
            self.model = YOLO(self.modelName)
            self.modelType = "yolo"
        elif modelName.endswith(".onnx"):
            self.model = YOLO(modelName)
            self.modelType = "onnx"
        else:
            print("目前仅支持yolo(.pt, .engine)模型，onnx模型")
            # print("目前仅支持yolo(.pt)模型，openVINO模型")
            exit()

    def Predict(self, img:np.ndarray, task: Literal['detect', 'track'] = 'detect'):
        if self.modelType == "yolo":
            tempmodel:YOLO = self.model
            if task == 'detect':
                results = tempmodel(img, verbose=False, conf = confidenceCoefficient, device = device, task = "detect")
            elif task == 'track':
                results = tempmodel.track(img, verbose=False, conf = confidenceCoefficient, device = device)
            for result in results:
                if len(result.boxes):
                    box = result.boxes[0]
                    xyxy = np.array(box.xyxy[0].tolist(), int)
                    return [int((xyxy[0]+xyxy[2])*0.5), int((xyxy[1]+xyxy[3])*0.5)]
            else:
                return None
        else:
            resizedimg = cv2.resize(img, (640, 480))
            if self.modelType == "onnx":
                if task == 'detect':
                    results = self.model(resizedimg, imgsz=(480,640), verbose=False, conf = confidenceCoefficient, device = device)
                elif task == 'track':
                    results = self.model.track(resizedimg, imgsz=(480,640), verbose=False, conf = confidenceCoefficient, device = device)
                for result in results:
                    if len(result.boxes):
                        box = result.boxes[0]
                        xyxy = np.array(box.xyxy[0].tolist(), int)
                        return [int((xyxy[0]+xyxy[2])*0.5 * (1440/640)), int((xyxy[1]+xyxy[3])*0.5 * (1080/480))]
                else:
                    return None 
            return None
        
class StableBlobDetector:
    def __init__(self, min_area=500, max_area=2000, history=100):
        # 改用KNN背景减除器，对静止目标更友好
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
            history=history,
            dist2Threshold=400,  # 距离阈值，越小越敏感
            detectShadows=False
        )
        self.min_area = min_area
        self.max_area = max_area
        
        # 时序滤波缓冲（平滑抖动）
        self.alpha = 0.7  # EMA系数：0.5-0.8（越大越跟手，越小越平滑）
        self.ema_pos = None
        self.kalman = self._init_kalman()
        self.last_valid_pos = None
        
        # 控制背景更新：检测到目标后可暂停更新以捕捉静止物体
        self.freeze_bg_counter = 0
        
    def init_background(self, frame, frames=10):
        """用多帧初始化背景模型"""
        for _ in range(frames):
            self.bg_subtractor.apply(frame, learningRate=0.01)
    
    def calculate_centroid(self, contour):
        """使用矩计算精确质心，而非bbox中心"""
        M = cv2.moments(contour)
        if M["m00"] > 100:  # 避免除零和噪声
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None
    
    def _init_kalman(self):
        kalman = cv2.KalmanFilter(4, 2)  # 4状态量(位置+速度), 2观测量(位置)
        kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kalman.transitionMatrix = np.array([
            [1,0,1,0],  # x = x + vx
            [0,1,0,1],  # y = y + vy  
            [0,0,1,0],  # vx = vx
            [0,0,0,1]   # vy = vy
        ], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        return kalman

    def smooth_position(self, new_pos):
        if new_pos is None or new_pos == [-1, -1]:
            return self.last_valid_pos if self.last_valid_pos else [-1, -1]
        
        if self.ema_pos is None:
            self.ema_pos = np.array(new_pos, dtype=np.float32)
        else:
            self.ema_pos = self.alpha * np.array(new_pos) + (1-self.alpha) * self.ema_pos
        
        self.kalman.correct(np.array(new_pos, dtype=np.float32))
        prediction = self.kalman.predict()
        kalman_pos = [int(prediction[0]), int(prediction[1])]
        
        # 混合使用：EMA用于平滑，Kalman用于预测减少滞后
        final_x = int(self.alpha * self.ema_pos[0] + (1-self.alpha) * kalman_pos[0])
        final_y = int(self.alpha * self.ema_pos[1] + (1-self.alpha) * kalman_pos[1])
        
        return [final_x, final_y]
    
    def detect(self, frame, enable_bg_learning=True):
        """
        主检测函数
        enable_bg_learning: 设为False可"冻结"背景，防止静止目标消失
        """
        # 动态调整学习率：检测到目标后短暂暂停背景更新（解决静止目标问题）
        learning_rate = -1 if enable_bg_learning else 0
        
        # 1. 背景减除
        fg_mask = self.bg_subtractor.apply(frame, learningRate=learning_rate)
        
        # 2. 形态学清理（更激进的噪声去除）
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        
        # 开运算去除小噪声，闭运算填充目标内部空洞
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        # 可选：高斯模糊进一步平滑边缘（减少质心抖动）
        fg_mask = cv2.GaussianBlur(fg_mask, (5,5), 0)
        _, fg_mask = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)
        
        # 3. 查找轮廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_blob = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area*0.5 < area < self.max_area*1.5):
                continue
            
            # 计算形状质量（排除长条噪声）
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # 计算质心
            centroid = self.calculate_centroid(cnt)
            if not centroid:
                continue
            
            # 评分：面积适中 + 形状接近圆形（0.7-1.0最佳）
            score = area * (circularity + 0.1)
            
            if score > best_score:
                best_score = score
                best_blob = {
                    'centroid': centroid,
                    'area': area,
                    'circularity': circularity,
                    'contour': cnt
                }
        
        # 4. 结果处理
        if best_blob:
            raw_pos = list(best_blob['centroid'])
            smoothed_pos = self.smooth_position(raw_pos)
            self.last_valid_pos = smoothed_pos
            
            # 检测到有效目标时，短暂冻结背景更新（3-5帧）以捕捉静止目标
            self.freeze_bg_counter = 5
            return smoothed_pos, fg_mask, best_blob
        else:
            # 未检测到目标时，允许背景更新
            if self.freeze_bg_counter > 0:
                self.freeze_bg_counter -= 1
            return [-1, -1], fg_mask, None
        
def BlobPredictInit(frame:np.ndarray):
    global blobDetector
    blobDetector = StableBlobDetector(min_area=500, max_area=2000)
    blobDetector.init_background(frame)

def BlobPredict(frame:np.ndarray):
    global blobDetector
    if blobDetector is None:
        BlobPredictInit(frame)
    # 有freeze_bg_counter机制，静止目标也能保持几帧
    pos, fg_mask, blob_info = blobDetector.detect(frame)
    return pos


try:
    if detectMethod == 'yolo':
        model = Model(modelNmae)
    else:
        model = None
except Exception as e:
    print(f"failed to load model:{e}")

if multiThread:
    if detectMethod == 'yolo':
        def Predict():
            while True:
                frame, _ind = task_queue.get()
                result = model.Predict(frame, task=Task)
                result_queue.put((result, _ind))
    else:
        def Predict():
            while True:
                frame, _ind = task_queue.get()
                result = BlobPredict(frame)
                result_queue.put((result, _ind))
else:
    def Predict(frame:np.ndarray):
        if detectMethod == 'yolo':
            return model.Predict(frame, task=Task)
        elif detectMethod == 'blob':
            return BlobPredict(frame)
        else:
            return []

# endregion----------------------------------------model load and predict function end-----------------------------------

#selectPlace: list[int] = [-1, -1, -1, -1, -1, -1]#type: mark; type(check pos region), 0-rectange, 1-circle ; x/centerx ; y/centery ; w/rad ; h/inner

# region ----------------------------------------scene and selectAreas info-----------------------------------

startTime = -1
unityFixedUscaledTimeOffset:float = 0
createdTime = time.process_time()
lastframeInd:int = -1
frameIndForSingleThread: int = -1
sync = False
syncInd:int = -1
markCountPerType = 32
sceneInfo:list[float] = []#[sceneCenter[0], sceneCenter[1], sceneRadius, sceneAngle]
selectionSaveTxtName = "scene and selectAreas.txt"

f_selectionSaveTxt = open(selectionSaveTxtName, 'r+' if os.path.exists(selectionSaveTxtName) else 'w+', encoding='utf-8')
content = f_selectionSaveTxt.readlines()
sceneContent = [s for s in content if s.startswith("scene:")]
areaContent = [s for s in content if s.startswith("selectAreas:")]
if len(sceneContent) and (args.load or PyWinMessageBox.YesOrNo("load Previous SceneInfo?", "save & load") == 'YES'):
    for line in sceneContent:
        line = line.replace("\n", "")
        sceneInfo = [float(i) for i in (line.split(':')[1]).split(';')]
        print("scene info loaded: " + line)
if len(areaContent) and (args.load or PyWinMessageBox.YesOrNo("load Previous select areas?", "save & load") == 'YES'):
    for line in areaContent:
        line = line.replace("\n", "")
        selectAreas.append([int(i) for i in (line.split(':')[1]).split(';')])
        print("selectAreas info loaded: " + line)

realMouseCenter = [-1, -1]
def ProcessMouseNearRegion(pos, _frame):
    _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
    tempMask = np.zeros_like(_frame)
    tempMask = cv2.circle(tempMask, pos, 80, (255,255,255), 160)
    masked_frame = cv2.bitwise_and(_frame, _frame, mask=tempMask)
    np.tile(masked_frame[:, :, np.newaxis], (1, 1, 3))

    return  np.tile(masked_frame[:, :, np.newaxis], (1, 1, 3))

def PointOffset(point, offset:int):
    return (point[0] + offset, point[1] - offset)

def drawSelectArea(frame, selectAreas:list[list[int]], color = None):
    global markCountPerType
    fontSize = 1.5
    fontThick = 2
    
    for selectPlace in selectAreas:
        mark = selectPlace[0] // markCountPerType
        if color == None:
            drawcolor = (255, 0, 0) if mark == 0 else (0, 0, 255)
        else:
            drawcolor = color

        try:
            if selectPlace[1] == 0:
                frame = cv2.circle(frame, (selectPlace[2], selectPlace[3]), selectPlace[4], drawcolor, 2)
                frame = cv2.putText(frame, str(selectPlace[0] % markCountPerType), PointOffset((selectPlace[2], selectPlace[3]), -10), cv2.FONT_HERSHEY_SIMPLEX, fontSize, drawcolor, fontThick)

            elif selectPlace[1] == 1:
                frame = cv2.rectangle(frame, (selectPlace[2], selectPlace[3]), (selectPlace[4], selectPlace[5]), drawcolor, 2)
                frame = cv2.putText(frame, str(selectPlace[0] % markCountPerType), PointOffset(((selectPlace[2] + selectPlace[4]) // 2, (selectPlace[3] + selectPlace[5]) // 2), -10), cv2.FONT_HERSHEY_SIMPLEX, fontSize, drawcolor, fontThick)
        except:
            pass
# endregion ----------------------------------------scene and selectAreas info-----------------------------------

#region Matplotlib

class GUI:
    def __init__(self, frame, selectList:list[list[int]], sceneInfo:list[float]):
        # cv2.imshow("__init__ origin frame: ", frame)
        self.oframe = copy.deepcopy(frame)
        self.frame = copy.deepcopy(frame)
        self.sceneInfo = sceneInfo
        (h ,w, _) = frame.shape
        self.selectListRef = selectList
        self.oselectList = copy.deepcopy(selectList)
        self.selectList = copy.deepcopy(selectList)
        self.type0List = [item for item in selectList if len(item) >= 0 and 0 <= item[0] < markCountPerType]
        self.type1List = [item for item in selectList if len(item) >= 0 and markCountPerType <= item[0] < markCountPerType*2]
        # 创建GUI
        self.fig, self.ax = plt.subplots(figsize=(w * 0.008, h * 0.008 + 1))
        self.buttonWidth = 0.1
        self.buttonHeight = 0.04
        # tempAxPos = self.ax.get_position().get_points()
        # tempAxPos = np.append(tempAxPos[0], tempAxPos[1])
        self.ax.set_position([0.04, self.buttonWidth, 0.94, 0.98])

        self.DrawAllContent()

        self.ax.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

        self.createButton()

        plt.show()

    def DrawAllContent(self):
        sceneCenter = (int(sceneInfo[0]), int(sceneInfo[1]))
        sceneRadius = int(sceneInfo[2])
        sceneAngle = sceneInfo[3]
        sceneType = "rect" if len(sceneInfo) == 5 and sceneInfo[4] == 1 else "circle"
        if sceneType == "circle":
            self.frame = cv2.circle(self.frame, sceneCenter, sceneRadius, (0, 255, 0), 2)
            self.frame = CircleSelect.draw_arrow(self.frame, sceneCenter, sceneRadius, sceneAngle, (0, 255, 0), 2)
        elif sceneType == "rect":
            self.frame = cv2.rectangle(self.frame, (sceneCenter[0], sceneCenter[1]), (sceneRadius, sceneAngle), (0, 255, 0), 2)
        
        drawSelectArea(self.frame, self.selectList)

    def createButton(self):
        
        self.button0circle_ax = plt.axes([0.1, self.buttonHeight * 2.1, self.buttonWidth, self.buttonHeight])
        self.button0circle = widgets.Button(self.button0circle_ax, 'Trigger region\n circle ')
        self.button0circle.on_clicked(self.add_to_list)

        self.button0rect_ax = plt.axes([0.1, self.buttonHeight, self.buttonWidth, self.buttonHeight])
        self.button0rect = widgets.Button(self.button0rect_ax, 'Trigger region\n rect ')
        self.button0rect.on_clicked(self.add_to_list)

        self.button1circle_ax = plt.axes([0.25, self.buttonHeight * 2.1, self.buttonWidth, self.buttonHeight])
        self.button1circle = widgets.Button(self.button1circle_ax, 'Destination\n circle ')
        self.button1circle.on_clicked(self.add_to_list)

        self.button1rect_ax = plt.axes([0.25, self.buttonHeight, self.buttonWidth, self.buttonHeight])
        self.button1rect = widgets.Button(self.button1rect_ax, 'Destination\n rect ')
        self.button1rect.on_clicked(self.add_to_list)

        self.circleArray_ax = plt.axes([0.4, self.buttonHeight * 2.1, self.buttonWidth, self.buttonHeight])
        self.circleArray_button = widgets.Button(self.circleArray_ax, 'Circle Array\n Last Selection')
        self.circleArray_button.on_clicked(self.CircleArray)

        self.pop_ax = plt.axes([0.55, self.buttonHeight * 2.1, self.buttonWidth, self.buttonHeight])
        self.pop_button = widgets.Button(self.pop_ax, 'Pop')
        self.pop_button.on_clicked(self.ClearList)

        self.clear_ax = plt.axes([0.55, self.buttonHeight, self.buttonWidth, self.buttonHeight])
        self.clear_button = widgets.Button(self.clear_ax, 'Clear')
        self.clear_button.on_clicked(self.ClearList)

        self.close_ax = plt.axes([0.7, self.buttonHeight, self.buttonWidth, self.buttonHeight])
        self.close_button = widgets.Button(self.close_ax, 'Finish')
        self.close_button.on_clicked(self.Close_gui)

    def Init(self, _oframe):
        # cv2.imshow("__init__ origin frame: ", _oframe)

        self.frame = copy.deepcopy(_oframe)
        (h ,w, _) = self.frame.shape
        self.type0List = [item for item in self.selectList if len(item) >= 0 and 0 <= item[0] < markCountPerType]
        self.type1List = [item for item in self.selectList if len(item) >= 0 and markCountPerType <= item[0] < markCountPerType*2]
        # 创建GUI
        self.fig, self.ax = plt.subplots(figsize=(w * 0.008, h * 0.008 + 1))

        self.ax.set_position([0.04, self.buttonWidth, 0.94, 0.98])

        self.DrawAllContent()
        
        self.ax.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

        self.createButton()

        plt.show()

    def ClearList(self, event):
        clearAll = True if event.inaxes == self.clear_ax else False
        if len(self.selectList):
            if clearAll:
                self.selectList.clear()
            else:
                print("deleted: [" + ",".join([str(s) for s in self.selectList[-1]]) + "]")
                self.selectList = self.selectList[:-1]
            plt.close(self.fig)
            self.Init(self.oframe)
    
    def CircleArray(self, event):#仅支持圆形环形阵列
        if len(self.selectList):
            last:list[int] = self.selectList[-1]
            if last[1] != 0:
                return

            mark = last[0]
            markTypeMax = (mark // markCountPerType + 1) * markCountPerType
            center_x = self.sceneInfo[0]
            center_y = self.sceneInfo[1]
            
            original_center_x = last[2]
            original_center_y = last[3]
            oroginal_angle = math.atan2(original_center_y - center_y, original_center_x - center_x)
            radius = last[4]
            inner = last[5]
            
            # 计算环的半径（形状中心到旋转中心的距离）
            ring_radius = math.hypot(original_center_x - center_x, original_center_y - center_y)
            
            for i in range(7):
                if mark + i < markTypeMax:
                    angle_deg = (i + 1) * 360 / 8  # degrees
                    angle_rad = math.radians(angle_deg) + oroginal_angle
                    # 计算新的圆心坐标
                    new_center_x = center_x + ring_radius * math.cos(angle_rad)
                    new_center_y = center_y + ring_radius * math.sin(angle_rad)
                    # 保持半径不变
                    new_shape = [mark + i + 1, 0, int(new_center_x + 0.5), int(new_center_y + 0.5), radius, inner]
                    self.selectList.append(new_shape)
            plt.close(self.fig)
            self.Init(self.oframe)

    def add_to_list(self, event):
        selectPlace: list[int] = [-1, -1, -1, -1, -1, -1]
        # self = param
        fig = self.fig
        frame = self.frame
        selectList = self.selectList
        plt.close(fig)
        ButtonLabel = 0 if event.inaxes in [self.button0circle_ax, self.button0rect_ax] else 1
        selectType = 0 if event.inaxes in [self.button0circle_ax, self.button1circle_ax] else 1
        if (ButtonLabel == 0 and len(self.type0List) < markCountPerType) or (ButtonLabel == 1 and len(self.type1List) < markCountPerType):
            if selectType == 0:#circle
                selectPlace[0] = len(self.type0List) if ButtonLabel == 0 else markCountPerType + len(self.type1List)
                selectPlace[1] = 0
                _center, radius, inner = defineCircle.define_circle_by_center_and_point(frame)
                if(_center != None):
                    selectPlace[2:6] = [_center[0], _center[1], radius, 1 if inner else 0]
                else:
                    return
            elif selectType == 1:#rect
                selectPlace[0] = len(self.type0List) if ButtonLabel == 0 else markCountPerType + len(self.type1List)
                selectPlace[1] = 1
                gROI = cv2.selectROI("ROI frame", frame, False)
                selectPlace[2:6] = [gROI[0], gROI[1], gROI[0] + gROI[2], gROI[1] + gROI[3]]
                cv2.destroyWindow("ROI frame")

            selectList.append(selectPlace)
            print("added: [" + ",".join([str(s) for s in selectPlace]) + "]")
        else:{
            print("列表已满")
        }

        self.Init(self.oframe)

    def Close_gui(self, event):
        self.selectListRef.clear()
        for select in self.selectList:
            self.selectListRef.append(select)
        plt.close(self.fig)

#endregion

# region ----------------------------------------communicate-----------------------------------
class ProcessingCommunicate:
    def __init__(self, port:int, test:bool = False):
        if port == -1:
            print("inaviable port!")
            Quit()
        self.care = ""
        self.test = test
        self.msgId = -1
        self.port = port
        self.connected = False
        self.socketInstance = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Connect()

    def __del__(self):
        self.Disconnect()
        print("socket closed")
    
    def Connect(self, slient:bool = False):
        if self.port != -1:
            address = ('127.0.0.1', self.port)
            if self.socketInstance.connect_ex(address) != 0:
                if not slient:
                    print("Failed to connect to server")
            else:
                self.connected = True
                self.msgId = 0
                if CameraType == "basler":
                    BaslerSyncEnable()

        return True if self.connected else False

    def CheckOnlineClientsCount(self):
        return 1 if self.connected else 0
    
    def Disconnect(self):
        self.socketInstance.close()
        self.connected = False
        BaslerSyncEnable(False)

    def WriteContent(self, msg:str):
        if self.connected:
            self.msgId += 1
            msg = str(self.msgId) + msg if self.test else msg
            try:
                if self.socketInstance.sendall(msg.encode()) != None:
                    print(f"Failed to send message: {msg}")
            except Exception as e:#ConnectionError
                self.connected = False
                self.msgId = -1
                print(f"error occured in sending message: {msg}, exception: {e}")
        else:
            if self.test:
                print("lost connection to server")

    def InitBuffer(self):
        return True
    def UpdateOnlineStatus(self):
        pass
    def ShowAllData(self):
        pass
    
#endregion
def getFrame() -> tuple[bool, np.ndarray, int]:
    global frameIndForSingleThread
    if multiThread:
        ret, frameInd, frame = grabber.get_last_frame()
        # grabber.clear_buffer()
        return ret, frame, frameInd
    else:
        try:
            if CameraType == "basler":
                # _t = time.time()
                if camera.WaitForFrameTriggerReady(baslerWaitTime, pylon.TimeoutHandling_ThrowException):
                    camera.ExecuteSoftwareTrigger()
                    # print("this step?")
                # grabResult = self.baslerCamera.GrabOne(100, pylon.TimeoutHandling_ThrowException)#
                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                # print(f"grab time:{time.time()-_t}")
                ret =  grabResult is not None
                # ret =  grabResult.GrabSucceeded()
                if not ret:
                    for i in range(3):
                        print(f"failed to get frame {i+1} times")
                        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                        frame = cv2.cvtColor(np.array(grabResult.Array, np.uint8), cv2.COLOR_GRAY2RGB)
                        ret =  grabResult.GrabSucceeded()
                        if ret:
                            break
                    print("lost connection to basler")
                frame = cv2.cvtColor(np.array(grabResult.Array, np.uint8), cv2.COLOR_GRAY2RGB)
                frameIndForSingleThread += 1
                return ret, frame, frameIndForSingleThread
            else:
                ret, frame,  = camera.read()
                if not ret:
                    print("no camera connected")
                frameIndForSingleThread += 1
            return ret, frame, frameIndForSingleThread
        except Exception as e:
            print(e)
            return False, None, None
    
def TrygetFrame(waitTime:float = 0.01) -> tuple[bool, np.ndarray, int]:
    timer:list[float] = [time.time(), -1]
    while True:
        ret, frame, frameInd = getFrame()
        if ret:
            return ret, frame, frameInd
        else:
            if timer[1] == -1:
                timer[1] = time.time()
                time.sleep(waitTime * 0.05)
            if time.time() - timer[1] > waitTime:
                return ret, frame, frameInd
            
def GetResultOrPutTask(frame:np.ndarray, frameInd:int) -> tuple[bool, list[int], int]:
    results = []
    if multiThread:
        if task_queue.empty():
            task_queue.put((frame, frameInd))

        if not result_queue.empty():
            results, resultInd = result_queue.get(timeout=0.01)
            return True, results, resultInd
        else:
            return False, [], -1
    else:
        results = Predict(frame)
        return True, results, frameInd

grabber = None
if multiThread:
    grabber = FrameGrabber(_cameraType= CameraType, fps=FPS)
    grabber.start()
    if not CCare:
        grabber.startRecord()
    threading.Thread(target=Predict, daemon=True).start()

def Quit():
    if multiThread:
        grabber.stop()
    quit()

fristFrame = None
selectSceneMask = None
selectMask = None
connectSignMask = None
PreBoolMask = None
ret, fristFrame, _ = TrygetFrame(1)
if not ret:
    print("no camera connected")
    Quit()
selectSceneMask = np.zeros_like(fristFrame)
selectMask = np.zeros_like(fristFrame)

if len(sceneInfo) == 0:
    sceneInfo = [None]
    while(sceneInfo[0] is None):
        sceneCenter, sceneRadius, sceneAngle = defineCircle.define_circle_by_three_points(fristFrame)
        # sceneAngle -= 90 #defineCircle中以正上方为0
        if sceneCenter is None:
            if PyWinMessageBox.YesOrNo("Use Rectange instead?", "scene info define") == 'YES':
                gROI = cv2.selectROI("scene frame", fristFrame, False)
                sceneInfo = [gROI[0], gROI[1], gROI[0] + gROI[2], gROI[1] + gROI[3], 1]
                print("created: [" + ",".join([str(s) for s in sceneInfo]) + "]")
                cv2.destroyWindow("scene frame")
                break
            continue
        else:
            sceneInfo = [sceneCenter[0], sceneCenter[1], sceneRadius, sceneAngle, 0]
        print("created: [" + ",".join([str(s) for s in sceneInfo]) + "]")

    selectChanged = True
else:
    sceneCenter = (int(sceneInfo[0]), int(sceneInfo[1]))
    sceneRadius = int(sceneInfo[2])
    sceneAngle = sceneInfo[3]

sceneType = "rect" if len(sceneInfo) == 5 and sceneInfo[4] == 1 else "circle"
if sceneType == "circle":
    selectSceneMask = cv2.circle(selectSceneMask, sceneCenter, sceneRadius, (0, 255, 0), 2)
    selectSceneMask = CircleSelect.draw_arrow(selectSceneMask, sceneCenter, sceneRadius, sceneAngle, (0, 255, 0), 2)
elif sceneType == "rect":
    selectSceneMask = cv2.rectangle(selectSceneMask, (int(sceneInfo[0]), int(sceneInfo[1])), (int(sceneInfo[2]), int(sceneInfo[3])), (0, 255, 0), 2)
drawSelectArea(selectMask, selectAreas)
PreBoolMask = ~(selectMask.any(axis=-1))
selectMask[PreBoolMask] = selectSceneMask[PreBoolMask]
PreBoolMask = selectMask.any(axis=-1)

availableMask = np.ones(fristFrame.shape, dtype= np.uint8)
if os.path.exists("tempMask.jpg"):
    tempMask = cv2.imread("tempMask.jpg")
else:
    tempMask = None
if(type(tempMask) != type(None) and tempMask.shape == fristFrame.shape):
    availableMask = tempMask
else:
    while True:         
        gROI = cv2.selectROI("ROI frame", fristFrame * availableMask, False)

        availableMask[gROI[1]:(gROI[1] + gROI[3]), gROI[0]:(gROI[0] + gROI[2])] = 0

        if keyboard.is_pressed('esc'):
            cv2.destroyWindow("ROI frame")
            break
            
        if gROI == (0,0,0,0):
            cv2.destroyWindow("ROI frame")
            break
    cv2.imwrite("tempMask.jpg", availableMask)

def simulateMousePosUpdate(event, x, y, flags, param):
    simulateMousePos[0:2] = [x, y]

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", simulateMousePosUpdate)

ProcessStartTime = time.time()

if CType == "unity":
    CInstance = IPCTest.SharedMemoryObj('UnityShareMemoryTest', "server", "UnityProject" if CCare else "", 32+5*16*1024)#~80KB
else:
    CInstance = ProcessingCommunicate(CPort)
if not CInstance.InitBuffer():
    Quit()

if CCare:
    threading.Thread(target=CareMessageReceive, daemon = True).start()

if performanceAnalysis:
    profiler = Profiler()
    profiler.start()

if detectMethod == 'blob':
    _, frame, _ = TrygetFrame()
    BlobPredictInit(frame)

fps:float = 0
hideAltInterval = 1
hideAltTime = -1
quitAfterClientOffline = False

while CameraType != "basler" or (multiThread or camera.IsGrabbing()):
    ret, frame, frameInd = getFrame()
    if not ret:
        print("frame stream stoped")
        break
    rectedFrame = frame * availableMask

    if recordPredictResult:
        if not multiThread:
            WriteFrame(frame, True)
        outWritten:bool = False

    # if realMouseCenter[0] > 0:
    #     Predictframe = ProcessMouseNearRegion(realMouseCenter, frame)
    # else:
    #     # Predictframe = copy.deepcopy(frame) * availableMask
    #     Predictframe = frame * availableMask
    Predictframe = frame * availableMask

    if startTime < 0:
        startTime = time.time()
    if not ret:
        break

    receiveUnityTimeSuccess = -1
    lastReceiveUnityTime = -1
    onlineNumber = CInstance.CheckOnlineClientsCount()
    if CInstance.care != "" and CType == "unity":
        if onlineNumber > 0:
            while not message_receive_queue.empty():
                message:str = message_receive_queue.get_nowait()
                print(f"from unity: {message}")
                if message.startswith("cmd:"):
                    if message[4:] == "quit":
                        print("quit command received")
                        # Quit()
                        quitAfterClientOffline = True
            if CInstance.careindex == -1:
                CInstance.CheckApplies()
            else:
                if onlineNumber > 0 and not sync:
                    syncTryTimes = 0
                    syncTryTimesMax = 100
                    while(syncTryTimes < syncTryTimesMax):
                        timeMsg = CInstance.ReadToStr(CInstance.careindex)
                        timeMsg.reverse()
                        for msg in timeMsg:
                            if msg.startswith("time:"):
                                print("from unity: "+msg)
                                temptime = float(msg[5:])
                                if receiveUnityTimeSuccess < 1:#至少连续接收两次
                                    print("success: "+ str(temptime))

                                    if lastReceiveUnityTime == -1:
                                        lastReceiveUnityTime = temptime
                                        print("lastReceiveUnityTime init: "+ str(lastReceiveUnityTime))

                                    else:
                                        print("lastReceiveUnityTime update: "+ str(lastReceiveUnityTime))
                                        if temptime - lastReceiveUnityTime < 0.05:#顺利接收
                                            receiveUnityTimeSuccess += 1
                                            unityFixedUscaledTimeOffset = time.process_time() - createdTime - temptime
                                            lastReceiveUnityTime = temptime
                                            for i in range(10):
                                                CInstance.WriteContent("scene:" + f"{";".join([str(s) for s in sceneInfo])};{len(selectAreas)}")
                                                time.sleep(0.01)
                                            sync = True
                                            syncInd = 0
                                            CInstance.WriteClear()
                                            for i in range(5):
                                                for selectedAreaSync in selectAreas:
                                                    CInstance.WriteContent("select:" + ";".join(map(str, selectedAreaSync)))
                                                    time.sleep(0.01)

                                            print("sync succeed")
                                            syncTryTimes = syncTryTimesMax
                                            BaslerSyncEnable()

                                            if multiThread:
                                                grabber.startRecord()
                                            else:
                                                startRecord()

                                            break

                                        else:
                                            print("interval:" + str(temptime - lastReceiveUnityTime))
                                            receiveUnityTimeSuccess = -1
                                            lastReceiveUnityTime = -1
                                            syncInd = -1
                                    break

                                else :
                                    if abs(unityFixedUscaledTimeOffset - (time.process_time() - createdTime - temptime)) > 0.5:
                                        print("still too lag")
                                        receiveUnityTimeSuccess = -1
                                        lastReceiveUnityTime = -1
                                        syncInd = -1
                        CInstance.WriteContent(f"time:{time.process_time() - createdTime}")
                        syncTryTimes += 1
                        time.sleep(0.01)
                    # UnityShm.ReadToStr(1)
                # UnityShm.WriteContent("scene:" + f"{sceneCenter[0]};{sceneCenter[1]};{sceneRadius:.2f};{sceneAngle:.2f}")
            # else:
            #     UnityShm.WriteContent(f"time:{time.process_time() - createdTime}")
            #     UnityShm.ReadToStr(1)
        elif onlineNumber == 0:
            if sync:
                print("0 online member")
                BaslerSyncEnable(False)

            receiveUnityTimeSuccess = -1
            unityFixedUscaledTimeOffset = 0
            sync = False
            if multiThread and grabber.Isrecording():
                grabber.VideoClearPublic()
            elif not multiThread:
                VideoClear()

            if quitAfterClientOffline:
                Quit()

        else:#sync = true
            readMsg = CInstance.ReadToStr(CInstance.careindex)
    elif CType == "processing":
        if onlineNumber <= 0 and processingcConnectRetryTime - time.time() >= ConnectRetryInterval:
            if CInstance.Connect(slient = True):
                print("connected")
            else:
                processingcConnectRetryTime = time.time()

    CommunicatePrepared:bool = (CInstance.care != "" and sync) or CInstance.care == ""

    # if UnityShmPrepared:
    resultInd = -1
    if CommunicatePrepared:

        if not multiThread or lastframeInd < 0 or lastframeInd != frameInd:
            _p_ret, results, resultInd = GetResultOrPutTask(Predictframe, frameInd)
            if _p_ret:
                lastframeInd = resultInd
                syncInd += 1 if syncInd >= 0 else 0

        # if multiThread:
        #     if task_queue.empty() and (lastframeInd < 0 or lastframeInd != frameInd):
        #         task_queue.put((Predictframe, frameInd))

        #     if not result_queue.empty():
        #         results, resultInd = result_queue.get_nowait()
        #         lastframeInd = resultInd
        #         # print(resultInd)
        #         syncInd += 1 if syncInd >= 0 else 0

        #     else:
        #         results = []
        #         # resultInd = lastframeInd
        # else: 
        #     results = model.Predict(Predictframe, task= Task)
        #     resultInd = frameInd
            # syncInd += 1 if syncInd >= 0 else 0

    else:
        results = []

    
    rectedFrame[PreBoolMask] = selectMask[PreBoolMask]

    if type(results) == list and len(results) == 2:
        realMouseCenter = results
        rectedFrame = cv2.circle(rectedFrame, realMouseCenter, 5, (255,255,0), 10)
        
        if recordPredictResult:
            WriteFrame(rectedFrame, False)
            outWritten = True

        if not hide:
            cv2.putText(rectedFrame, f"fps: {fps:.2f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, FontSize, (0, 0, 255), FontThick)
            cv2.imshow("frame", rectedFrame)
            cv2.waitKey(1)
    else:
        
    # index = index +1
        realMouseCenter = [-1, -1]
        missed_frame_count += 1
        if recordMissframe and CommunicatePrepared and missed_frame_count % missed_frame_rate_divider == 0:
            cv2.imwrite(missedFrameSaveFolder + timestr + f"frame{frame_count}.jpg", frame)

        if not hide:
            cv2.putText(rectedFrame, f"fps: {fps:.2f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, FontSize, (0, 0, 255), FontThick)
            cv2.imshow("frame", rectedFrame)
            cv2.waitKey(1)

    frame_count += 1
    
    posRes = None
    '''
    刚同步上时，因为等待延迟导致刚向predict queue传入1~2次新帧后又卡半秒，并且刚传入还没来得及取即开始等待，导致结果中frameInd会发生1-25的跳变，而时间只差20ms
    '''
    if not simulate:
        if resultInd != -1 and type(results) != type(None) and results[0] >= 0:
            # UnityShm.WriteContent("pos" + ";".join([str(i) for i in xyxy]), True)
        # UnityShm.WriteClear()
            temp = realMouseCenter.copy()
            temp.append(syncInd if syncInd >= 0 else -1)
            temp.append(int((time.time() - ProcessStartTime) * 100))
            temp.append(resultInd)
            simulateMousePos[2] = syncInd if syncInd >= 0 else -1
            CInstance.WriteContent("pos:" + ";".join([str(i) for i in temp]))
            posRes = temp
            # print(f"pos:{realMouseCenter}")

    # break
    else:
        simulateMousePos[2] = syncInd if syncInd >= 0 else -1
        simulateMousePos[3] = int((time.time() - ProcessStartTime) * 100)
        simulateMousePos[4] = frameInd
        # print(simulateMousePos)
        CInstance.WriteContent("pos:" + ";".join([str(i) for i in simulateMousePos]))
        # posRecFile.write(",".join([str(i) for i in simulateMousePos]) + "\n")
        posRes = simulateMousePos.copy()

    if posRecFile is not None and posRes is not None:
        posRecFile.write(",".join([str(i) for i in posRes]) + "\n")
        # posRecFile.flush()


    if (frame_count + 1) % 60 == 0:
        costTime = time.time() - startTime
        # print(str(60/costTime)+"fps")
        startTime = time.time()
        if costTime > 0:
            fps = 60/costTime
            CInstance.UpdateOnlineStatus()#作为server不需要检查返回值

            if posRecFile is not None:
                posRecFile.flush()

    _time = time.time()
    if not hide:
        # cv2.putText(rectedFrame, f"fps: {fps:.2f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, FontSize, (0, 0, 255), FontThick)
        # cv2.imshow("frame", rectedFrame)
        # cv2.waitKey(1)
        
        if keyboard.is_pressed("shift+h"):
            if _time > hideAltTime:
                hide = not hide
                hideAltTime = _time + hideAltInterval
                if hide:
                    cv2.destroyWindow("frame")
        elif keyboard.is_pressed("shift+space") and detectMethod == 'blob':
            BlobPredictInit(frame)
        if not ignoreKeyboardEvent:
            if keyboard.is_pressed("shift+v"):
                if CommunicatePrepared:
                    CInstance.ShowAllData()
                    while keyboard.is_pressed("shift+v"):
                        continue
            elif keyboard.is_pressed("shift+s"):
                gui = GUI(frame, selectAreas, sceneInfo)

                selectMask.fill(0)
                drawSelectArea(selectMask, selectAreas)
                PreBoolMask = ~(selectMask.any(axis=-1))
                selectMask[PreBoolMask] = selectSceneMask[PreBoolMask]
                PreBoolMask = selectMask.any(axis=-1)

                oselect_set = set(map(tuple, gui.oselectList))
                select_set = set(map(tuple, selectAreas))

                added = select_set - oselect_set
                deleted = oselect_set - select_set
                if len(added) != 0 or len(deleted) != 0:
                    selectChanged = True
                if CommunicatePrepared:
                    for item in added:
                        CInstance.WriteContent("select:" + ";".join(map(str, item)))

                    # 处理被删除的内容
                    for item in deleted:
                        item_list = list(item)
                        item_list[0] = (item_list[0] + 1) * -1
                        CInstance.WriteContent("select:" + ";".join(map(str, item_list)))

            elif keyboard.is_pressed("shift+space"):
                cv2.waitKey()
            elif keyboard.is_pressed("shift+m"):
                simulate = not simulate
                print("simulate " + ("on" if simulate else "off"))
                while keyboard.is_pressed("shift+m"):
                    continue
        else:
            if keyboard.is_pressed("shift+h"):
                if _time > hideAltTime:
                    hide = not hide
                    hideAltTime = _time + hideAltInterval
            # if (frame_count + 1) % 60 == 0:
            #     print(f"fps: {fps:.2f}")

    if keyboard.is_pressed("shift+esc"):
        if not sync or PyWinMessageBox.YesOrNo("Unity Project still online, force exit?", "Warning") == "YES":
            break

if performanceAnalysis:
    profiler.stop()
    profiler.print()

if selectChanged and PyWinMessageBox.YesOrNo("save current Info?", "save & load") == 'YES':
    f_selectionSaveTxt.seek(0, 0)
    f_selectionSaveTxt.truncate(0)
    f_selectionSaveTxt.write("scene:"+";".join([str(i) for i in sceneInfo]) + "\n")
    for selectArea in selectAreas:
        f_selectionSaveTxt.write("selectAreas:"+";".join([str(i) for i in selectArea]) + "\n")

f_selectionSaveTxt.close()

if multiThread:
    grabber.stop()
    grabber.join()
    print("thread released")

elif camera != None:
    if CameraType == "basler":
        camera.StopGrabbing()
        camera.Close()
    else:
        camera.release()

if recordResult and outRaw != None:
    if recordPredictResult:
        out.release()
    if not multiThread and outRaw != None:
        outRaw.release()
    if posRecFile != None:
        posRecFile.close()
    print("video stream released")
cv2.destroyAllWindows()
del CInstance