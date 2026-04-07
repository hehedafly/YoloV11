import os
import cv2
import time
import random
from pathlib import Path
import copy
from ultralytics import YOLO
from typing import Literal, List, Tuple, Dict, Any, Optional, Callable
from skimage.metrics import structural_similarity as ssim
from math import*
import numpy as np
import keyboard

KNN_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows = False)

MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

bg_subtractor=MOG2_subtractor

mediaName = r"E:\pythonFiles\YoloV8\output_clips\12_26_1907outputraw_clip.mp4"
# mediaName = "OutputMissedFramePic1012124927"
mediaNamePure = Path(mediaName).stem if mediaName.endswith('.mp4') else mediaName
useModel = False 
recordMissedFrame = True
extractStaticObject = False
staticObjectROIs:Dict[int, np.ndarray] = {}
prev_roi_gray:Dict[int, np.ndarray] = {}
conf = 0.3
sample = 0.8
minArea = 650
maxArea = 2400
if useModel:
    model = YOLO("models/TopViewMiniscopeBodyBestWithAddition.pt")
picList = []
if(mediaName.endswith('.mp4')):
    camera = cv2.VideoCapture(mediaName)
else:
    camera = None
    if os.path.exists(mediaName):
        for file in os.listdir(mediaName):
            if file.endswith('.jpg'):
                if random.randrange(0, 10, 1) < (sample * 10):
                    picList.append(os.path.join(mediaName, file))
    else:
        print("folder not exist")
        quit()

waitMillSec = 1

show = True
PoseExtract = True if not useModel else False
PoseExtractDivider = 20
recFrame = 0
recDivider = 5
timeStr = time.strftime("%m%d%H%M%S", time.gmtime())
if not PoseExtract:
    tempPicFolderName = "OutputMouseBodyPic" + timeStr
    # tempTxtFolderName = "OutputMouseBodyTxt" + timeStr
    tempTxtFolderName = tempPicFolderName
    tempROIFolderName = tempPicFolderName
    tempStaticObjectFolderName = "OutputStaticObjectPic" + timeStr
    tempStaticObjectTexFolderName = tempStaticObjectFolderName
    tempMissedFrameFolderName = "OutputMissedFramePic" + timeStr
    if not os.path.exists(tempPicFolderName):
        os.makedirs(tempPicFolderName)
    if not os.path.exists(tempTxtFolderName):
        os.makedirs(tempTxtFolderName)
    if not os.path.exists(tempROIFolderName):
        os.makedirs(tempROIFolderName)
    if not os.path.exists(tempStaticObjectFolderName):
        os.makedirs(tempStaticObjectFolderName)
    if not os.path.exists(tempStaticObjectTexFolderName):
        os.makedirs(tempStaticObjectTexFolderName)
    if recordMissedFrame and not os.path.exists(tempMissedFrameFolderName):
        os.makedirs(tempMissedFrameFolderName)

# tailColor = 200
else:
    show = False
    continueShoot = True
    recCount = 0

    tempPosePicFolderName = "PoseExtract" + timeStr
    # tempTxtFolderName = "OutputMouseBodyTxt" + timeStr
    tempPosTxtFolderName = tempPosePicFolderName
    tempPoseROIFolderName = tempPosePicFolderName
    if not os.path.exists(tempPosePicFolderName):
        os.makedirs(tempPosePicFolderName)
    if not os.path.exists(tempPosTxtFolderName):
        os.makedirs(tempPosTxtFolderName)

def getFrame() -> tuple[bool, np.ndarray, str]:
    if camera is None and len(picList) > 0:
        fname = picList.pop(0)
        pic = cv2.imread(fname)
        
        return True, pic, fname
    elif camera is not None:
        ret, frame = camera.read()
        return ret, frame, ""
    else:
        return False, None, ""

_, fristFrame, _ = getFrame()

[height, width, _] = fristFrame.shape
availableMask = np.ones(fristFrame.shape, dtype= np.uint8)

# PosKeyPointsName = ["鼻尖", "左耳", "右耳", "尾根"]
# PosKeyPointsName = ["鼻尖", "头顶", "身体", "尾根"]
PosKeyPointsName = ["tip", "head", "body", "tailbase"]
PosKeyPointsCount = len(PosKeyPointsName)
clickColorLs:list[list[int]] = []
for i in range(1, PosKeyPointsCount + 1):
    tempcolor = i * (255 // PosKeyPointsCount)
    clickColorLs.append([tempcolor, 255 - abs(255 - int(tempcolor * 2)), 255 - tempcolor])

# def ClickEvent(event, x, y, flags, posLs:list[list[int]], colorMask:np.ndarray):
def ClickEvent(event, x, y, flags, param):
    # if cv2.getWindowProperty("poseMark") != -1:
    tempMousePos = param[0]
    tempMousePos[0], tempMousePos[1] = x, y

    if posLs != None and len(posLs) < PosKeyPointsCount:
        if event == cv2.EVENT_LBUTTONDOWN:#可见点
            posLs.append([x, y, 2])
            # cv2.circle(colorMask, (x, y), 2, clickColorLs[len(posLs)], 4)
            # colorMask
        elif event == cv2.EVENT_RBUTTONDOWN:#不可见点
            posLs.append([x, y, 1])
            # cv2.circle(colorMask, (x, y), 2, clickColorLs[len(posLs)], 1)
        # elif event == -1:
        #     posLs.pop()
        cv2.waitKey(1)

def DrawPosPoints(rawFrame:np.ndarray, x, y, w, h):
    global posLs
    global recCount
    global width
    global height
    global continueShoot

    if not continueShoot:
        return
    posLs = []
    # tempMaskLs:list[np.ndarray] = [] 
    scaled = 4
    h = min(h + 20, height - y)
    w = min(w + 20, width - x)
    img = rawFrame[y:y+h, x:x+w]
    resizedImg = cv2.resize(img, (img.shape[1] * scaled, img.shape[0] * scaled), interpolation=cv2.INTER_LINEAR)
    tempMask = np.zeros(resizedImg.shape)

    enableDel:bool = True
    mousePos = [-1, -1]

    cv2.imshow("poseMark", img)
    cv2.setMouseCallback("poseMark", ClickEvent, param=[mousePos])
    while(True):
        key = cv2.waitKey(1) & 0xff
        # if key != 255:
        #     print(key)
        if key == 8:
            enableDel = True
        if key == 78 and continueShoot == True:
            continueShoot = False
            print("stop shooting")
            cv2.destroyAllWindows()
            break
        if keyboard.is_pressed("shift+esc"):
            exit()

        if len(posLs):
            tempMask = np.zeros(resizedImg.shape)
            for i in range(len(posLs)):
                nowRadius = scaled * 2 if posLs[i][2] == 1 else scaled
                nowThick = 2  if posLs[i][2] == 1 else scaled * 2
                cv2.circle(tempMask, posLs[i][0:2], nowRadius, clickColorLs[i], nowThick)
        bool_mask = (tempMask > 0).astype(bool)
        tempImg = resizedImg.copy()
        tempImg[bool_mask] = tempMask[bool_mask]
        if len(posLs) < len(PosKeyPointsName):
            cv2.putText(tempImg, PosKeyPointsName[len(posLs)], (mousePos[0],mousePos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("poseMark", tempImg)
        if keyboard.is_pressed('backspace') and enableDel:
            enableDel = False
            if len(posLs):
                posLs.pop()
                # print(f"delet one dot, now:{len(posLs)}")
                tempMask = np.zeros(resizedImg.shape)
                for i in range(len(posLs)):

                    cv2.circle(tempMask, posLs[i][0:2], 2, clickColorLs[i], posLs[i][2]*2)
            bool_mask = (tempMask > 0).astype(bool)
            tempImg = resizedImg.copy()
            tempImg[bool_mask] = tempMask[bool_mask]
            cv2.imshow("poseMark", tempImg)
            # tempMask = np.zeros(img.shape) if len(tempMaskLs) == 0 else tempMaskLs[-1]
        elif keyboard.is_pressed('enter'):
            if len(posLs) == PosKeyPointsCount:
                tempPointsStrLs:list[str] = []
                for points in posLs:
                    _x:float = float(points[0]/scaled+x)/width
                    _y:float = float(points[1]/scaled+y)/height
                    tempPointsStrLs.append(" ".join([str(i) for i in [_x, _y, points[2]]]))
                fileName = mediaNamePure + str(int(recCount)) + "pose"
                cv2.imwrite(tempPoseROIFolderName +"/"+ fileName +".png", tempImg)
                fileName = mediaNamePure +str(int(recCount))
                cv2.imwrite(tempPosTxtFolderName +"/"+ fileName +".jpg", rawFrame)
                with open(tempPosTxtFolderName +"/"+ fileName +".txt", "w+") as file:
                    file.write("0 "+ " ".join([str(i) for i in [(x + w*0.5)/width, (y + h*0.5)/height, w/width, h/height]]) + " " + " ".join(tempPointsStrLs))
                recCount += 1
                print(recCount)

                cv2.destroyWindow("poseMark")
                break
            elif len(posLs) == 0:
                 #跳过
                cv2.destroyWindow("poseMark")
                break


if os.path.exists("tempMask.jpg"):
    tempMask = cv2.imread("tempMask.jpg")
else:
    tempMask = None
while True:
    if(type(tempMask) != type(None) and tempMask.shape == fristFrame.shape):
        availableMask = tempMask
        
    gROI = cv2.selectROI("ROI frame", fristFrame * availableMask, False)

    availableMask[gROI[1]:(gROI[1] + gROI[3]), gROI[0]:(gROI[0] + gROI[2])] = 0

    if keyboard.is_pressed('s'):
        cv2.imwrite("tempMask.jpg", availableMask)

    if gROI == (0,0,0,0):
        cv2.destroyWindow("ROI frame")
        break
if extractStaticObject and not PoseExtract:
    while True:
        ROIName = f"static Object {len(staticObjectROIs)}"
        gROI = cv2.selectROI(ROIName, fristFrame, False)

        if keyboard.is_pressed('esc'):
            cv2.destroyWindow(ROIName)
            break
            
        if gROI == (0,0,0,0):
            cv2.destroyWindow(ROIName)
            break
        
        staticObjectROIs[len(staticObjectROIs)] = gROI
        x, y, w, h = gROI
        roi_region = cv2.cvtColor(fristFrame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        prev_roi_gray[len(staticObjectROIs) - 1] = roi_region
        cv2.rectangle(fristFrame, gROI, (0,255,0), 2)
        cv2.destroyWindow(ROIName)

frameInd:int = 0
PoseExtractFrameInd:int = 0

while True:
    ret, frame, fname = getFrame()
    if not ret:
        break

    oframe = frame.copy()
    frame[availableMask == 0] = 255
    # frame = frame * availableMask

    # pixel_sum = np.sum(frame, axis=2)
    # BlackMask = pixel_sum > tailColor
    # frame[mask] = preFrame[mask]
    # preFrame = frame.copy()
    foreground_mask = bg_subtractor.apply(frame)
    # modelPredictfail = False
    
    if useModel:
        # foreground_mask = bg_subtractor.apply(frame)

        # # 如果大于240像素，则阈值设为255，如果小于则设为0    # 创建二值图像，它只包含白色和黑色像素
        # ret , threshold = cv2.threshold(foreground_mask.copy(), 150, 255, cv2.THRESH_BINARY)

        # # 膨胀扩展或加厚图像中的兴趣区域。
        # # threshold[BlackMask] = 0
        # threshold = cv2.medianBlur(threshold, 7)

        results = model(frame, verbose=False, conf = conf)
        
        for result in results:
            if not len(result.boxes):
                if recordMissedFrame:
                    fileName = mediaNamePure + str(int(frameInd))
                    cv2.imwrite(tempMissedFrameFolderName +"/"+ fileName +".jpg", oframe)
                else:
                    print(f"No object detected in file {fname}")
                # modelPredictfail = True
            for box in result.boxes:
                class_id = result.names[box.cls[0].item()]
                # if class_id == 0:
                xyxy = np.array(box.xyxy[0].tolist(), int)

                [x,y,w,h] = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                # tempROI = np.sum(frame[y:y+h, x:x+w], axis= 2) / 3
                # tempResult = np.sum(cv2.threshold(tempROI, 200, 1, cv2.THRESH_BINARY)[1])
                # if recFrame % recDivider == 0:
                #     fileName = mediaNamePure + str(int(recFrame / recDivider)) + "ROI"
                #     cv2.imwrite(tempROIFolderName +"/"+ fileName +".png", frame[y:y+h, x:x+w])
                #     fileName = mediaNamePure +str(int(recFrame / recDivider))
                #     cv2.imwrite(tempTxtFolderName +"/"+ fileName +".jpg", frame)
                #     with open(tempTxtFolderName +"/"+ fileName +".txt", "w+") as file:
                #         file.write("0 "+ " ".join([str(i) for i in [(x + w*0.5)/width, (y + h*0.5)/height, w/width, h/height]]))
                #     print("Marked " + str(int(recFrame / recDivider)+1) + "Frames")
                
                # if show:
                #     cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
                # recFrame+=1
                if not PoseExtract:


                    mouseBlock = np.uint8(np.sum(frame[y:y+h, x:x+w, :], axis=2) / 3)
                    _, tempResult = cv2.threshold(mouseBlock, np.max(mouseBlock) * 0.45, 255, cv2.THRESH_BINARY_INV)
                    # _, tempResult = cv2.threshold(tempResult, np.max(mouseBlock) * 0.25, 255, cv2.THRESH_BINARY)
                    tempResult = cv2.medianBlur(tempResult, 7) 
                    # tempResultMixed = tempResult* np.right_shift(threshold[y:y+h, x:x+w], 7)
                    tempResultMixed = cv2.dilate(tempResult, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 4)
                    # tempContours, tempHier = cv2.findContours(tempResultMixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    tempContours, tempHier = cv2.findContours(tempResult, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # if show:
                    #     cv2.imshow("tempChildPic",tempResult)
                    #     # cv2.imshow("predicResult", cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2))
                    #     # cv2.imshow("tempThreshold", threshold[y:y+h, x:x+w])
                    #     cv2.imshow("tempMixedResult",tempResultMixed)
                    for tempContour in tempContours:
                        tempChildArea = cv2.contourArea(tempContour)
                        (_x,_y,_w,_h) = cv2.boundingRect(tempContour)
                        if tempChildArea > minArea and tempChildArea < maxArea:
                            # print(tempChildArea)
                            if recFrame % recDivider == 0 and 2.5 > _w/_h > 0.4: 
                                fileName = mediaNamePure + str(int(recFrame / recDivider)) + "ROI"
                                cv2.imwrite(tempROIFolderName +"/"+ fileName +".png", oframe[y+_y:y+_y+_h, x+_x:x+_x+_w])
                                fileName = mediaNamePure +str(int(recFrame / recDivider))
                                cv2.imwrite(tempTxtFolderName +"/"+ fileName +".jpg", oframe)
                                with open(tempTxtFolderName +"/"+ fileName +".txt", "w+") as file:
                                    file.write("0 "+ " ".join([str(i) for i in [(x+_x + _w*0.5)/width, (y+_y + _h*0.5)/height, _w/width, _h/height]]))
                                print("Marked " + str(int(recFrame / recDivider)+1) + "Frames")
                            
                                if show:
                                    cv2.rectangle(frame, (x+_x,y+_y), (x+_x+_w, y+_y+_h), (0,255,0), 2)
                                    cv2.putText(frame, str(tempChildArea), (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                
                                break
                        else:
                            if show:
                                cv2.rectangle(frame, (x+_x,y+_y), (x+_x+_w, y+_y+_h), (0,0,255), 2)
                                cv2.putText(frame, str(tempChildArea), (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                            # cv2.imwrite(tempTxtFolderName +"/"+ fileName +".jpeg", frame)
                            print(f"area not fit: {tempChildArea}")
                    recFrame+=1
                else:
                    if keyboard.is_pressed('n') and continueShoot == False:
                        print("continue shooting")
                        continueShoot = True
                    
                    if continueShoot or PoseExtractFrameInd % PoseExtractDivider == 0:
                        DrawPosPoints(frame, max(min(x - 10, width), 0), max(min(y - 10, width), 0), w, h)

                    PoseExtractFrameInd += 1
                        
    
    # if not useModel or modelPredictfail:
    if not useModel:

        if frame.shape[0] < 200:
            x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]
        else:
        # 每一帧既用于计算前景掩码，也用于更新背景。
        # 如果大于240像素，则阈值设为255，如果小于则设为0    # 创建二值图像，它只包含白色和黑色像素
            ret , threshold = cv2.threshold(foreground_mask.copy(), 200, 255, cv2.THRESH_BINARY)

            # 膨胀扩展或加厚图像中的兴趣区域。
            # threshold[BlackMask] = 0
            threshold = cv2.medianBlur(threshold, 7)
            dilated = cv2.dilate(threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 4)

            # 查找轮廓
            contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 检查每个轮廓是否超过某个值，如果超过则绘制边界框
            for contour in contours:
                tempArea = cv2.contourArea(contour)
                if tempArea > minArea*0.6 and tempArea < maxArea*1.5:
                    (x,y,w,h) = cv2.boundingRect(contour)
                    if show:
                        cv2.rectangle(frame, (x,y), (x+w, y+h), (100,100,100), 2)
                    print(f"tempArea: {tempArea}")
                    mouseBlock = np.uint8(np.sum(frame[y:y+h, x:x+w, :], axis=2) / 3)
                    _, tempResult = cv2.threshold(mouseBlock, np.max(mouseBlock) * 0.3, 255, cv2.THRESH_BINARY_INV)
                    # _, tempResult = cv2.threshold(mouseBlock, np.max(mouseBlock) * 0.45, 255, cv2.THRESH_TOZERO_INV)
                    # _, tempResult = cv2.threshold(tempResult, np.max(mouseBlock) * 0.25, 255, cv2.THRESH_BINARY)
                    tempResult = cv2.medianBlur(tempResult, 7) 
                    tempResultMixed = tempResult* np.right_shift(threshold[y:y+h, x:x+w], 7)
                    tempResultMixed = cv2.dilate(tempResultMixed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3)
                    tempContours, tempHier = cv2.findContours(tempResultMixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if show:
                        cv2.imshow("tempChildPic",tempResult)
                        cv2.imshow("tempThreshold", threshold[y:y+h, x:x+w])
                        cv2.imshow("tempMixedResult",tempResultMixed)
                    for tempContour in tempContours:
                        tempChildArea = cv2.contourArea(tempContour)
                        if tempChildArea > minArea and tempChildArea < maxArea:
                            print(tempChildArea)
                            (_x,_y,_w,_h) = cv2.boundingRect(tempContour)
                            if recFrame % recDivider == 0 and 2.5 > _w/_h > 0.4: 
                                fileName = mediaNamePure + str(int(recFrame / recDivider)) + "ROI"
                                cv2.imwrite(tempROIFolderName +"/"+ fileName +".png", frame[y+_y:y+_y+_h, x+_x:x+_x+_w])
                                fileName = mediaNamePure +str(int(recFrame / recDivider))
                                cv2.imwrite(tempTxtFolderName +"/"+ fileName +".jpg", frame)
                                with open(tempTxtFolderName +"/"+ fileName +".txt", "w+") as file:
                                    file.write("0 "+ " ".join([str(i) for i in [(x+_x + _w*0.5)/width, (y+_y + _h*0.5)/height, _w/width, _h/height]]))
                                # print("Marked " + str(int(recFrame / recDivider)+1) + "Frames" + f"{"Detected" if modelPredictfail else ""}")
                                # print("Marked " + str(int(recFrame / recDivider)+1) + "Frames")
                            
                            if show:
                                cv2.rectangle(frame, (x+_x,y+_y), (x+_x+_w, y+_y+_h), (0,255,0), 2)
                    recFrame+=1

            else:
                print(f"no movement dectected in frame{frameInd}")

    if extractStaticObject and not PoseExtract:
        for staticObjectId in staticObjectROIs.keys():
            x, y, w, h = staticObjectROIs[staticObjectId]
            roi_region = cv2.cvtColor(oframe[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            save:bool = False
            score, _ = ssim(prev_roi_gray[staticObjectId], roi_region, full=True)
            if score < 0.9 :
                save = True
                prev_roi_gray[staticObjectId] = roi_region

            if save :
                fileName = mediaNamePure + str(int(recFrame / recDivider)) + "ROI"
                cv2.imwrite(tempStaticObjectFolderName +"/"+ fileName +".png", oframe[y:y+h, x:x+w])
                fileName = mediaNamePure +str(int(recFrame / recDivider))
                cv2.imwrite(tempStaticObjectFolderName +"/"+ fileName +".jpg", oframe)
                with open(tempStaticObjectTexFolderName +"/"+ fileName +".txt", "w+") as file:
                    file.write(f'{staticObjectId + 1} '+ " ".join([str(i) for i in [(x + w*0.5)/width, (y + h*0.5)/height, w/width, h/height]]))
    elif PoseExtract:
        if not keyboard.is_pressed('shift') and keyboard.is_pressed('n') and continueShoot == False:
            print("continue shooting")
            continueShoot = True
        
        if continueShoot or PoseExtractFrameInd % PoseExtractDivider == 0:
            DrawPosPoints(frame, max(min(x - 10, width), 0), max(min(y - 10, width), 0), w, h)

        PoseExtractFrameInd += 1

    frameInd+=1
    
    if show:
        if not useModel:
            # cv2.imshow("Subtractor", foreground_mask)
            cv2.imshow("threshold", threshold)
        else:
            cv2.imshow("detection", frame)
        key = cv2.waitKey(waitMillSec) & 0xff
        # print(key)
        if key != 255:
            if key == 27:
                break
            elif key == 32:
                cv2.waitKey()
            elif key == ord('h'):
                show = False
                cv2.destroyAllWindows()
    else:
        if keyboard.is_pressed('h'):
            show = True

if camera is not None:
    camera.release()
cv2.destroyAllWindows()