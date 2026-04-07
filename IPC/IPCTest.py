import os
import time
import datetime
from multiprocessing import shared_memory
import numpy as np
import traceback
from subprocess import call
import keyboard
import cv2

def IntToBytes(value) -> bytearray:
        upper = value >> 8
        down = value - upper * 256
        return bytearray([upper, down])
def BytesToInt(byteLs) -> int:
    if len(byteLs) == 2:
        return byteLs[0]*256 + byteLs[1]
    else:
        return -1

class SharedMemoryObj:
    '''
    initialization: 0-31 0x00, 32.. 0xFF
    name = server/<custom>
    index = 0(server) or 1-4
    care = certain name("UnityProject")/"", only one allowed in this version, if "": ignore all write/read Mark, else update marks base on cared one
    careindex = index/-1, if cared one online and applied, update to index of cared one, if multiple have same name, take the frist one applied, until its offline, then wait for another apply.

    0:              server Online status(0/1)                                           $write by Server
    1:              max client number(4 max)                                            $write by Server
    2:              now client number(0-4)                                              $write by client, read then check online status
    3-6:            clients online status(0/1)                                          $write by client, check from 0 to 3, frist zero value as client's own index
    7:              client index applied                                                $write by client, check if careindex == -1
    8:              name length of client applied                                       $write by client
    9-31            name of client applied                                              $write by client

    32 - 32+16*1024:Server write buffer
    .. - ..+16:1024:client0 write buffer
    ......

    In every write buffer:
    0:              writing/finish(0/1)                                                 $write by self
    (1-2)*4:        read mark for others(0-4096<0x0F,0x00>), i                          $write by others, if larger than write mark, back to zero
    9-10:           written mark(0-4096),                                               $write by self, if equal to cared one's read mark, back to zero, else if careindex = -1, back to zero when everyone read or ran out of write buffer.
                                                                                                        if ran out of write buffer, back to (writemark - max(readmark)/cared one readmark)< 16 ? ~ : 0
    11-12           newest message start index                                          $write by self
    13-14          newest message end index                                            $write by self
    15..            messages                                                            $write by self

    In every message:
    0-1:            length(0<0x00, 0x00>-15360<0xE0, 0x00>)
    2-..:           content
    between every message: 0xFF, 0xFF
    '''

    def __init__(self, shm_name='default', name = "custom", care = "", size=32+5*16*1024, heartbeat = False):
        global shm_buffer

        self.shm = None
        self.name = name
        self.care = care
        self.UID = -1
        self.careindex = -1
        self.contentbegin = 0
        self.maxClientNum = 4

        self.maxOfflineTick = 5
        self.offlineTick = 0
        self.careofflineTick = 0
        self.careOnlineStatus = [-1] * self.maxClientNum
        self.clientOfflineTick = [-1] * self.maxClientNum
        try:
            self.shm = shared_memory.SharedMemory(name=shm_name, create=(name == "server"), size=size)
            if name == "server":
                print("shared memory \""+shm_name+"\" created with size ≈"+str(int(size/1024))+ "KB")
                self.UID = 0
            else:
                print("mapped to "+shm_name)
        except:
            if name == "server":
                print("shm with same name already exists")
                self.shm = shared_memory.SharedMemory(name=shm_name, create=False, size=size)
                # os.unlink(self.shm)
                self.shm.buf[0] = 0
                self.shm.close()
                self.shm.unlink()
                self.shm = None

            else:
                print("shm doesn't exist")
            self.shm = None
            return

 
        shm_buffer = self.shm.buf
        self.writeBufferStartPos = -1
        self.writebufferLength = 16 * 1024
        self.writeBufferStartPosAll = list(range(32, 32 + (self.maxClientNum + 1) * 16 *1024, 16 *1024))
        self.messageStartPosLs: list[int] = []
        self.newMessageStartPos = -1
        self.newMessageEndPos = -1  #contains [0xFF, 0xFF]split mark
        self.writtenMark = -1

        self.unwrittenMsg = []
        self.heartbeat = heartbeat

        shm_buffer[0:32] = bytearray([0] * 32)
        shm_buffer[32:] = bytearray([0xFF] * len(shm_buffer[32:]))

    
    def __del__(self):
        if self.shm != None:
            if self.name == "server":
                self.shm.buf[0] = 0
                self.shm.close()
                self.shm.unlink()
            else:#应添加server检查care部分
                self.shm.buf[3 + self.UID] = bytearray([0] * len(self.shm.buf[3 + self.UID]))
                self.shm.buf[2] = self.shm.buf[2] - 1
                self.shm.close()
        print("server offline")
        
    def InitBuffer(self) -> bool:
        if self.shm == None:
            return False

        global shm_buffer
        if self.name == "server":
            shm_buffer[0] = 1
            shm_buffer[1] = self.maxClientNum
            shm_buffer[3:7] = bytearray([0] * 4) # client status all init
            shm_buffer[7:31] = bytearray([0] * 24) # client apply init
        else:
            maxNum = shm_buffer[1]
            if shm_buffer[2] < maxNum:
                for i in range(maxNum):
                    if shm_buffer[3 + i] == 0:# client status 0 = offline 1 = online
                        self.UID = i + 1 #client index 1-4
                        shm_buffer[3 + i] = 1

            else:
                print("")#已满
                return False
        self.writeBufferStartPos = self.writeBufferStartPosAll[self.UID]
        shm_buffer[self.writeBufferStartPos : (self.writeBufferStartPos+15)] = bytearray([0] * 15)
        self.writtenMark = 0
        self.newMessageStartPos = 15
        self.newMessageEndPos = 15
        return True

    def ApplyForCare(self):
        global shm_buffer
        shm_buffer[7] = self.UID
        shm_buffer[8] = len(self.name)
        shm_buffer[9:len(self.name)] = self.name.encode()
    
    def CheckApplies(self):
        global shm_buffer
        if self.careindex == -1 and shm_buffer[7] != 0xFF and shm_buffer[8] != 0:#someone applied and match the anme
            applyName = bytes(shm_buffer[9:shm_buffer[8]+9]).decode()
            if applyName == self.care:
                self.careindex = shm_buffer[7]#update care
            shm_buffer[7] = 0xFF#clear apply

    def CheckOnlineClientsCount(self):
        global shm_buffer
        return int(shm_buffer[2])
    
    def CheckOnlineClients(self):
        global shm_buffer
        clientIDs: list[int] = []
        for i in range(0, self.maxClientNum + 1):
            if i != self.UID and shm_buffer[3 + i - 1] >= 1:
                clientIDs.append(i)
        return clientIDs
    
    def UpdateOnlineStatus(self) -> int:
        global shm_buffer
        if not self.heartbeat:
            return 1

        if self.name != "server":
            if shm_buffer[0] == 0:
                return -1  # Server offline
            
            # Update client's own status
            client_status_idx = 2 + self.UID
            shm_buffer[client_status_idx] = (shm_buffer[client_status_idx] % 256) + 1

            # Update server status tracking
            if shm_buffer[0] != self.ca[0]:
                self.careOnlineStatus[0] = shm_buffer[0]
                self.offlineTick = 0
            else:
                self.offlineTick += 1

            # Check offline timeout
            if self.offlineTick == self.maxOfflineTick:
                self.offlineTick = 0
                return -2
            else:
                return 1
        else:
            # Update server heartbeat
            shm_buffer[0] = (shm_buffer[0] % 256) + 1

            # Monitor client statuses
            for i in range(self.maxClientNum):
                client_idx = 3 + i
                if shm_buffer[client_idx] > 0:
                    if shm_buffer[client_idx] != self.careOnlineStatus[i]:
                        self.careOnlineStatus[i] = shm_buffer[client_idx]
                        self.clientOfflineTick[i] = 0
                    else:
                        self.clientOfflineTick[i] += 1

                        # Handle client timeout
                        if self.clientOfflineTick[i] >= self.maxOfflineTick:
                            shm_buffer[2] = (shm_buffer[2] - 1) % 256  # Ensure byte wrap
                            shm_buffer[client_idx] = 0
                            self.clientOfflineTick[i] = 0
                            print("client "+str(i+1)+" offline")
            return 1
    
    def WriteClear(self, clearPos = 0, esayclear = False):#clearPos: 0: all, 1-messages: count of latest messages shall be saved 
        global shm_buffer

        shm_buffer[self.writeBufferStartPos] = 0 #写开始
        if esayclear:
            shm_buffer[self.writeBufferStartPos + 15: self.writeBufferStartPos + self.newMessageEndPos] = bytearray([0xFF] * (self.newMessageEndPos - 15))
            self.writtenMark = 0
            self.newMessageStartPos = 15
            self.newMessageEndPos = 15
            shm_buffer[self.writeBufferStartPos + 9: self.writeBufferStartPos + 11] = IntToBytes(self.writtenMark)#write mark
            shm_buffer[self.writeBufferStartPos + 11: self.writeBufferStartPos + 15] = bytearray([0] * 4)
        else:
            unreadMsg: bytearray = bytearray([])

            if clearPos and clearPos <= len(self.messageStartPosLs):#备份将被清理的数据
                unreadMsg = shm_buffer[self.messageStartPosLs[-1 * clearPos] : self.newMessageEndPos]
                self.messageStartPosLs = self.messageStartPosLs[clearPos :]
            # elif clearPos == 0 and self.careindex != -1:
            #     careUnreadIndex = BytesToInt(shm_buffer[32 + self.careindex * 2 - 1 : 32 + self.careindex * 2 + 1]) - 1 #cared one's read message count
            #     careUnreadIndex = min(careUnreadIndex, len(self.messageStartPosLs) - 1)
            #     if careUnreadIndex:
            #         unreadMsg = shm_buffer[self.messageStartPosLs[careUnreadIndex] : self.newMessageEndPos]
            #         self.messageStartPosLs = self.messageStartPosLs[careUnreadIndex :]
            #     else:
            #         self.messageStartPosLs = []
            else:
                    self.messageStartPosLs = []
            
            shm_buffer[self.writeBufferStartPos + 15 : self.writeBufferStartPos + self.writebufferLength] = bytearray([0xFF] * (self.writebufferLength - 15))
            if clearPos or self.careindex != -1:
                shm_buffer[self.writeBufferStartPos + 15 : self.writeBufferStartPos + 15 + len(unreadMsg)] = unreadMsg
            
            self.writtenMark = len(self.messageStartPosLs)
            shm_buffer[self.writeBufferStartPos + 9: self.writeBufferStartPos + 11] = IntToBytes(self.writtenMark)#write mark
            self.newMessageStartPos = 15 + 0 if len(self.messageStartPosLs) == 0 else (self.messageStartPosLs[-1] - self.messageStartPosLs[0])
            self.newMessageEndPos = self.newMessageStartPos + len(unreadMsg)#unreadMsg包括split condon
        shm_buffer[self.writeBufferStartPos] = 1#写结束


    def WriteContent(self, _string:str, clear = False, waitEvenIfFilled = False) -> bool:
        global shm_buffer
        if len(_string):
            if clear or self.newMessageEndPos + len(_string) + 2 - self.writeBufferStartPos > self.writebufferLength:
                self.WriteClear(esayclear= clear)
            else:
                self.CheckReadMarkInOwnWriteBuffer()
            if waitEvenIfFilled:
                self.unwrittenMsg.append(_string)
                return False
            shm_buffer[self.writeBufferStartPos] = 0
            self.writtenMark += 1
            self.newMessageStartPos = self.newMessageEndPos
            self.newMessageEndPos = self.newMessageStartPos + (len(_string) + 2) + 2#2for length mark, 2for split condon

            shm_buffer[self.writeBufferStartPos + 9: self.writeBufferStartPos + 11] = IntToBytes(self.writtenMark)
            
            shm_buffer[self.writeBufferStartPos + 11 : self.writeBufferStartPos + 13] = IntToBytes(self.newMessageStartPos)
            shm_buffer[self.writeBufferStartPos + 13 : self.writeBufferStartPos + 15] = IntToBytes(self.newMessageEndPos)
            shm_buffer[self.writeBufferStartPos + self.newMessageStartPos : self.writeBufferStartPos + self.newMessageEndPos - 2] = IntToBytes(len(_string)) + _string.encode()
            shm_buffer[self.writeBufferStartPos] = 1

            self.messageStartPosLs.append(self.newMessageStartPos)
            return True
        else:
            return False
        
    def CheckReadMarkInOwnWriteBuffer(self):
        if self.careindex != -1: #check others' readmark is not necessary, just continue writing until overwrite then clear in next write
            return  
        else:#if cared one have read all the msg, clear all
            careIdPos = self.UID if self.UID < self.careindex else self.UID - 1
            careReadmark = BytesToInt(shm_buffer[self.writeBufferStartPos + 1 + careIdPos * 2 : self.writeBufferStartPos + 1 + careIdPos * 2 + 2])
            if self.writtenMark >= 20 and self.writtenMark - careReadmark <= 10:
                self.WriteClear(self.writtenMark - careReadmark)
            

    def Read(self, readId, readMethod = "new") -> list[bytes]:
        global shm_buffer
        
        if self.CheckOnlineClientsCount() == 0:
            return []
        if readId < 0 or readId > self.maxClientNum:
            return []
        
        # for i in range(self.CheckOnlineClients()):
        #     tempId = i
        #     while tempId == self.index or shm_buffer[3 + i - 1] == 0:#server 一定在线, i = 0时，对客户端，server shm_buffer[3 + i<0> - 1] = max client number != 0，对server tempId = self.index
        #         tempId += 1
        #     if tempId <= self.maxClientNum:
        #         readPos = self.writeBufferStartPosAll[tempId]
        readPos = self.writeBufferStartPosAll[readId]
        readable = shm_buffer[readPos]
        writePos = self.UID if self.UID < readId else self.UID - 1
        if readable:
            readmark = BytesToInt(shm_buffer[readPos + 1 + writePos * 2 : readPos + 1 + writePos * 2 + 2].tobytes())
            writemark = BytesToInt(shm_buffer[readPos + 9 : readPos + 11].tobytes())
            if writemark >= self.writebufferLength / 4:
                print("wrong write mark read")
                return []
            if writemark != 0 and readmark > writemark:
                readmark = 0
            elif readMethod != "all" and readmark == writemark:
                return []#没有新内容可读
            # else:
            #     
            
            endPos = readPos + BytesToInt(shm_buffer[readPos + 13 : readPos + 15])
            allMsgRaw = bytearray(shm_buffer[readPos + 15 : endPos])
            allMsg = allMsgRaw.split( bytes([0xFF, 0xFF]))
            allMsg = [bytes(msgUnit) for msgUnit in allMsg]
            if readMethod == "all":#无论读没读过，全部返回
                readmark = writemark
                shm_buffer[readPos + 1 + writePos * 2 : readPos + 1 + writePos * 2 + 2] = IntToBytes(readmark)
                res: list[bytes] = allMsg
                return res
            elif readMethod == "new":#一次性返回所有未读
                selectedMsg = allMsg[readmark : writemark]
                readmark = writemark
                shm_buffer[readPos + 1 + writePos * 2 : readPos + 1 + writePos * 2 + 2] = IntToBytes(readmark)
                return selectedMsg
            elif readMethod == "newone":#返回单条未读
                selectedMsg = allMsg[readmark : readmark]
                readmark += 1
                shm_buffer[readPos + 1 + writePos * 2 : readPos + 1 + writePos * 2 + 2] = IntToBytes(readmark)
                return selectedMsg
            elif readMethod == "newest":
                selectedMsg = allMsg[writemark-1 : writemark]
                readmark = writemark
                shm_buffer[readPos + 1 + writePos * 2 : readPos + 1 + writePos * 2 + 2] = IntToBytes(readmark)
                return selectedMsg
        return []
    # else:
    #     return []
            
    def ReadToStr(self, readId, readMethod = "new") -> list[str]:
        msgs: list[bytearray] = self.Read(readId, readMethod)
        if len(msgs):
            strMsgs: list[str] = []
            for msg in msgs:
                if bytearray([0xFF, 0xFF]) in msg:
                    return []
                if len(msg) > 3 and (len(msg) - 2) == BytesToInt(msg[0:2]):
                    strMsgs.append(msg[2:].decode())
            return strMsgs
        else:
            return []

    def ShowAllData(self):
        global shm_buffer
        tempData = np.zeros(166*165*3, np.uint8)
        tempData[0:len(shm_buffer)] = np.array(shm_buffer, np.uint8, copy=True)
        tempData[len(shm_buffer):] = [128] * (len(tempData) - len(shm_buffer))
        # tempData[tempData == 255] = 0
        tempData.shape = 166, 165, 3
        cv2.imshow("msgs", cv2.resize(tempData, (0, 0), None, 4, 4))
        # cv2.imshow("msgs", tempData)
        cv2.waitKey(0)

    def TestStart(self):
        self.InitBuffer()
        while True:
            if keyboard.is_pressed('esc'):
                break
            elif keyboard.is_pressed('ctrl+shift+v'):
                self.ShowAllData()
            if self.CheckOnlineClientsCount() > 0:
                self.WriteContent("from python" + datetime.datetime.now().strftime("%H_%M_%S"), True)
                time.sleep(0.005)
                # for other in self.CheckOnlineClients():
                #     for newmsg in (self.ReadToStr(other, "all")):
                #         print(newmsg)

        
# if __name__ == '__main__':
#     UnityShm = SharedMemoryObj('UnityShareMemoryTest', "server", "UnityProject", 32+5*16*1024)#~80KB
#     UnityShm.Start()
#     del UnityShm
    