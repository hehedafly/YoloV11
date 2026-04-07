import ctypes

user32 = ctypes.windll.user32
MessageBoxW = user32.MessageBoxW
MessageBoxW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
MessageBoxW.restype = ctypes.c_int

# class PythonWinMessageBox:
# 定义常量
MB_OK = 0x0
MB_OKCANCEL = 0x1
MB_ABORTRETRYIGNORE = 0x2
MB_YESNOCANCEL = 0x3
MB_YESNO = 0x4
MB_RETRYCANCEL = 0x5
MB_ICONHAND = 0x10
MB_ICONQUESTION = 0x20
MB_ICONEXCLAMATION = 0x30
MB_ICONINFORMATION = 0x40
MB_ICONASTERISK = MB_ICONINFORMATION
MB_ICONWARNING = MB_ICONEXCLAMATION
MB_ICONERROR = MB_ICONHAND
MB_ICONSTOP = MB_ICONHAND
MB_DEFBUTTON1 = 0x0
MB_DEFBUTTON2 = 0x100
MB_DEFBUTTON3 = 0x200
MB_DEFBUTTON4 = 0x300
IDOK = 1
IDCANCEL = 2
IDABORT = 3
IDRETRY = 4
IDIGNORE = 5
IDYES = 6
IDNO = 7


def YesOrNo(text, title):
    result = MessageBoxW(None, text, title, MB_YESNO | MB_ICONQUESTION)
    if result == IDYES:
        return "YES"
    elif result == IDNO:
        return "NO"


def OkCancel(text, title):
    result = MessageBoxW(None, text, title, MB_OKCANCEL | MB_ICONQUESTION)
    if result == IDOK:
        return "OK"
    elif result == IDCANCEL:
        return "CANCEL"


def Information(text, title):
    result = MessageBoxW(None, text, title, MB_OK | MB_ICONINFORMATION)
    if result == IDOK:
        return "OK"


def Warning(text, title):
    result = MessageBoxW(None, text, title, MB_OK | MB_ICONWARNING)
    if result == IDOK:
        return "OK"


def Error(text, title):
    result = MessageBoxW(None, text, title, MB_OK | MB_ICONERROR)
    if result == IDOK:
        return "OK"