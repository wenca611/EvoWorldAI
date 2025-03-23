import ctypes
import time

# Načtení knihovny user32.dll
user32 = ctypes.WinDLL("user32", use_last_error=True)

# Konstanta pro uvolnění klávesy
KEYEVENTF_KEYUP = 0x0002

# Mapování znaků na virtuální klávesy
keyCodeMap = {
    'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45, 'f': 0x46,
    'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A, 'k': 0x4B, 'l': 0x4C,
    'm': 0x4D, 'n': 0x4E, 'o': 0x4F, 'p': 0x50, 'q': 0x51, 'r': 0x52,
    's': 0x53, 't': 0x54, 'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58,
    'y': 0x59, 'z': 0x5A, 'space': 0x20
}

def press_key(hexKeyCode):
    """

    :param hexKeyCode:
    :return:
    """
    """Stisknutí klávesy"""
    user32.keybd_event(hexKeyCode, 0, 0, 0)

def release_key(hexKeyCode):
    """

    :param hexKeyCode:
    :return:
    """
    """Uvolnění klávesy"""
    user32.keybd_event(hexKeyCode, 0, KEYEVENTF_KEYUP, 0)

def send_key(key):
    """

    :param key:
    :return:
    """
    """Stiskne a uvolní klávesu podle zadaného znaku."""
    hex_code = keyCodeMap.get(key)
    if hex_code:
        press_key(hex_code)
        time.sleep(0.05)  # Krátká pauza pro simulaci stisku
        release_key(hex_code)

def execute_action(action):
    """

    :param action:
    :return:
    """
    """Spustí akci podle mapování"""
    if action == "wait":
        time.sleep(0.1)  # 100ms pauza
    elif isinstance(action, list):  # Kombinace kláves
        for key in action:
            send_key(key)
    else:
        send_key(action)

# Mapování akcí
ACTION_MAP = {
    0: "a",
    1: "w",
    2: "d",
    3: "space",
    4: ["a", "w"],
    5: ["d", "w"],
    6: "wait"
}