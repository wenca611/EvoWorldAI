import numpy as np
import pyautogui
import cv2

# Hledaná barva v RGB (oranžový obdélník s černým textem)
TARGET_COLOR = np.array([69, 122, 229])
TOLERANCE = 20  # Povolená odchylka

# Ořezání na všech stranách
TOP_CROP = 100  # Oříznutí 100 px nahoře
BOTTOM_CROP = 350  # Oříznutí 200 px dole
LEFT_CROP = 350  # Oříznutí 200 px zleva
RIGHT_CROP = 700  # Oříznutí 500 px zprava

# Minimální rozměry hledaného obdélníku
MIN_WIDTH = 100
MIN_HEIGHT = 40


def detect_and_restart_game(image: np.ndarray) -> bool:
    """
    Ořízne obrázek (100px nahoře, 200px dole, 200px zleva a zprava),
    hledá největší oranžový obdélník a klikne na něj pro restart hry.

    :param image: Screenshot jako numpy pole (HxWx3 nebo HxWx4)
    :return: True, pokud byl obdélník detekován a kliknuto na restart, jinak False
    """
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        return False  # Ochrana proti neplatným vstupům

    # Získání rozměrů obrázku
    h, w = image.shape[:2]

    # Ořezání obrázku
    cropped_image = image[TOP_CROP:h - BOTTOM_CROP, LEFT_CROP:w - RIGHT_CROP]

    # Pokud má obrázek 4 kanály (RGBA), převedeme na RGB
    if cropped_image.shape[-1] == 4:
        cropped_image = cropped_image[:, :, :3]

    # Vytvoření masky pro hledanou barvu
    lower_bound = np.clip(TARGET_COLOR - TOLERANCE, 0, 255)
    upper_bound = np.clip(TARGET_COLOR + TOLERANCE, 0, 255)
    mask = cv2.inRange(cropped_image, lower_bound, upper_bound)

    # Najdeme kontury (hranice oranžového obdélníku)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False  # Nenašel se žádný odpovídající obdélník

    # Vybereme největší konturu podle plochy (šířka * výška)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Přepočítáme souřadnice zpět do původního obrazu
    x += LEFT_CROP
    y += TOP_CROP  # Posun kvůli hornímu ořezu

    # Musí splnit minimální velikost
    if w >= MIN_WIDTH and h >= MIN_HEIGHT:
        center_x = x + w // 2
        center_y = y + h // 2

        # print(f"Konec hry detekován na ({center_x}, {center_y}), restartuji...")
        pyautogui.click(center_x, center_y)
        return True

    return False
