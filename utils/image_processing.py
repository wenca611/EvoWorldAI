import os
import cv2
import numpy as np

# Název složky pro šablony
TEMPLATES_FOLDER = "templates"


def create_template_paths():
    """

    :return:
    """
    template_paths = {}
    template_mapping = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '/': 'slash'
    }

    # Zajistíme existenci složky pro šablony
    if not os.path.exists(TEMPLATES_FOLDER):
        os.makedirs(TEMPLATES_FOLDER)

    for label, base_name in template_mapping.items():
        templates = []
        # Prohledáme složku pro šablony s názvem základního jména a číslovkami
        for i in range(50):
            file_name = os.path.join(TEMPLATES_FOLDER, f'{base_name}{i}.png')
            if os.path.exists(file_name):
                templates.append(file_name)
        if templates:
            template_paths[label] = templates
    return template_paths


def load_templates(template_paths):
    """

    :param template_paths:
    :return:
    """
    templates = []
    for path in template_paths:
        template = cv2.imread(path)
        if template is not None:
            template = template[:, 1:]
            templates.append(template)
    return templates


def preprocess_image(img):
    """

    :param img:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_value = int(0.007 * 255)
    mask = gray > threshold_value
    img_processed = gray.copy()
    img_processed[mask] = 255
    return img_processed.astype(np.uint8)


def ensure_grayscale(img):
    """

    :param img:
    :return:
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def match_multiple_templates(img, templates):
    """

    :param img:
    :param templates:
    :return:
    """
    img_gray = ensure_grayscale(img)
    results = []
    for template in templates:
        template_gray = ensure_grayscale(template)
        result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        results.append(result)
    return results


def is_too_close(new_pt, found_positions, min_distance=4):
    """

    :param new_pt:
    :param found_positions:
    :param min_distance:
    :return:
    """
    for _, x, _ in found_positions:
        if abs(new_pt[0] - x) < min_distance:
            return True
    return False


template_paths = create_template_paths()
templates_dict = {label: load_templates(paths) for label, paths in template_paths.items()}


def process_image(img) -> str:
    """

    :param img:
    :return:
    """
    results_dict = {label: match_multiple_templates(img, templates) for label, templates in templates_dict.items()}
    threshold = 0.90
    found_positions = []

    for label, result_set in results_dict.items():
        for result in result_set:
            locations = np.where(result >= threshold)
            for pt in zip(*locations[::-1]):
                if not is_too_close(pt, found_positions):
                    found_positions.append((label, pt[0], pt[1]))
                    pt = (pt[0] + 4, pt[1])

    found_positions_sorted = sorted(found_positions, key=lambda x: x[1])
    detected_numbers = [number for number, _, _ in found_positions_sorted]
    return "".join(detected_numbers)


def images_are_different(img1, img2, threshold=30):
    """

    :param img1:
    :param img2:
    :param threshold:
    :return:
    """
    if img1 is None or img2 is None:
        return True  # Pokud jeden z obrázků neexistuje, považujeme je za rozdílné

    if not isinstance(img1, np.ndarray) or not isinstance(img2, np.ndarray):
        print("ERROR: Jeden z obrázků není numpy ndarray!")
        return True

    diff = cv2.absdiff(img1, img2)
    return np.mean(diff) > threshold  # Průměrná rozdílnost pixelů


def calculate_fill_percentage(img):
    """

    :param img:
    :return:
    """
    # Pokud má obrázek 4 kanály (RGBA), odstraníme alfa kanál a použijeme jen RGB
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_rgb = img[:, :, :3]  # Pouze RGB kanály
    else:
        img_rgb = img

    # Převod na odstíny šedi (bez ohledu na počet kanálů)
    gray_row = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)[0, :]  # Pouze první řádek obrázku

    # Počítáme procenta zaplnění
    white_threshold = 200
    total_pixels = gray_row.shape[0]
    filled_pixels = np.sum(gray_row < white_threshold)

    fill_percentage = (filled_pixels / total_pixels) * 100
    return round(fill_percentage, 2)
