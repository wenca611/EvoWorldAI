#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python 3.10+

"""
Projekt: EvoWorld Bot
Autor: Ing. Václav Pastušek
Datum: 2025-03-22
Verze: 1.0.0
Licence: MIT
Popis: Tento skript implementuje DDQN (Double Deep Q-Network) agenta pro hru EvoWorld.io.
Agent se učí hrát pomocí posilovaného učení (Reinforcement Learning, RL).
Využívá neuronovou síť k odhadu hodnot akčních strategií a kombinuje evaluační a target síť,
což řeší problém přeceňování Q-hodnot, typický pro standardní DQN.
Agent vnímá hru jako obrazové vstupy, analyzuje je a provádí akce na základě predikcí své sítě.
Trénuje se s využitím Replay Bufferu a Target Network Update, aby se zlepšila stabilita učení.
Cílem je dosáhnout optimální strategie přežití a evoluce v herním prostředí.
"""

# 🛠️ Standardní knihovny Pythonu
import atexit  # Spustí funkci při ukončení programu (např. uklízení prostředků)
import json  # Práce s JSON soubory (ukládání a načítání dat)
import os  # Přístup k funkcím operačního systému (soubory, proměnné prostředí)
import threading  # Práce s vlákny pro souběžné provádění úloh
from _typeshed import SupportsWrite  # Typový hint pro objekty, které podporují zápis řetězců (str)
from time import sleep  # Pauza v běhu programu (časový delay)
import random  # Generování náhodných čísel a výběr prvků ze seznamů
from collections import deque  # Efektivní struktura FIFO/LIFO (fronta/zásobník)
from typing import Deque, Optional, Dict

# 🔢 Knihovny pro vědecké výpočty a strojové učení
import numpy as np  # Numerické výpočty s maticemi a vektory
import tensorflow as tf  # Strojové učení a neuronové sítě

# ⌨️ Interakce s klávesnicí a obrazovkou
import keyboard  # Zachytávání vstupů z klávesnice a simulace stisků kláves
import mss  # Rychlé snímání obrazovky (screenshoty)

# 🌐 Automatizace webového prohlížeče (Selenium)
from selenium import webdriver  # Hlavní knihovna pro ovládání prohlížeče
from selenium.webdriver import ActionChains, Keys  # Simulace pohybu myši a kláves
from selenium.webdriver.chrome.options import Options  # Konfigurace Chromu
from selenium.webdriver.support.wait import WebDriverWait  # Čekání na načtení prvků
from selenium.webdriver.support import expected_conditions as EC  # Ověřování viditelnosti prvků
from selenium.webdriver.common.by import By  # Hledání prvků na stránce (např. By.ID, By.XPATH)

# 📂 Vlastní pomocné funkce (import z vlastního souboru `utils.py`)
from utils import *  # Import všech funkcí z utils.py (pravděpodobně vlastní pomocné metody)

# Vypnutí oneDNN pro snížení varování při inicializaci TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"]: str = "0"

# Cesty k modelům (pro evaluaci a trénování)
EVAL_MODEL_PATH: str = "eval_model.keras"
TARGET_MODEL_PATH: str = "target_model.keras"
EPSILON_PATH: str = 'epsilon.json'

# Možné akce pro agenta (klávesy, které může stisknout)
ACTION_MAP: dict[int, str | list[str]] = {
    0: "a",  # Pohyb doleva
    1: "w",  # Pohyb nahoru
    2: "d",  # Pohyb doprava
    3: "space",  # Skok
    4: ["a", "w"],  # Pohyb doleva + nahoru
    5: ["d", "w"],  # Pohyb doprava + nahoru
    6: "wait"  # Čekání (100ms neprovádí akci)
}

# Počet dostupných akcí
ACTION_LEN: int = len(ACTION_MAP)

# Řízení trénování
is_training: bool = False  # Indikuje, zda právě probíhá trénování
train_lock: threading.Lock = threading.Lock()  # Zámek pro synchronizaci trénování
can_train: bool = False  # Ovládání spuštění trénování

# Hyperparametry pro trénování agenta
GAMMA: float = 0.99  # Diskontní faktor pro budoucí odměny
ALPHA: float = 0.001  # Learning rate (rychlost učení)
EPSILON: float = 1.  # Počáteční hodnota epsilonu pro epsilon-greedy strategii
EPSILON_DECAY: float = 0.99995  # Rychlost snižování epsilonu
EPSILON_MIN: float = 0.0001  # Minimální hodnota epsilonu
BATCH_SIZE: int = 32  # Velikost batch při trénování
MEMORY_SIZE: int = 10_000  # Maximální velikost replay bufferu
FRAME_STACK: int = 4  # Počet posledních snímků, které agent vidí současně pro zachycení pohybu v čase
UPDATE_TARGET_FREQUENCY: int = 10  # Počet iterací před aktualizací target modelu

# Sledování EXP pro výpočet delta_exp (změna zkušeností)
prev_exp_value: int = 0

# Sledování předchozích hodnot kyslíku a vody
prev_oxygen: float = 100.
prev_water: float = 100.
oxygen_decrease_counter: float = 1.  # Počet snížení kyslíku
water_decrease_counter: float = 1.  # Počet snížení vody


# DDQN Model
# noinspection PyUnresolvedReferences
def build_model() -> tf.keras.Model:
    """
    Vytváří hluboký neuronový model pro Double Deep Q-Network (DDQN).

    Tento model používá konvoluční vrstvy pro extrakci vizuálních rysů z obrazových dat a následně je
    zpracovává pomocí plně propojených vrstev pro predikci hodnot Q, které odpovídají různým akcím.

    Model obsahuje:
    - 2 konvoluční vrstvy pro extrakci rysů z obrazových dat.
    - 1 plně propojenou vrstvu pro zpracování extrahovaných rysů.
    - Výstupní vrstvu s počtem neuronů odpovídajícím počtu dostupných akcí.

    :return: Kompilovaný Keras model připravený k trénování.
    """
    model: tf.keras.Sequential = tf.keras.Sequential([
        # 1. konvoluční vrstva
        # Conv2D - provádí konvoluci na obrazových datech a extrahuje rysy.
        # Má 32 filtrů (nebo "kernelů") o velikosti 3x3 pixelů.
        # Aktivace 'relu' zavádí nelinearitu a zlepšuje modelovu schopnost učit se složitější vzory.
        # input_shape určuje tvar vstupních dat (12x120xFRAME_STACK), což odpovídá čtyřem posledním obrazům,
        # které obsahují vizuální informace z několika po sobě jdoucích snímků.
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(12, 120, FRAME_STACK)),

        # 2. konvoluční vrstva
        # Conv2D s 64 filtry, velikost 3x3, aktivace 'relu'.
        # Tato vrstva je určena pro další zpracování rysů a pomáhá modelu zachytit složitější vzory.
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

        # Zploštění (Flatten)
        # Tato vrstva převádí 2D výstupy z předchozích vrstev na 1D vektor,
        # což umožňuje použití těchto výstupů v plně propojených vrstvách.
        tf.keras.layers.Flatten(),

        # Plně propojená vrstva (Dense) s 128 neurony
        # Každý neuron je propojen se všemi neurony předchozí vrstvy.
        # Aktivace 'relu' je použita pro nelineární transformaci a umožňuje složitější rozhodování.
        tf.keras.layers.Dense(128, activation='relu'),

        # Výstupní vrstva
        # Tato vrstva má tolik neuronů, kolik je dostupných akcí.
        # Aktivace 'linear' znamená, že výstupní hodnoty jsou libovolné reálné číslo (vhodné pro hodnoty Q v DQN).
        tf.keras.layers.Dense(ACTION_LEN, activation='linear')
    ])

    # Kompilace modelu
    # Model je kompilován s optimalizátorem Adam, který je efektivní pro trénování modelů na velkých datech.
    # Adam automaticky upravuje rychlost učení pro každý parametr.
    # Ztrátová funkce 'mse' (Mean Squared Error) se používá k minimalizaci rozdílu mezi předpovězenými
    # a skutečnými hodnotami Q.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ALPHA), loss='mse')

    return model


# Funkce pro načítání modelů a epsilonu
# noinspection PyUnresolvedReferences
def load_models() -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Načítá modely pro evaluaci a target model, pokud již existují, nebo vytvoří nové modely.
    Také načítá hodnotu epsilonu z JSON souboru.

    :return: Tuple obsahující eval model a target model.
    """
    global EPSILON

    # Načítání eval modelu
    if os.path.exists(EVAL_MODEL_PATH):
        print("Načítám eval model...")
        local_eval_model: tf.keras.Model = tf.keras.models.load_model(EVAL_MODEL_PATH)
    else:
        print("Eval model nenalezen, vytvářím nový...")
        local_eval_model: tf.keras.Model = build_model()

    # Načítání target modelu
    if os.path.exists(TARGET_MODEL_PATH):
        print("Načítám target model...")
        local_target_model: tf.keras.Model = tf.keras.models.load_model(TARGET_MODEL_PATH)
    else:
        print("Target model nenalezen, vytvářím nový...")
        local_target_model: tf.keras.Model = build_model()
        local_target_model.set_weights(eval_model.get_weights())  # Synchronizace modelů

    # Načítání hodnoty epsilonu
    if os.path.exists(EPSILON_PATH):
        with open(EPSILON_PATH) as f:
            EPSILON = json.load(f)  # Načtení hodnoty epsilonu z JSON souboru
        print(f"Načtená hodnota EPSILON: {EPSILON}")
    else:
        print("Soubor s EPSILON nenalezen, používám výchozí hodnotu.")

    return local_eval_model, local_target_model


# Načtení modelů při startu
eval_model, target_model = load_models()


# Funkce pro ukládání modelů a epsilonu
def save_models() -> None:
    """
    Ukládá aktuální modely a hodnotu epsilonu do souborů.
    Modely se ukládají do jejich příslušných cest (EVAL_MODEL_PATH, TARGET_MODEL_PATH),
    a hodnota epsilonu se zapisuje do JSON souboru.

    :return: None
    """
    global eval_model, target_model

    # Uložení modelů na disk
    eval_model.save(EVAL_MODEL_PATH)
    target_model.save(TARGET_MODEL_PATH)

    # Uložení hodnoty epsilonu do JSON souboru
    f: SupportsWrite[str]
    with open(EPSILON_PATH, 'w') as f:
        json.dump(EPSILON, f)

    print("Modely a EPSILON uloženy.")


# Automatické ukládání modelů a epsilonu při ukončení programu
atexit.register(save_models)

# Paměť pro replay buffer (FIFO fronta s omezenou velikostí)
memory: Deque = deque(maxlen=MEMORY_SIZE)

# Řídící proměnná pro ukončení hlavní smyčky
stop_loop: bool = False

# Proměnné pro uchovávání poslední přečtené hodnoty EXP a screenshotu
last_exp_img: Optional[np.ndarray] = None
last_exp_value: Optional[str] = None

# Konfigurace oblasti monitoru pro snímání obrazovky
monitor: Dict[str, int] = {"top": 0, "left": 0, "width": 1920, "height": 1080}

# FIFO zásobník pro uchovávání posledních snímků (frame stack)
frame_stack: Deque[np.ndarray] = deque(maxlen=FRAME_STACK)


def listen_for_key() -> None:
    """
    Funkce čeká na stisk klávesy 'x' a po jejím stisknutí ukončí hlavní smyčku.

    :return: None
    """
    global stop_loop

    print("Stiskni 'x' pro ukončení smyčky.")
    keyboard.wait('x')  # Čeká na stisk klávesy 'x'
    print("Klávesa 'x' stisknuta, ukončuji...")
    stop_loop: bool = True  # Nastaví globální proměnnou stop_loop na True, čímž ukončí smyčku


def select_action(state) -> int:
    """
    Vybere akci na základě aktuálního stavu a hodnoty epsilonu.
    Pokud je náhodná hodnota menší než epsilon, akce je vybrána náhodně podle váhy.
    Pokud je náhodná hodnota větší nebo rovná epsilonu, akce je vybrána podle maximální hodnoty Q funkce.

    :param state: Aktuální stav (obvykle ve formě NumPy pole)
    :return: číselná hodnota akce, která bude vykonána
    """
    # Pokud je náhodné číslo menší než epsilon, vybereme náhodnou akci s váhami
    if np.random.rand() < EPSILON:
        weighted_actions: list = []  # Seznam pro akce s váhami
        for action in range(ACTION_LEN):
            # Akce obsahující 'w' dostanou nižší váhu
            if 'w' not in ACTION_MAP[action] or 'wait' in ACTION_MAP[action]:
                # Pokud akce neobsahuje 'w' nebo obsahuje 'wait', přidáme akci čtyřikrát (vyšší váha)
                weighted_actions.extend([action] * 4)
            else:
                # Jinak přidáme akci s váhou 1
                weighted_actions.append(action)

        # Náhodný výběr akce na základě váhy
        return random.choice(weighted_actions)

    # Pokud je náhodné číslo větší než epsilon, vybereme akci s nejvyšší Q hodnotou
    q_values = eval_model(state)  # Získáme Q hodnoty pro daný stav
    return np.argmax(q_values[0])  # Vrátíme akci s nejvyšší Q hodnotou


def store_experience(state, action, reward: float, next_state, done: bool) -> None:
    """
    Uloží zkušenost (stav, akci, odměnu, následující stav, hotovo) do replay bufferu.

    :param state: Aktuální stav prostředí (může být libovolného typu, např. obraz nebo stav).
    :param action: Akce, kterou agent vykonal (může být libovolného typu, např. číslo nebo řetězec).
    :param reward: Odměna za vykonání akce (float).
    :param next_state: Následující stav po vykonání akce (může být libovolného typu, např. obraz nebo stav).
    :param done: Indikátor, zda byla epizoda dokončena (True/False).
    :return: None
    """
    memory.append((state, action, reward, next_state, done))


def compute_reward(exp_value: float, oxygen_percentage: float, water_percentage: float) -> float:
    """
    Funkce pro výpočet reward na základě změny EXP, kyslíku a vody.

    :param exp_value: Aktuální hodnota EXP
    :param oxygen_percentage: Aktuální procento kyslíku
    :param water_percentage: Aktuální procento vody
    :return: Vypočtená odměna (reward) na základě změn v hodnotách
    """
    # Globální proměnné pro uchování předchozích hodnot
    global prev_exp_value, prev_oxygen, prev_water, oxygen_decrease_counter, water_decrease_counter

    # Výpočet změny EXP (delta_exp)
    # Pokud je předchozí hodnota EXP None, znamená to, že se jedná o první iteraci, takže změna bude 0
    delta_exp: float = 0 if prev_exp_value is None else exp_value - prev_exp_value
    prev_exp_value: float = exp_value  # Aktualizace pro další iteraci

    # Výpočet změny kyslíku (delta_oxygen) a vody (delta_water)
    # Stejně jako u delta_exp, pokud je předchozí hodnota None, změna bude 0
    delta_oxygen: float = 0 if prev_oxygen is None else oxygen_percentage - prev_oxygen
    delta_water: float = 0 if prev_water is None else water_percentage - prev_water

    # Penalizace za opakovaný úbytek kyslíku
    if delta_oxygen < 0:
        oxygen_decrease_counter += 0.1  # Zvyšujeme penalizaci při opakovaném úbytku kyslíku
    else:
        oxygen_decrease_counter = 0.1  # Reset penalizace při nárůstu kyslíku

    # Penalizace za opakovaný úbytek vody
    if delta_water < 0:
        water_decrease_counter += 0.1  # Zvyšujeme penalizaci při opakovaném úbytku vody
    else:
        water_decrease_counter = 0.1  # Reset penalizace při nárůstu vody

    # Aktualizace předchozích hodnot pro kyslík a vodu
    prev_oxygen: float = oxygen_percentage
    prev_water: float = water_percentage

    # Úprava penalizace podle počtu úbytků kyslíku a vody
    oxygen_penalty: float = delta_oxygen * oxygen_decrease_counter
    water_penalty: float = delta_water * water_decrease_counter

    # Výsledná reward funkce:
    # Používáme logaritmickou funkci pro EXP změny a přidáváme penalizace za úbytky kyslíku a vody
    reward: float = np.log(max(delta_exp, 1)) / 10.0 + oxygen_penalty + water_penalty

    return reward


def train_ddqn() -> None:
    """
    Funkce pro trénování Double Deep Q-Network (DDQN) modelu. Trénuje model na základě vzorků z replay bufferu.

    Využívá eval_model pro výpočet hodnot Q a target_model pro generování hodnoty Q pro target.
    Také pravidelně aktualizuje target_model váhy na základě eval_modelu.

    :return: None
    """
    global can_train, is_training

    train_iterations: int = 0  # Počet trénovacích iterací
    while not stop_loop:
        if can_train and not is_training:
            with train_lock:
                if is_training:
                    sleep(0.001)
                    continue
                is_training: bool = True  # Zahájení trénování

            # Pokud je v paměti málo vzorků, přeskočíme trénování
            if len(memory) < BATCH_SIZE:
                with train_lock:
                    is_training: bool = False  # Zastavení trénování
                    continue

            # Vytvoření dávky (batch) pro trénování
            batch: tuple[any, any, any, any, any] = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Předzpracování stavu pro trénování
            states = np.array(states).reshape(-1, 12, 120, FRAME_STACK)
            next_states = np.array(next_states).astype("float32") / 255.0
            next_states = np.squeeze(next_states, axis=1)

            # Predikce Q-hodnot pro stavy a následující stavy
            target_qs = eval_model.predict(states, verbose=0)
            next_qs = eval_model.predict(next_states, verbose=0)
            target_qs_next = target_model.predict(next_states, verbose=0)

            # Výpočet cílových Q-hodnot pro trénování
            for i in range(BATCH_SIZE):
                if dones[i]:
                    target_qs[i][actions[i]] = rewards[i]  # Pokud je done, nastavíme odměnu na hodnotu
                else:
                    best_action: int = np.argmax(next_qs[i])  # Nejlepší akce z predikce pro následující stav
                    target_qs[i][actions[i]] = rewards[i] + GAMMA * target_qs_next[i][best_action]  # Q-hodnota pro akci

            # Trénování eval_modelu
            eval_model.fit(states, target_qs, epochs=1, verbose=0)
            # print("Eval model trained.")

            train_iterations += 1
            if train_iterations % UPDATE_TARGET_FREQUENCY == 0:
                target_model.set_weights(eval_model.get_weights())  # Aktualizace target_modelu na základě eval_modelu
                # print("Target model updated.")

            with train_lock:
                is_training: bool = False  # Zastavení trénování
            can_train: bool = False  # Nastavení flagu, že trénování není možné
        else:
            sleep(0.001)  # Krátká pauza, pokud nemůžeme trénovat


def is_valid_exp_value(exp_text: str) -> Optional[float]:
    """
    Funkce kontroluje, zda text reprezentující zkušenostní hodnotu (EXP) obsahuje platnou číselnou hodnotu.
    Pokud ano, vrátí ji jako číslo typu float, jinak vrátí None.

    :param exp_text: Řetězec obsahující hodnotu EXP ve formátu "číslo/něco"
    :return: Pokud je hodnota před '/' platná, vrátí ji jako float. Jinak vrátí None.
    """
    # Kontrola, zda exp_text obsahuje '/'
    if "/" in exp_text:
        parts = exp_text.split("/")  # Rozdělíme text na části podle '/'
        try:
            return float(parts[0])  # Pokusíme se převést první část na float
        except ValueError:
            return None  # Pokud převod selže, vrátíme None
    return None  # Pokud '/' není přítomno, vrátíme None


# Funkce pro nastavení flagu pro trénování
def request_training() -> None:
    """
    Funkce nastaví flag `can_train` na hodnotu True, což signalizuje, že hlavní smyčka
    požaduje spuštění trénování modelu na jiném vlákně. Tento flag je zkontrolován
    v jiném vlákně, které začne trénovat, aniž by blokovalo hlavní smyčku.

    :return: None
    """
    global can_train  # Označení proměnné can_train jako globální, aby bylo možné ji měnit
    can_train: bool = True  # Nastavení flagu, že hlavní smyčka požaduje zahájení trénování na jiném vlákně


def log_performance(iteration: int, reward: float, action: int) -> None:
    """
    Funkce pro logování výkonu během trénování.

    :param iteration: Počet iterací (nebo kroků) během trénování.
    :param reward: Hodnota odměny získaná za aktuální akci.
    :param action: Číslo, které reprezentuje konkrétní akci provedenou agentem.
    :return: None (funkce nevrací žádnou hodnotu, pouze zapisuje do souboru)
    """
    # Otevření souboru "training_log.txt" v režimu přidávání (append mode)
    with open("training_log.txt", "a") as log_file:
        # Zapsání informací o iteraci, odměně a akci do souboru
        log_file.write(f"{iteration}, Reward: {reward:.3f}, Action: {action}\n")


def game_loop() -> None:
    """
    Hlavní smyčka pro interakci se hrou, získávání obrazů a rozhodování o akcích.

    :return: None
    """
    global last_exp_value, last_exp_img, stop_loop, EPSILON, is_training, prev_water, prev_oxygen

    # Nastavení Chrome pro maximální okno
    chrome_options: Options = Options()
    chrome_options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://evoworld.io")  # Otevření webu
    actions = ActionChains(driver)

    # čekání a objevení tlačítka, 3xTAB a enter
    sleep(8)
    for _ in range(3):
        actions.send_keys(Keys.TAB)
    actions.send_keys(Keys.ENTER)
    actions.perform()

    try:
        # Čekání, až se skryje loading bar
        WebDriverWait(driver, 300).until(
            EC.invisibility_of_element_located((By.CLASS_NAME, "loadingBar"))
        )

        # Čekání na tlačítko pro začátek hry a kliknutí na něj
        start_game_button = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '#uiContainer .main .center .gameStartBox .btnStartGame'))
        )
        start_game_button.click()
        print("Kliknuto na start!")
    except Exception as e:
        print(f"Chyba při startu: {e}")

    try:
        # Čekání na tlačítko pro bonus zkušeností a kliknutí na něj
        exp_bonus_button = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, '#uiContainer .popups .popup#join-popup .select-exp-bonus .grey-button'))
        )
        exp_bonus_button.click()
        print("Kliknuto na bonus zkušeností!")
    except Exception as e:
        print(f"Chyba při kliknutí na bonus zkušeností: {e}")

    # Čekání na načtení elementu status (označení, že hra běží)
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, 'status')))
    sleep(0.2)

    loop: int = 0  # Počítadlo smyčky
    with mss.mss() as sct:  # Použití mss pro snímky obrazovky
        while not stop_loop:  # Hlavní herní smyčka
            loop += 1
            screenshot = sct.grab(monitor)  # Snímek obrazovky
            image = np.array(screenshot)  # Převod na NumPy array
            exp_crop = image[149:161, 900:1020]  # Oříznutí oblasti pro EXP
            processed_image = preprocess_image(exp_crop)  # Zpracování obrázku

            # Pokud je obrázek nový nebo se změnil, zpracuj jej
            if last_exp_img is None or images_are_different(last_exp_img, processed_image):
                last_exp_img = processed_image
                last_exp_value = process_image(processed_image)

            # Oříznutí pro hladiny kyslíku a vody
            oxygen_crop = image[166:172, 578:1253]
            oxygen_percentage: float = calculate_fill_percentage(oxygen_crop)
            water_crop = image[185:190, 578:1253]
            water_percentage: float = calculate_fill_percentage(water_crop)

            # Detekce restartu hry a reakce na to
            if detect_and_restart_game(image):
                sleep(2)
                prev_water: float = 100.
                prev_oxygen: float = 100.
                reward: float = -1.  # Penalizace za smrt
                store_experience(state, action, reward, next_state, False)  # Uložení zkušenosti se smrtí
                continue

            # Přidání nového zpracovaného snímku do zásobníku rámců
            frame_stack.append(processed_image)
            if len(frame_stack) < FRAME_STACK:  # Pokud není dostatek rámců, pokračuj
                continue

            state = np.stack(frame_stack, axis=-1)  # Sestavení stavu
            state = np.expand_dims(state, axis=0)  # Rozšíření dimenzí

            # Výběr akce na základě stavu
            action: int = select_action(state)
            keys: str | list[str] = ACTION_MAP[action]  # Mapování akce na klávesy
            execute_action(keys)  # Spuštění akce (stisk klávesy)

            # Aktualizace stavu a odměny
            next_state = np.stack(frame_stack, axis=-1)
            next_state = np.expand_dims(next_state, axis=0)
            reward = compute_reward(
                is_valid_exp_value(last_exp_value) or 0,
                oxygen_percentage,
                water_percentage
            )
            store_experience(state, action, reward, next_state, False)

            # Kontrola, zda je potřeba trénovat
            if loop % 20 == 0 and not is_training:
                request_training()

            # Dekrementování epsilonu pro exploraci/exploitaci
            if EPSILON > EPSILON_MIN:
                EPSILON *= EPSILON_DECAY

            # log_performance(loop, reward, action)


if __name__ == "__main__":
    # Spuštění herní smyčky v samostatném vláknu a trénování DDQN
    threading.Thread(target=listen_for_key).start()
    threading.Thread(target=train_ddqn, daemon=True).start()
    game_loop()


