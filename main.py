import os
import subprocess
import time
import tkinter as tk
from tkinter import messagebox
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
import pyautogui
import pyperclip
import sounddevice as sd
import speech_recognition as sr
from mediapipe.tasks.python import vision

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

MODEL_PATH = "hand_landmarker.task"
SCREEN_W, SCREEN_H = pyautogui.size()

THUMB_TIP = 4
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

CLICK_DISTANCE = 0.05  # normalized thumb-index distance that counts as a pinch/click
SMOOTHING = 0.4  # 0 = frozen cursor, 1 = jumps straight to the fingertip each frame
SCROLL_SENSITIVITY = 800  # normalized y movement -> scroll "clicks" per frame
VOLUME_STEP_DISTANCE = 0.04  # normalized y movement needed to trigger one volume step

MODES = ["cursor", "scroll", "volume"]  # left fist cycles through these, in order
# MediaPipe's handedness is from the subject's own perspective. If your left hand
# ends up controlling the cursor instead of switching modes, flip this to True.
SWAP_HANDEDNESS = False

DOUBLE_TAP_WINDOW = 0.4  # seconds between two left-fist closes that counts as a double tap
VOICE_SAMPLE_RATE = 16000
VOICE_RECORD_SECONDS = 4  # how long to record after a double tap
VOICE_LANGUAGE = "pt-BR"
NOTEPAD_OPEN_DELAY = 0.6  # seconds to wait for Notepad's window before pasting into it

recognizer = sr.Recognizer()

tk_root = tk.Tk()
tk_root.withdraw()  # only used to host the confirmation popup, never shown itself


def is_finger_up(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def is_fist(landmarks):
    fingers = [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP), (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]
    return not any(is_finger_up(landmarks, tip_idx, pip_idx) for tip_idx, pip_idx in fingers)


def dictate_to_notepad(frame):
    """Records mic audio, confirms the transcript with the user, then types it into Notepad."""
    cv2.putText(frame, "Listening...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('control hands', frame)
    cv2.waitKey(1)

    recording = sd.rec(int(VOICE_RECORD_SECONDS * VOICE_SAMPLE_RATE),
                        samplerate=VOICE_SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()

    audio = sr.AudioData(recording.tobytes(), VOICE_SAMPLE_RATE, 2)
    try:
        text = recognizer.recognize_google(audio, language=VOICE_LANGUAGE)
    except (sr.UnknownValueError, sr.RequestError):
        return  # couldn't understand the audio, or no network

    if not messagebox.askyesno("Confirmar ditado", f'Digitar isto no Bloco de Notas?\n\n"{text}"'):
        return

    subprocess.Popen(["notepad.exe"])
    time.sleep(NOTEPAD_OPEN_DELAY)
    pyperclip.copy(text)
    pyautogui.hotkey('ctrl', 'v')

options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)
landmarker = vision.HandLandmarker.create_from_options(options)

vid = cv2.VideoCapture(0)
start_time = time.monotonic()
cursor_x, cursor_y = pyautogui.position()
is_clicking = False
mode_index = 0
was_fist = False
gesture_anchor_y = None
last_close_time = None
pending_mode_cycle = False

while True:
    ret, frame = vid.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.monotonic() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    left_hand = None
    right_hand = None
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        vision.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )

        category = handedness[0].category_name
        if SWAP_HANDEDNESS:
            category = "Right" if category == "Left" else "Left"
        if category == "Left":
            left_hand = hand_landmarks
        else:
            right_hand = hand_landmarks

    # left hand: one fist close cycles the active mode, two in a row start voice dictation
    now = time.monotonic()
    fist = is_fist(left_hand) if left_hand is not None else False
    if fist and not was_fist:
        if last_close_time is not None and (now - last_close_time) <= DOUBLE_TAP_WINDOW:
            pending_mode_cycle = False
            last_close_time = None
            dictate_to_notepad(frame)
        else:
            last_close_time = now
            pending_mode_cycle = True
        gesture_anchor_y = None  # avoid a jump when the new mode starts reading
    was_fist = fist

    if pending_mode_cycle and last_close_time is not None and now - last_close_time > DOUBLE_TAP_WINDOW:
        mode_index = (mode_index + 1) % len(MODES)
        pending_mode_cycle = False

    mode = MODES[mode_index]

    # right hand: performs the active mode's action
    if right_hand is None:
        gesture_anchor_y = None
        is_clicking = False
    elif mode == "cursor":
        index_tip = right_hand[INDEX_TIP]
        thumb_tip = right_hand[THUMB_TIP]

        target_x = index_tip.x * SCREEN_W
        target_y = index_tip.y * SCREEN_H
        cursor_x += (target_x - cursor_x) * SMOOTHING
        cursor_y += (target_y - cursor_y) * SMOOTHING
        pyautogui.moveTo(cursor_x, cursor_y)

        pinch_distance = ((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2) ** 0.5
        if pinch_distance < CLICK_DISTANCE and not is_clicking:
            pyautogui.click()
            is_clicking = True
        elif pinch_distance >= CLICK_DISTANCE:
            is_clicking = False

    elif mode == "scroll":
        current_y = right_hand[MIDDLE_TIP].y
        if gesture_anchor_y is not None:
            clicks = int((gesture_anchor_y - current_y) * SCROLL_SENSITIVITY)
            if clicks:
                pyautogui.scroll(clicks)
        gesture_anchor_y = current_y

    elif mode == "volume":
        current_y = right_hand[MIDDLE_TIP].y
        if gesture_anchor_y is None:
            gesture_anchor_y = current_y
        elif current_y < gesture_anchor_y - VOLUME_STEP_DISTANCE:
            pyautogui.press('volumeup')
            gesture_anchor_y = current_y
        elif current_y > gesture_anchor_y + VOLUME_STEP_DISTANCE:
            pyautogui.press('volumedown')
            gesture_anchor_y = current_y

    label = f"mode: {mode}" + (" (switching...)" if fist else "")
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('control hands', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
landmarker.close()





