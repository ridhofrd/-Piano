import numpy as np
import cv2
import time
import argparse
import ast
import streamlit as st # type: ignore
import pygame
import mediapipe as mp
import logging
import os

# Suppress MediaPipe warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Constants
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
SHAPE = (800, 600, 3)

# Hand Detection
class HandsDetection:
    def __init__(self, model_path):
        mp_hands = mp.solutions.hands
        self.detector = mp_hands.Hands(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            max_num_hands=2
        )
        self.x = []
        self.y = []
        self.z = []

    def detect(self, image):
        self.x = []
        self.y = []
        self.z = []
        results = self.detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                x, y, z = [], [], []
                for landmark in hand_landmark.landmark:
                    x.append(landmark.x)
                    y.append(landmark.y)
                    z.append(landmark.z)
                height, width, _ = image.shape
                x = [int(lm * width) for lm in x]
                y = [int(lm * height) for lm in y]
                z = [int(lm * width) for lm in z]
                self.x.append(x)
                self.y.append(y)
                self.z.append(z)

# Piano Keyboard
class Piano:
    def __init__(self, pts, num_octaves, height_and_width):
        self.pts = pts
        self.num_octaves = num_octaves
        self.white = []
        self.black = []
        self.height_of_black = height_and_width[0]
        self.width_of_black = height_and_width[1]
        self.get_keyboard_keys()

    def section_formula(self, x1, y1, x2, y2, m, n):
        if m + n == 0:
            return int(x1), int(y1)
        x = (m * x2 + n * x1) / (m + n)
        y = (m * y2 + n * y1) / (m + n)
        return int(x), int(y)

    def add_octaves(self, pts, list_, n):
        [[[x0, y0]], [[x1, y1]], [[x2, y2]], [[x3, y3]]] = pts
        for i in range(n):
            temp = np.zeros((4, 1, 2), dtype=np.int32)
            x, y = self.section_formula(x0, y0, x1, y1, i, n - i)
            temp[0][0] = [x, y]
            x, y = self.section_formula(x0, y0, x1, y1, i + 1, n - i - 1)
            temp[1][0] = [x, y]
            x, y = self.section_formula(x3, y3, x2, y2, i + 1, n - i - 1)
            temp[2][0] = [x, y]
            x, y = self.section_formula(x3, y3, x2, y2, i, n - i)
            temp[3][0] = [x, y]
            list_.append(temp)

    def add_white_keys(self, pts):
        [[[x0, y0]], [[x1, y1]], [[x2, y2]], [[x3, y3]]] = pts
        for i in range(7):
            temp = np.zeros((4, 1, 2), dtype=np.int32)
            x, y = self.section_formula(x0, y0, x1, y1, i, 7 - i)
            temp[0][0] = [x, y]
            x, y = self.section_formula(x0, y0, x1, y1, i + 1, 7 - i - 1)
            temp[1][0] = [x, y]
            x, y = self.section_formula(x3, y3, x2, y2, i + 1, 7 - i - 1)
            temp[2][0] = [x, y]
            x, y = self.section_formula(x3, y3, x2, y2, i, 7 - i)
            temp[3][0] = [x, y]
            self.white.append(temp)

    def add_black_keys(self, pts):
        [[[x0, y0]], [[x1, y1]], [[x2, y2]], [[x3, y3]]] = pts
        last_x_, last_y_ = (x3, y3)
        last_x_up, last_y_up = (x0, y0)
        for i in range(1, 7):
            x_, y_ = self.section_formula(x3, y3, x2, y2, i, 7 - i)
            x_up, y_up = self.section_formula(x0, y0, x1, y1, i, 7 - i)
            if i == 3:
                last_x_up, last_y_up = x_up, y_up
                last_x_, last_y_ = x_, y_
                continue
            temp = np.zeros((4, 1, 2), dtype=np.int32)
            xy_coords = np.zeros((4, 1, 2), dtype=np.int32)
            n, d = self.width_of_black
            x, y = self.section_formula(last_x_, last_y_, x_, y_, 2 * d - n, n)
            temp[3][0] = [x, y]
            x, y = self.section_formula(last_x_up, last_y_up, x_up, y_up, 2 * d - n, n)
            xy_coords[0][0] = [x, y]
            x, y = self.section_formula(last_x_, last_y_, x_, y_, 2 * d + n, -n)
            temp[2][0] = [x, y]
            x, y = self.section_formula(last_x_up, last_y_up, x_up, y_up, 2 * d + n, -n)
            xy_coords[1][0] = [x, y]
            n_, d_ = self.height_of_black
            x, y = self.section_formula(temp[2][0][0], temp[2][0][1], xy_coords[1][0][0], xy_coords[1][0][1], n_, d_ - n_)
            temp[1][0] = [x, y]
            x, y = self.section_formula(temp[3][0][0], temp[3][0][1], xy_coords[0][0][0], xy_coords[0][0][1], n_, d_ - n_)
            temp[0][0] = [x, y]
            last_x_up, last_y_up = x_up, y_up
            last_x_, last_y_ = x_, y_
            self.black.append(temp)

    def add_minor_keys(self, pts):
        [[[x0, y0]], [[x1, y1]], [[x2, y2]], [[x3, y3]]] = pts
        for i in range(2):
            temp = np.zeros((4, 1, 2), dtype=np.int32)
            x, y = self.section_formula(x0, y0, x1, y1, i, 2 - i)
            temp[0][0] = [x, y]
            x, y = self.section_formula(x0, y0, x1, y1, i + 1, 2 - i - 1)
            temp[1][0] = [x, y]
            x, y = self.section_formula(x3, y3, x2, y2, i + 1, 2 - i - 1)
            temp[2][0] = [x, y]
            x, y = self.section_formula(x3, y3, x2, y2, i, 2 - i)
            temp[3][0] = [x, y]
            self.white.append(temp)
        temp = np.zeros((4, 1, 2), dtype=np.int32)
        xy_coords = np.zeros((4, 1, 2), dtype=np.int32)
        x_, y_ = self.section_formula(x3, y3, x2, y2, 1, 1)
        x_up, y_up = self.section_formula(x0, y0, x1, y1, 1, 1)
        n, d = self.width_of_black
        x, y = self.section_formula(x3, y3, x_, y_, 2 * d - n, n)
        temp[3][0] = [x, y]
        x, y = self.section_formula(x0, y0, x_up, y_up, 2 * d - n, n)
        xy_coords[0][0] = [x, y]
        x, y = self.section_formula(x3, y3, x_, y_, 2 * d + n, -n)
        temp[2][0] = [x, y]
        x, y = self.section_formula(x0, y0, x_up, y_up, 2 * d + n, -n)
        xy_coords[1][0] = [x, y]
        n_, d_ = self.height_of_black
        x, y = self.section_formula(temp[2][0][0], temp[2][0][1], xy_coords[1][0][0], xy_coords[1][0][1], n_, d_ - n_)
        temp[1][0] = [x, y]
        x, y = self.section_formula(temp[3][0][0], temp[3][0][1], xy_coords[0][0][0], xy_coords[0][0][1], n_, d_ - n_)
        temp[0][0] = [x, y]
        self.black.append(temp)

    def get_keyboard_keys(self):
        [[[x0, y0]], [[x1, y1]], [[x2, y2]], [[x3, y3]]] = self.pts
        n = self.num_octaves
        x_up, y_up = self.section_formula(x0, y0, x1, y1, 2, 7 * n)
        x_, y_ = self.section_formula(x3, y3, x2, y2, 2, 7 * n)
        pts = np.array([[[x0, y0]], [[x_up, y_up]], [[x_, y_]], [[x3, y3]]])
        self.add_minor_keys(pts)
        list_of_octaves = []
        pts = np.array([[[x_up, y_up]], [[x1, y1]], [[x2, y2]], [[x_, y_]]])
        self.add_octaves(pts, list_of_octaves, n)
        for i in range(n):
            self.add_white_keys(list_of_octaves[i])
            self.add_black_keys(list_of_octaves[i])

    def make_keyboard(self, img):
        img = cv2.fillPoly(img, self.white, (255, 255, 255))
        img = cv2.polylines(img, self.white, True, (255, 0, 255), 2)
        img = cv2.fillPoly(img, self.black, (0, 0, 0))
        return img

    def change_color(self, img, pressed_keys):
        white_keys = [self.white[i] for i in pressed_keys['White']]
        black_keys = [self.black[i] for i in pressed_keys['Black']]
        img = cv2.fillPoly(img, white_keys, (0, 255, 0))
        img = cv2.polylines(img, self.white, True, (255, 0, 255), 2)
        img = cv2.fillPoly(img, self.black, (0, 0, 0))
        img = cv2.fillPoly(img, black_keys, (128, 128, 128))
        return img

# Check Keys
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def tap_detection(previous_x, previous_y, x, y, threshold):
    keys_to_check = [4, 8, 12, 16, 20]
    tapped_keys = []
    for key in keys_to_check:
        dis = y[key] - previous_y[key]
        if dis > threshold:
            tapped_keys.append(key)
    return tapped_keys

def check_keys(x, y, white, black, white_piano_notes, black_piano_notes, tapped_keys):
    keys_to_check = tapped_keys
    pressed_notes = []
    pressed_keys = {"White": [], "Black": []}
    for key_to_check in keys_to_check:
        x1, y1 = x[key_to_check], y[key_to_check]
        flag = False
        for i, key in enumerate(black):
            distance = cv2.pointPolygonTest(np.array(key), (x1, y1), measureDist=False)
            if distance > 0:
                pressed_keys['Black'].append(i)
                if i < len(black_piano_notes):
                    pressed_notes.append(black_piano_notes[i])
                else:
                    st.warning(f"Black key index {i} out of range. Skipping note.")
                flag = True
                break
        if flag:
            continue
        for i, key in enumerate(white):
            distance = cv2.pointPolygonTest(np.array(key), (x1, y1), measureDist=False)
            if distance > 0:
                pressed_keys['White'].append(i)
                if i < len(white_piano_notes):
                    pressed_notes.append(white_piano_notes[i])
                else:
                    st.warning(f"White key index {i} out of range. Skipping note.")
                break
        pressed_notes = list(set(pressed_notes))
        pressed_keys['White'] = list(set(pressed_keys['White']))
        pressed_keys['Black'] = list(set(pressed_keys['Black']))
    return pressed_keys, pressed_notes

# Piano Sound
def play_piano_sound(notes):
    try:
        pygame.mixer.init()
        pygame.mixer.stop()
        channels = []
        for note in notes:
            file_path = os.path.join("notes", f"{note}.wav")
            if os.path.exists(file_path):
                sound_effect = pygame.mixer.Sound(file_path)
                channel = pygame.mixer.Channel(len(channels))
                channel.play(sound_effect)
                channels.append(channel)
            else:
                st.warning(f"Sound file {file_path} not found.")
    except Exception as e:
        st.error(f"Error playing sound: {e}")

# Top View
def make_connections(img, x, y, color=(0, 0, 255), thickness=2):
    connections = list(mp.solutions.hands.HAND_CONNECTIONS)
    for connection in connections:
        start_point = (x[connection[0]], y[connection[0]])
        end_point = (x[connection[1]], y[connection[1]])
        img = cv2.line(img, start_point, end_point, color, thickness)
    return img

def top_view(frame, x, z, diff):
    for x_, z_ in zip(x, z):
        y_ = int(frame.shape[0] * 5 / 6) + np.array(z_)
        frame = make_connections(frame, np.array(x_) - diff, y_)
    return frame

# Piano Configuration
def circle_fingertips(img, x, y):
    radius = 4
    color = (0, 0, 255)
    thickness = -1
    fingertips = [4, 8, 12]
    if len(x) > 0:
        for x_, y_ in zip(x, y):
            for tip in fingertips:
                cv2.circle(img, (x_[tip], y_[tip]), radius, color, thickness)
    if len(x) == 2:
        pts = [[[x[0][4], y[0][4]]], [[x[1][4], y[1][4]]], [[x[1][8], y[1][8]]], [[x[0][8], y[0][8]]]]
        img = cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, color, 2)
    return img

def get_coordinates(model_path, shape, distance_threshold):
    FRAME_WINDOW = st.image([])
    hd = HandsDetection(model_path)
    cap1 = cv2.VideoCapture(0)
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        frame = cv2.resize(frame, (shape[0], shape[1]))
        frame = cv2.flip(frame, 1)
        hd.detect(frame)
        frame = circle_fingertips(frame, hd.x, hd.y)
        FRAME_WINDOW.image(frame, channels="BGR")
        if len(hd.x) == 2:
            thumb_x0, thumb_y0 = hd.x[0][4], hd.y[0][4]
            thumb_x1, thumb_y1 = hd.x[1][4], hd.y[1][4]
            forefinger_x0, forefinger_y0 = hd.x[0][8], hd.y[0][8]
            forefinger_x1, forefinger_y1 = hd.x[1][8], hd.y[1][8]
            middlefinger_x0, middlefinger_y0 = hd.x[0][12], hd.y[0][12]
            middlefinger_x1, middlefinger_y1 = hd.x[1][12], hd.y[1][12]
            distance0 = cv2.norm(np.array((forefinger_x0, forefinger_y0)), np.array((middlefinger_x0, middlefinger_y0)), cv2.NORM_L2)
            distance1 = cv2.norm(np.array((forefinger_x1, forefinger_y1)), np.array((middlefinger_x1, middlefinger_y1)), cv2.NORM_L2)
            if distance1 < distance_threshold and distance0 < distance_threshold:
                pts = [[[thumb_x0, thumb_y0]], [[thumb_x1, thumb_y1]], [[forefinger_x1, forefinger_y1]], [[forefinger_x0, forefinger_y0]]]
                cap1.release()
                FRAME_WINDOW.empty()
                return pts

def piano_configuration(model_path, shape, distance_threshold):
    pts = get_coordinates(model_path, shape, distance_threshold)
    return pts

# Virtual Piano
class VirPiano:
    def __init__(self, model_path='model/hand_landmarker.task', num_octaves=2, list_of_octaves=[3, 4], 
                 height_and_width_black=[[5, 8], [5, 8]], shape=(800, 600, 3), tap_threshold=20, 
                 piano_config_threshold=30, piano_config=1):
        self.model_path = model_path
        self.hand_detection = HandsDetection(self.model_path)
        self.shape = shape
        self.image = np.zeros(self.shape, np.uint8)
        self.num_octaves = num_octaves
        self.list_of_octaves = list_of_octaves
        self.height_and_width_black = height_and_width_black
        self.tap_threshold = tap_threshold
        self.piano_config_threshold = piano_config_threshold
        self.pts = np.array([[[100, 350]], [[700, 350]], [[700, 550]], [[100, 550]]])
        self.piano_keyboard = Piano(self.pts, self.num_octaves, self.height_and_width_black)
        self.piano_config = piano_config
        self.x = []
        self.y = []
        self.z = []
        self.previous_x = []
        self.previous_y = []
        self.white_piano_notes, self.black_piano_notes = self.get_piano_notes()

    def get_piano_notes(self):
        white_piano_notes = ['A0', 'B0']
        black_piano_notes = ['Bb0']
        for i in range(self.list_of_octaves[0], self.list_of_octaves[1] + 1):
            for note in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                white_piano_notes.append(f"{note}{i}")
            for note in ['Bb', 'Db', 'Eb', 'Gb', 'Ab']:
                black_piano_notes.append(f"{note}{i}")
        expected_black_keys = self.num_octaves * 5 + 1
        while len(black_piano_notes) < expected_black_keys:
            black_piano_notes.append(f"Extra_Bb{len(black_piano_notes)}")
        return white_piano_notes, black_piano_notes

    def circle_fingertips(self, img):
        radius = 4
        color = (0, 0, 255)
        thickness = -1
        fingertips = [4, 8, 12, 16, 20]
        if len(self.x) > 0:
            for x, y in zip(self.x, self.y):
                for tip in fingertips:
                    cv2.circle(img, (x[tip], y[tip]), radius, color, thickness)
        return img

    def start(self):
        FRAME_WINDOW = st.image([])
        TOP_FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        previousTime = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video frame.")
                    break
                frame = cv2.resize(frame, (self.shape[0], self.shape[1]))
                frame = cv2.flip(frame, 1)

                if self.piano_config == 1:
                    self.pts = piano_configuration(self.model_path, self.shape, self.piano_config_threshold)
                    self.piano_keyboard = Piano(self.pts, self.num_octaves, self.height_and_width_black)
                    self.piano_config = 0
                    cap.release()
                    cap = cv2.VideoCapture(0)
                    continue

                self.image = frame.copy()
                self.hand_detection.detect(frame)
                self.x = self.hand_detection.x
                self.y = self.hand_detection.y
                self.z = self.hand_detection.z

                self.image = self.piano_keyboard.make_keyboard(self.image)

                pressed_keys = {"White": [], "Black": []}
                pressed_notes = []
                if len(self.previous_x) == len(self.x):
                    for (x_, y_), (previous_x_, previous_y_) in zip(zip(self.x, self.y), zip(self.previous_x, self.previous_y)):
                        tapped_keys = tap_detection(previous_x_, previous_y_, x_, y_, self.tap_threshold)
                        keys, notes = check_keys(x_, y_, self.piano_keyboard.white, self.piano_keyboard.black, 
                                                 self.white_piano_notes, self.black_piano_notes, tapped_keys)
                        for note in notes:
                            pressed_notes.append(note)
                        for w in keys['White']:
                            pressed_keys['White'].append(w)
                        for b in keys['Black']:
                            pressed_keys['Black'].append(b)

                self.image = self.piano_keyboard.change_color(self.image, pressed_keys)

                top_view_shape = (500, 250)
                pts1 = np.reshape(np.array(self.pts, dtype=np.float32), (-1, 2))
                pts2 = np.float32([[0, 0], [top_view_shape[0], 0], [top_view_shape[0], top_view_shape[1]], [0, top_view_shape[1]]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                top_view_image = cv2.warpPerspective(self.image, M, top_view_shape)
                top_view_image = cv2.flip(top_view_image, 0)
                top_view_image = top_view(top_view_image, self.x, self.z, self.pts[3][0][0])
                TOP_FRAME_WINDOW.image(top_view_image, channels="BGR", caption="Top View")

                self.image = self.circle_fingertips(self.image)

                self.previous_x = self.x
                self.previous_y = self.y

                if len(pressed_notes) > 0:
                    play_piano_sound(pressed_notes)

                currentTime = time.time()
                fps = 1 / (currentTime - previousTime) if currentTime > previousTime else 0
                previousTime = currentTime
                cv2.putText(self.image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (88, 205, 54), 3)

                FRAME_WINDOW.image(self.image, channels="BGR", caption="Piano View")

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()

# Streamlit App
def main():
    st.set_page_config(page_title="Virtual Piano", layout="wide")
    st.title("Virtual Piano Interaktif")

    col1, col2 = st.columns(2)
    with col1:
        num_octaves = st.slider("Number of octaves:", min_value=1, max_value=7, value=2, step=1)
        start_number = st.slider("Starting octave number:", min_value=1, max_value=7, value=3, step=1)
        end_number = st.slider("Ending octave number:", min_value=start_number, max_value=7, value=4, step=1)
        list_of_octaves = [start_number, end_number]
        piano_config_bool = st.checkbox('Configure Piano', value=True)
        piano_config = 1 if piano_config_bool else 0
    with col2:
        tap_threshold = st.number_input("Tapping threshold:", min_value=1, max_value=50, value=20, step=1)
        piano_config_threshold = st.number_input("Threshold for piano configuration:", min_value=1, max_value=50, value=30, step=1)
        height_and_width_black = st.text_area("Height and width of black keys:", "[[5, 8], [5, 8]]")
        try:
            height_and_width_black = ast.literal_eval(height_and_width_black)
        except (ValueError, SyntaxError):
            st.warning("Invalid input for black keys dimensions. Using default [[5, 8], [5, 8]].")
            height_and_width_black = [[5, 8], [5, 8]]

    stop_button = st.button("Stop Playing")
    if not stop_button:
        vp = VirPiano(
            num_octaves=num_octaves,
            list_of_octaves=list_of_octaves,
            height_and_width_black=height_and_width_black,
            tap_threshold=tap_threshold,
            piano_config_threshold=piano_config_threshold,
            piano_config=piano_config
        )
        vp.start()
    else:
        st.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='model/hand_landmarker.task')
    parser.add_argument('--num_octaves', default=2, type=int)
    parser.add_argument('--list_of_octaves', default=[3, 4], type=ast.literal_eval)
    parser.add_argument('--height_and_width_black', default=[[5, 8], [5, 8]], type=ast.literal_eval)
    parser.add_argument('--shape', default=(800, 600, 3), type=ast.literal_eval)
    parser.add_argument('--tap_threshold', default=20, type=int)
    parser.add_argument('--piano_config_threshold', default=30, type=int)
    parser.add_argument('--piano_config', default=1, type=int)
    args = parser.parse_args()

    main()