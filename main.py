import cv2
import numpy as np
import pygame
import os
import threading
from gtts import gTTS
import uuid
import time
import queue
from flask import Flask, render_template_string, url_for

# Create the sounds directory if it doesn't exist
if not os.path.exists("./sounds"):
    os.makedirs("./sounds")

# Global variables for managing announcements
last_announcement = ""
last_announcement_time = 0
announcement_cooldown = 5  # seconds

# Global flag to control the detection loop
detection_running = False

# Create a queue to store announcements
announcement_queue = queue.Queue()

# Define a list of obstacle classes (from COCO dataset) relevant for navigation
obstacle_classes = [
    "person", "bicycle", "car", "motorbike", "bus", "truck", "chair", "sofa",
    "table", "pottedplant", "bottle", "traffic light", "stop sign"
]

# Function to generate and play speech using gTTS and pygame
def speech(text):
    unique_filename = f"./sounds/output_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(unique_filename)
    
    pygame.mixer.init()
    if os.path.exists(unique_filename):
        print(f"Playing file {unique_filename}...")
        pygame.mixer.music.load(unique_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        print("Playback finished.")
        os.remove(unique_filename)
    else:
        print(f"Error: {unique_filename} does not exist.")

# Announcement player thread function to sequentially play queued announcements
def announcement_player():
    while True:
        message = announcement_queue.get()  # Blocks until a message is available
        if message is None:
            break  # Signal to exit
        print(f"Announcing: {message}")
        speech(message)
        announcement_queue.task_done()

# Start the announcement player thread (daemon thread)
announcement_thread = threading.Thread(target=announcement_player, daemon=True)
announcement_thread.start()

# Load YOLO model
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

# Load COCO class names
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the camera (it will be opened in the detection thread)
cap = cv2.VideoCapture(0)

# Function to detect stairs using basic image processing
def detect_stairs(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    count_horizontal = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
            if abs(angle) < 15:  # Nearly horizontal
                count_horizontal += 1
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if count_horizontal > 10:
        return True, f"Stairs detected with {count_horizontal} horizontal edges."
    else:
        return False, ""

def estimate_depth(box, frame_area):
    x, y, w, h = box
    box_area = w * h
    ratio = box_area / frame_area
    if ratio > 0.15:
        return "very close"
    elif ratio > 0.05:
        return "close"
    else:
        return "far away"

def detect_and_announce():
    global last_announcement, last_announcement_time, detection_running, cap
    detection_running = True
    if not cap.isOpened():
        cap.open(0)

    while detection_running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting detection loop.")
            break

        height, width, channels = frame.shape
        frame_area = width * height

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []
        instructions = []
        obstacles_detected = {"left": 0, "center": 0, "right": 0}

        for out in outs:
            out = out.reshape(-1, 85)
            for obj in out:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        if len(indexes) > 0:
            for i in indexes.flatten():
                label = str(classes[class_ids[i]])
                x, y, w, h = boxes[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
                
                object_center_x = x + w / 2
                if object_center_x < width / 3:
                    horizontal_pos = "on your left"
                    pos_key = "left"
                elif object_center_x > (2 * width / 3):
                    horizontal_pos = "on your right"
                    pos_key = "right"
                else:
                    horizontal_pos = "in front of you"
                    pos_key = "center"

                depth = estimate_depth([x, y, w, h], frame_area)
                instruction = f"Your {label} is {horizontal_pos} and {depth}"
                instructions.append(instruction)
                if label in obstacle_classes and depth != "far away":
                    obstacles_detected[pos_key] += 1

        stairs_detected, stairs_msg = detect_stairs(frame)
        if stairs_detected:
            instructions.append("Stairs detected ahead")

        navigation_advice = ""
        if obstacles_detected["center"] > 0:
            if obstacles_detected["left"] <= obstacles_detected["right"]:
                navigation_advice = "Warning: Obstacle ahead. Consider moving to your left."
            else:
                navigation_advice = "Warning: Obstacle ahead. Consider moving to your right."
        else:
            navigation_advice = "Path ahead is clear."
        if obstacles_detected["left"] > 0 and obstacles_detected["center"] == 0:
            navigation_advice += " However, there are obstacles on your left."
        if obstacles_detected["right"] > 0 and obstacles_detected["center"] == 0:
            navigation_advice += " However, there are obstacles on your right."

        full_message = ""
        if instructions:
            full_message = ", ".join(instructions) + ". " + navigation_advice

        cv2.imshow("Detection Feed", frame)
        if full_message:
            current_time = time.time()
            if (full_message != last_announcement) or (current_time - last_announcement_time > announcement_cooldown):
                print(f"Queueing: {full_message}")
                announcement_queue.put(full_message)
                last_announcement = full_message
                last_announcement_time = current_time

        if cv2.waitKey(1) & 0xFF == 27:
            detection_running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# Flask web app integration
app = Flask(__name__)
detection_thread = None

# Main index page HTML template
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Object Detection for Visually Impaired Users</title>
    <style>
        body {
            font-size: 1.5em;
            font-family: Arial, sans-serif;
            background-color: #f8f9fa;
            color: #212529;
            text-align: center;
            padding: 20px;
        }
        h1 {
            color: #0056b3;
            font-size: 2.5em;
            margin: 10px;
        }
        p {
            font-size: 1em;
        }
        button {
            font-size: 1.2em;
            padding: 10px 20px;
            margin: 10px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            background-color: #007bff;
            color: white;
        }
        button:hover {
            background-color: #0056b3;
        }
        .start-btn {
            background-color: #28a745;
        }
        .stop-btn {
            background-color: #dc3545;
        }
        .img-container {
            margin-top: 20px;
        }
        .img-container img {
            width: 100%;
            height: auto;
            object-fit: cover;
            border: 2px solid #007bff;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <h1>Object Detection for Visually Impaired Users</h1>
    <p>This system assists visually impaired users by detecting obstacles, stairs, and providing audio instructions.</p>
    <button class="start-btn" onclick="location.href='/start'">Start Detection</button>
    <button class="stop-btn" onclick="location.href='/stop'">Stop Detection</button>
    <div class="img-container">
        <p>Customer Using The Model</p>
        <img src="{{ url_for('static', filename='accessibility.jpeg') }}" alt="Accessibility Assistance">
    </div>
</body>
</html>
'''

# Template for the message pages (Detection Started / Stopped)
message_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body {
            font-size: 1.5em;
            font-family: Arial, sans-serif;
            background-color: #f8f9fa;
            color: #212529;
            text-align: center;
            padding: 20px;
        }
        h1 {
            color: #0056b3;
            font-size: 2.5em;
            margin: 20px;
        }
        a {
            font-size: 1.2em;
            text-decoration: none;
            color: #007bff;
            border: 1px solid #007bff;
            padding: 10px 20px;
            border-radius: 5px;
        }
        a:hover {
            background-color: #007bff;
            color: white;
        }
    </style>
</head>
<body>
    <h1>{{ message }}</h1>
    <p><a href="/">Return Home</a></p>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/start')
def start_detection():
    global detection_thread, detection_running
    if detection_thread is None or not detection_thread.is_alive():
         detection_thread = threading.Thread(target=detect_and_announce, daemon=True)
         detection_thread.start()
         return render_template_string(message_template, title="Detection Started", message="Detection Started.")
    else:
         return render_template_string(message_template, title="Already Running", message="Detection is already running.")

@app.route('/stop')
def stop_detection():
    global detection_running
    detection_running = False
    return render_template_string(message_template, title="Detection Stopped", message="Detection Stopped.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
    # Signal the announcement thread to exit when Flask shuts down
    announcement_queue.put(None)
    announcement_thread.join()
