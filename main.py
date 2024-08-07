import threading
import time
import os
from dotenv import load_dotenv
import cv2
from deepface import DeepFace
import requests
import json
from datetime import datetime, timedelta
import queue

load_dotenv()

db_path = os.getenv("DATABASE_PATH")
model_name = "Facenet512"
detector_backend = "opencv"
distance_metric = "cosine"

class VideoCaptureThread:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.capture.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.capture.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.capture.release()

class FaceRecognitionThread:
    def __init__(self, frame_queue, result_queue, skip_frames=10):
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.stopped = False
        self.latest_requests = []  # Store latest request names
        self.detection_counts = {}  # Dictionary to keep track of detection counts
        self.start_time = time.time()  # Start time for FPS calculation

    def start(self):
        threading.Thread(target=self.recognize_faces, args=(), daemon=True).start()
        return self

    def recognize_faces(self):
        last_post_times = {}  # Dictionary to store the last post time for each detected face
        while not self.stopped:
            frame = self.frame_queue.get()
            if frame is None:
                continue

            if self.frame_count % self.skip_frames == 0:
                try:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    face_objs = DeepFace.extract_faces(img_path=small_frame, detector_backend=detector_backend)
                    people_found = DeepFace.find(img_path=small_frame, db_path=db_path, model_name=model_name, detector_backend=detector_backend, distance_metric=distance_metric, enforce_detection=False)
                    detected_faces = []
                    processed_coords = set()

                    for person in people_found:
                        if not person.empty:
                            x = int(person['source_x'][0] * 2)
                            y = int(person['source_y'][0] * 2)
                            w = int(person['source_w'][0] * 2)
                            h = int(person['source_h'][0] * 2)
                            name = os.path.basename(person['identity'][0].split('/')[1]).split('.')[0]
                            if (x, y, w, h) not in processed_coords:
                                detected_faces.append((x, y, w, h, name))
                                processed_coords.add((x, y, w, h))
                                
                                # Update detection count
                                if name not in self.detection_counts:
                                    self.detection_counts[name] = 0
                                self.detection_counts[name] += 1

                                # Check if the detection count has reached 5
                                current_time = datetime.now()
                                if self.detection_counts[name] == 5 and (name not in last_post_times or current_time - last_post_times[name] >= timedelta(minutes=10)):
                                    data = {"userNrp": name}
                                    response = requests.post("http://localhost:3000/logs", json=data)
                                    if response.status_code == 200:
                                        print(f"Successfully logged face detection for {name}.")
                                        # Update latest requests
                                        self.latest_requests.append(name)
                                        if len(self.latest_requests) > 2:
                                            self.latest_requests.pop(0)
                                        # Update the last post time
                                        last_post_times[name] = current_time
                                    else:
                                        print(f"Failed to log face detection for {name}.")
                                    # Reset detection count
                                    self.detection_counts[name] = 0

                    for face_obj in face_objs:
                        facial_area = face_obj['facial_area']
                        x = int(facial_area['x'] * 2)
                        y = int(facial_area['y'] * 2)
                        w = int(facial_area['w'] * 2)
                        h = int(facial_area['h'] * 2)
                        overlap = False
    
                        # Check for overlap with existing detected faces
                        for (dx, dy, dw, dh, dname) in detected_faces:
                            if (x < dx + dw and x + w > dx and y < dy + dh and y + h > dy):
                                overlap = True
                                break

                        if not overlap and (x, y, w, h) not in processed_coords:
                            detected_faces.append((x, y, w, h, "Unknown"))
                            processed_coords.add((x, y, w, h))

                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    folder_path = os.path.join('test_' + model_name + '_' + detector_backend, name)
                    os.makedirs(folder_path, exist_ok=True)
                    # Save frames with detected faces
                    for (x, y, w, h, name) in detected_faces:
                        file_path = os.path.join(folder_path, f'{timestamp}.jpg')
                        cv2.imwrite(file_path, frame)
 
                    self.result_queue.put(detected_faces)

                except Exception as e:
                    print(f"Error during face recognition: {e}")
                    self.result_queue.put([])

            self.frame_count += 1

    def calculate_fps(self):
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        return fps

    def stop(self):
        self.stopped = True

class UserJsonUpdaterThread:
    def __init__(self, interval=10):
        self.interval = interval
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update_users, args=(), daemon=True).start()
        return self

    def update_users(self):
        while not self.stopped:
            save_users_to_json()
            time.sleep(self.interval)

    def stop(self):
        self.stopped = True

def save_users_to_json():
    response = requests.get("http://localhost:3000/users")
    if response.status_code == 200:
        users = response.json()
        with open('users.json', 'w') as f:
            json.dump(users, f)
    else:
        print("Failed to fetch users.")

def load_users_from_json():
    with open('users.json', 'r') as f:
        return json.load(f)['data']  # Access the 'data' key in the JSON response

def face_recognition(video_stream):
    frame_queue = queue.Queue(maxsize=10)
    result_queue = queue.Queue(maxsize=10)
    recognition_thread = FaceRecognitionThread(frame_queue, result_queue).start()
    detected_faces = []  # List to store detected faces and their coordinates
    users = load_users_from_json()  # Load users from the JSON file

    # To detect changes in the JSON file, track the last modified time
    users_json_path = 'users.json'
    last_modified_time = os.path.getmtime(users_json_path)

    while True:
        frame = video_stream.read()
        frame_queue.put(frame)

        # Check if the users.json file has been updated
        current_modified_time = os.path.getmtime(users_json_path)
        if current_modified_time != last_modified_time:
            users = load_users_from_json()
            last_modified_time = current_modified_time

        # Update detected faces if new results are available
        if not result_queue.empty():
            detected_faces = result_queue.get()

        for (x, y, w, h, name) in detected_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if name != "Unknown":
                cv2.putText(frame, os.path.basename(name).split('.')[0], (x, y), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

        # Access the latest requests from the recognition thread
        latest_requests = recognition_thread.latest_requests

        # Display the two latest requests at the bottom of the frame
        for i, request_name in enumerate(latest_requests[-2:]):
            # Find the associated name from the JSON data
            display_name = next((user['name'] for user in users if user['nrp'] == request_name), request_name)
            display_text = f"Hello (ID: {request_name}) {display_name}!"
            y_position = frame.shape[0] - (i + 1) * 30  # Position from the bottom
            cv2.putText(frame, display_text, (10, y_position), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

        # Calculate and display FPS
        fps = recognition_thread.calculate_fps()
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 960, 720)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.stop()
    recognition_thread.stop()
    cv2.destroyAllWindows()

def main():
    save_users_to_json()  # Save users to JSON file before starting face recognition
    video_stream = VideoCaptureThread().start()
    user_updater_thread = UserJsonUpdaterThread(interval=10).start()  # Start the user updater thread
    face_recognition(video_stream)

if __name__ == "__main__":
    main()
