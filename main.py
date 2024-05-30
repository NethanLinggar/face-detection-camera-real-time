import threading
import time
import os
from dotenv import load_dotenv
import cv2
from deepface import DeepFace
import requests
from datetime import datetime, timedelta
import queue

# List of available backends, models, and distance metrics
# backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
# models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
# metrics = ["cosine", "euclidean", "euclidean_l2"]

load_dotenv()

db_path = os.getenv("DATABASE_PATH")
model_name = os.getenv("MODEL_NAME")
detector_backend = os.getenv("DETECTOR_BACKEND")
distance_metric = os.getenv("DISTANCE_METRIC")

class VideoCaptureThread:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.capture.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            self.ret, self.frame = self.capture.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.capture.release()

class FaceRecognitionThread:
    def __init__(self, frame_queue, result_queue, skip_frames=30):
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.stopped = False

    def start(self):
        threading.Thread(target=self.recognize_faces, args=()).start()
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
                    people = DeepFace.find(img_path=small_frame, db_path=db_path, model_name=model_name, distance_metric=distance_metric, enforce_detection=False)
                    detected_faces = []
                    for person in people:
                        x = int(person['source_x'][0] * 2)
                        y = int(person['source_y'][0] * 2)
                        w = int(person['source_w'][0] * 2)
                        h = int(person['source_h'][0] * 2)
                        name = person['identity'][0].split('/')[1]
                        detected_faces.append((x, y, w, h, name))

                        # Check if it's time to send a POST request for this face
                        current_time = datetime.now()
                        if name not in last_post_times or current_time - last_post_times[name] >= timedelta(minutes=30):
                            data = {
                                "userNrp": os.path.basename(name).split('.')[0]
                            }
                            response = requests.post("http://localhost:3000/logs", json=data)
                            if response.status_code == 200:
                                print(f"Successfully logged face detection for {name}.")
                            else:
                                print(f"Failed to log face detection for {name}.")
                            last_post_times[name] = current_time

                    self.result_queue.put(detected_faces)

                except Exception as e:
                    print(f"Error during face recognition: {e}")
                    self.result_queue.put([])

            self.frame_count += 1

    def stop(self):
        self.stopped = True

def face_recognition(video_stream):
    frame_queue = queue.Queue(maxsize=10)
    result_queue = queue.Queue(maxsize=10)
    recognition_thread = FaceRecognitionThread(frame_queue, result_queue).start()
    detected_faces = []  # List to store detected faces and their coordinates

    while True:
        frame = video_stream.read()
        frame_queue.put(frame)

        # Update detected faces if new results are available
        if not result_queue.empty():
            detected_faces = result_queue.get()

        for (x, y, w, h, name) in detected_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, os.path.basename(name).split('.')[0], (x, y), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 960, 720)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_stream.stop()
    recognition_thread.stop()
    cv2.destroyAllWindows()

def main():
    video_stream = VideoCaptureThread().start()
    face_recognition(video_stream)

if __name__ == "__main__":
    main()
