import threading
from deepface import DeepFace
import cv2
import time

# List of available backends, models, and distance metrics
backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]

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

def face_recognition(video_stream):
    frame_count = 0
    skip_frames = 2  # Process every 2nd frame
    detected_faces = []  # List to store detected faces and their coordinates

    while True:
        # Read frame from the video stream
        frame = video_stream.read()

        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        if frame_count % skip_frames == 0:
            # Perform face recognition on the resized frame
            try:
                people = DeepFace.find(img_path=small_frame, db_path="database/", model_name="Facenet512", distance_metric="euclidean_l2", enforce_detection=False)
                
                # Update the detected faces list
                detected_faces = []
                for person in people:
                    x = int(person['source_x'][0] * 2)
                    y = int(person['source_y'][0] * 2)
                    w = int(person['source_w'][0] * 2)
                    h = int(person['source_h'][0] * 2)
                    name = person['identity'][0].split('/')[1]
                    detected_faces.append((x, y, w, h, name))
            except:
                detected_faces = []

        # Draw rectangles and labels for detected faces
        for (x, y, w, h, name) in detected_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 960, 720)
        cv2.imshow('frame', frame)

        # Check if the 'q' button is pressed to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    video_stream.stop()
    cv2.destroyAllWindows()

def main():
    # Start the video capture thread
    video_stream = VideoCaptureThread().start()

    # Run face recognition on the main thread
    face_recognition(video_stream)

if __name__ == "__main__":
    main()
