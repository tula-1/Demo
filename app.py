from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(_name_)

# Load the pre-trained model
model = model_from_json(open(
    "C:\\Users\\Divya\\OneDrive\\Documents\\FRONTEND\\model_filter.json", "r").read())
model.load_weights(
    'C:\\Users\\Divya\\OneDrive\\Documents\\FRONTEND\\model_filter.h5')

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(
    'C:\\Users\\Divya\\OneDrive\\Documents\\FRONTEND\\haarcascade_frontalface_default1.xml')

# Function to perform emotion detection on a given frame
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        image_pixels = img_to_array(roi_gray)
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels /= 255

        predictions = model.predict(image_pixels)
        max_index = np.argmax(predictions[0])
        emotion_labels = ['Angry', 'Disgust', 'Fear',
                          'Happy', 'Neutral', 'Sad', 'Surprise']
        emotion = emotion_labels[max_index]

        # Draw rectangle and emotion text on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# Function to generate video feed
def generate():
    video_capture = cv2.VideoCapture(0)

    while True:
        # Read the video frame
        success, frame = video_capture.read()

        if not success:
            break

        # Perform emotion detection on the frame
        frame = detect_emotion(frame)

        # Encode the frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    video_capture.release()

# Flask route for rendering the web page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=8000, debug=True)
