import cv2
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("emotion_model.h5")

# Emotion labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

# Load Haar Cascade
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# START CAMERA
# =========================
cap = cv2.VideoCapture(0)

# Better camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

print("Press Q to quit.")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60),
    )

    for x, y, w, h in faces:

        # Extract face ROI
        roi_gray = gray[y : y + h, x : x + w]

        try:
            # Resize to model input size
            roi_gray = cv2.resize(roi_gray, (48, 48))

            # Normalize
            roi = roi_gray.astype("float32") / 255.0

            # Reshape for model
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            # Prediction
            prediction = model.predict(roi, verbose=0)[0]

            max_index = np.argmax(prediction)

            emotion = emotion_labels[max_index]
            confidence = prediction[max_index] * 100

            # Dynamic color
            color = (0, 255, 0)

            if emotion == "Angry":
                color = (0, 0, 255)

            elif emotion == "Happy":
                color = (0, 255, 255)

            elif emotion == "Sad":
                color = (255, 0, 0)

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Label text
            text = f"{emotion} ({confidence:.1f}%)"

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        except Exception as e:
            print("Face processing error:", e)

    # Window title
    cv2.imshow("AI Emotion Detection", frame)

    # IMPORTANT:
    # Proper close handling
    key = cv2.waitKey(1)

    # Press Q or close window
    if key & 0xFF == ord("q"):
        break

    # Detect manual window close
    if cv2.getWindowProperty("AI Emotion Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# =========================
# CLEAN EXIT
# =========================
cap.release()
cv2.destroyAllWindows()

# Extra cleanup
cv2.waitKey(1)
