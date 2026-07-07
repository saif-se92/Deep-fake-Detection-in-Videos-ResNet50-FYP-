from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model("resnet_final (1)_auc100.keras", compile=False)

CLASS_NAMES = ["REAL","FAKE"]

# ========================
# FIXED VIDEO PREDICTION PIPELINE
# ========================

def predict_video(frames):
    if len(frames) == 0:
        return 0.0

    frames = np.array(frames)

    # Model prediction
    preds = model.predict(frames, verbose=0)

    # If output is (N,1) sigmoid
    preds = np.array(preds).flatten()

    avg = np.mean(preds)

    # Debug (VERY IMPORTANT)
    print("Raw predictions (first 10):", preds[:10])
    print("Average prediction:", avg)

    return avg


def extract_frames(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(frame_count // max_frames, 1)

    idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))

            # IMPORTANT: ResNet preprocessing
            frame = preprocess_input(frame)

            frames.append(frame)

        idx += 1

    cap.release()
    return np.array(frames)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video = request.files["video"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
        video.save(path)

        frames = extract_frames(path)
        avg = predict_video(frames)

        # FIXED decision logic
        label = "FAKE" if avg >= 0.5 else "REAL"

        confidence = abs(avg - 0.5) * 2 * 100

        return render_template(
            "result.html",
            video_path=path,
            prediction=label,
            confidence=round(confidence, 2)
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)


# from flask import Flask, render_template, request
# import tensorflow as tf
# import numpy as np
# import cv2
# import os
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
# from tensorflow.keras.models import Model

# app = Flask(__name__)

# UPLOAD_FOLDER = "static/uploads"
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# # ========================
# # resnet = ResNet50(weights="imagenet", include_top=False)

# # Your saved classifier
# # head = tf.keras.models.load_model("ResNet50_Deepfake.h5", compile=False)
# model = tf.keras.models.load_model("resnet_best.keras", compile=False)
# CLASS_NAMES = ["REAL","FAKE"]
# def predict_video(frames):
#      if len(frames) == 0:
#         return 0.0
#     #  features = resnet.predict(frames)
#      preds = model.predict(frames)
#      return np.mean(preds)
# def extract_frames(video_path, max_frames=20):
#     cap = cv2.VideoCapture(video_path)
#     frames = []

#     while len(frames) < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.resize(frame, (224, 224))
#         frame = frame / 255.0
#         frames.append(frame)

#     cap.release()
#     return np.array(frames)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         video = request.files["video"]
#         path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
#         video.save(path)

#         # frames = extract_frames(path)
#         # preds = model.predict(frames)
#         # avg = np.mean(preds)

#         frames = extract_frames(path)
#         avg = predict_video(frames)


#         label = CLASS_NAMES[int(avg > 0.5)]
#         confidence = avg * 100 if label == "FAKE" else (1 - avg) * 100

#         return render_template(
#             "result.html",
#             video_path=path,
#             prediction=label,
#             confidence=round(confidence, 2)
#         )

#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)

