import datetime
import os
import time

import cv2
import imutils
from PIL import ImageGrab

from keras.preprocessing.image import img_to_array
from keras.models import load_model

import numpy as np
import pandas as pd


class EmotionTracker:
    """Helper to gather facial expression data and output to CSV

    :param input_video: Path to input video (see `cv2.VideoCapture()`)
    :param sample_rate: Seconds to wait between facial snapshot
    :param output: Name of directory for data output
    :param model_path: Path to hdf5 format model for classifying facial expression emotion
    :param haarcascade_path: Path to haarcascade face detector

    :ivar input_video: Path to input video (see `cv2.VideoCapture()`)
    :ivar sample_rate: Seconds to wait between facial snapshot
    :ivar face_detector: Face detector (see `cv2.CascadeClassifier()`)
    :ivar model: keras model for classifying facial expression emotion
    :ivar emotions: list of emotion names considered
    :ivar file_name: Name of file the data will be saved to
    """

    def __init__(
        self,
        input_video=0,
        sample_rate=15,
        output="data",
        model_path="model/keras_emotion_mod.hdf5",
        haarcascade_path="haarcascade/haarcascade_frontalface_default.xml",
    ):
        self.input_video = input_video
        self.sample_rate = sample_rate

        self.face_detector = cv2.CascadeClassifier(haarcascade_path)
        self.model = load_model(model_path)

        self.emotions = ["angry", "scared", "happy", "sad", "surprised", "neutral"]
        self.file_name = os.path.join(
            output, datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S.csv")
        )

    @staticmethod
    def _rect_area(r):
        """Calculate area of bounding box (used to find largest face in frame)"""
        return (r[2] - r[0]) * (r[3] - r[1])

    def _predict_emotion(self, img, predict_width=300):
        """Classify emotions of largest face in an image

        :param img: Image to classify (numpy array)
        :param predict_width: Resizing done to image before predictions
        :return: Dictionary of form {emotion_name: confidence_level, ...}
        """
        # Preprocess image
        img = imutils.resize(img, width=predict_width)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect face_caps in image
        rects = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Return na predictions if no face_caps
        if len(rects) <= 0:
            return {e: np.nan for e in self.emotions}

        # Get biggest face location
        rect = sorted(rects, reverse=True, key=self._rect_area)[0]
        x, y, w, h = rect

        # Pre-process face for model
        roi = gray[y : y + h, x : x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict emotions and reformat
        probs = self.model.predict(roi)[0]
        return {e: p for e, p in zip(self.emotions, probs)}

    def gather_data(self, cap_rate=30, face_cap_dir=None, screen_cap_dir=None):
        """Capture facial data from input video

        Ctrl+C to quit

        :return: None; data is written as a csv to the location
                 specified in the `file_name` attribute
        """
        vidcap = cv2.VideoCapture(self.input_video)

        last_pred = None
        sec_since_pred = self.sample_rate
        sec_since_cap = cap_rate

        # Try-catch for quitting program with Ctrl+C gracefully
        try:
            while True:
                # Wait for sample rate
                if sec_since_pred >= self.sample_rate:
                    _, frame = vidcap.read()

                    screen_cap = np.zeros((312, 500, 3))
                    if screen_cap_dir is not None and sec_since_cap >= cap_rate:
                        screen_cap = ImageGrab.grab()
                        screen_cap = cv2.cvtColor(
                            np.array(screen_cap), cv2.COLOR_RGBA2BGR
                        )
                        screen_cap = imutils.resize(screen_cap, width=500)

                    preds = self._predict_emotion(frame)
                    last_pred = time.time()

                    time_stamp = datetime.datetime.utcnow()
                    time_stamp_filename = time_stamp.strftime("%Y_%m_%d__%H_%M_%S.jpg")
                    preds["timestamp"] = time_stamp.strftime("%Y-%m-%d %H:%M:%S")

                    preds_df = pd.DataFrame(
                        [preds], columns=["timestamp"] + self.emotions
                    )

                    # Create output data as needed (or append if exists)
                    file_exists = os.path.exists(self.file_name)
                    write_flag = "a" if file_exists else "w"
                    preds_df.to_csv(
                        self.file_name,
                        mode=write_flag,
                        index=False,
                        header=(not file_exists),
                    )

                    if face_cap_dir is not None and sec_since_cap >= cap_rate:
                        face_cap_file = os.path.join(face_cap_dir, time_stamp_filename)
                        cv2.imwrite(face_cap_file, frame)

                    if screen_cap_dir is not None and sec_since_cap >= cap_rate:
                        screen_cap_file = os.path.join(
                            screen_cap_dir, time_stamp_filename
                        )
                        cv2.imwrite(screen_cap_file, screen_cap)
                else:
                    time.sleep(1)

                sec_since_pred = time.time() - last_pred
                sec_since_cap = time.time() - last_pred
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--sample_rate", type=int, default=15)
    args = vars(ap.parse_args())

    emotion_tracker = EmotionTracker(sample_rate=args["sample_rate"])
    emotion_tracker.gather_data(
        cap_rate=30,
        face_cap_dir="images/face_caps",
        screen_cap_dir="images/screen_caps",
    )
