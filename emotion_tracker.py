import datetime
import os
import time
import cv2
import imutils
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pandas as pd


class EmotionTracker:
    def __init__(self,
                 input_video=0,
                 sample_rate=30,
                 output='data',
                 model_path='model/keras_emotion_mod.hdf5',
                 haarcascade_path='haarcascade/haarcascade_frontalface_default.xml'):
        self.input_video = input_video
        self.sample_rate = sample_rate

        self.face_detector = cv2.CascadeClassifier(haarcascade_path)
        self.model = load_model(model_path)

        self.emotions = ['angry', 'scared', 'happy', 'sad', 'surprised', 'neutral']
        self.file_name = os.path.join(output,
                                      datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S.csv'))

    @staticmethod
    def _rect_area(r):
        return (r[2] - r[0]) * (r[3] - r[1])

    def _predict_emotion(self, img, predict_width=300):
        img = imutils.resize(img, width=predict_width)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in image
        rects = self.face_detector.detectMultiScale(gray, scaleFactor=1.1,
                                                    minNeighbors=5, minSize=(30, 30),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)

        # Return na predictions if no faces
        if len(rects) <= 0:
            return {e: np.nan for e in self.emotions}

        # Get biggest face location
        rect = sorted(rects, reverse=True, key=self._rect_area)[0]
        x, y, w, h = rect

        # Pre-process face for model
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        probs = self.model.predict(roi)[0]
        return {e: p for e, p in zip(self.emotions, probs)}

    def gather_data(self):
        vidcap = cv2.VideoCapture(self.input_video)

        last_pred = None
        sec_since_pred = self.sample_rate

        try:
            while True:
                if sec_since_pred >= self.sample_rate:
                    _, frame = vidcap.read()
                    preds = self._predict_emotion(frame)
                    last_pred = time.time()

                    preds['timestamp'] = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                    preds_df = pd.DataFrame([preds], columns=['timestamp'] + self.emotions)

                    write_flag = 'a' if os.path.exists(self.file_name) else 'w'
                    preds_df.to_csv(self.file_name, mode=write_flag, index=False)
                else:
                    time.sleep(1)

                sec_since_pred = time.time() - last_pred
        except KeyboardInterrupt:
            return


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--sample_rate', type=int, default=30)
    args = vars(ap.parse_args())

    emotion_tracker = EmotionTracker(sample_rate=args['sample_rate'])
    emotion_tracker.gather_data()
