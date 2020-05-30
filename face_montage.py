"""Display expression highlights from each session in a montage"""
import os
import glob
from collections import defaultdict
import numpy as np
import cv2
import imutils

N_COLS = 3
N_REPEATS = 10

# Read
image_paths = glob.glob("images/face_caps/*/*.jpg")
image_paths.sort()
images = defaultdict(lambda: [])
for path in image_paths:
    image = cv2.imread(path)

    path_dir, file_name = os.path.split(path)
    _, emotion = os.path.split(path_dir)

    images[emotion].append(image)


min_len = min(len(x) for x in images.values())
emotions = list(images.keys())

# Build montages
montages = []
rows = []
for i in range(min_len):
    row = []
    for emotion in emotions:
        im = images[emotion][i]
        cv2.putText(im, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        row.append(im)

        if len(row) == N_COLS:
            rows.append(np.hstack(row))
            row = []

    montages.append(np.vstack(rows))
    rows = []

# Display
for _ in range(N_REPEATS):
    for montage in montages:
        im = imutils.resize(montage, width=1000)
        cv2.imshow("Feelings", im)
        key = cv2.waitKey(500)

        if key == 27:
            break

cv2.destroyAllWindows()
