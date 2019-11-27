# Emotion Tracker

A repo for gathering and storing facial expression data (face provided by yours truly).  I've only been collecting this data while working on stuff related to my job as an instructor at Thinkful (I've, of course, not been perfect at always remembering to start data collection while working).

## Data Gathering

* The data is gathered by [`emotion_tracker.py`](emotion_tracker.py)
* That file has a class (`EmotionTracker`) that takes a photo of my face every `n` seconds (currently 15)
* The collected photo is then run through a `keras` model to classify my facial expression as being one of the following emotions: `['angry', 'scared', 'happy', 'sad', 'surprised', 'neutral']`
* This data is then saved in the format as seen/described below.

## Data

* Currently data is stored in CSVs in the [`data`](data) directory.
  * The file names are timestamps representing when the data was gathered.
  * Eventual plans to move to a better database, but for now... ¯\\_(ツ)_/¯
* The file format is shown in the below table.
  * `timestamp` is pretty self-explanatory
  * The remaining columns are emotions and their values represent the model's confidence that my face was showing that emotion. (e.g. in the first row the model was most confident my face was sad)
  * Sometimes my face wasn't in front of the camera, when that happens the timestamp is present but the emotion columns are empty.
* The gaps between data being gathered is now set to 15 seconds.  This has been changed before, and it is subject to change again.

| timestamp           | angry    | scared    | happy      | sad      | surprised  |
| ------------------- | -------- | --------- | ---------- | -------- | ---------- |
| 2019-11-13 15:03:05 | 0.353169 | 0.1668002 | 0.00546338 | 0.408096 | 0.01083975 |
| 2019-11-13 15:04:20 |          |           |            |          |            |

