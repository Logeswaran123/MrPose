# :weight_lifting_man: Mr.Pose :weight_lifting_woman:

<p>
Hi human, <br />
I'm Mr.Pose! :robot: :wave: <br />
I help to estimate and measure exercises for humans. Lets do some exercise and keep the body fit. :muscle: :mechanical_arm:
</p>

## Description :scroll:

Exercises, in general, helps to maintain a physically fit body, loosen up joints, and relax muscles.

Mr.Pose is a visual analytics application that helps humans to track the accuracy of exercise, and count reps. <br />

Mr.Pose can support <br />
* Pushup
* Plank
* Squat
* JumpingJack

For Pushup, Plank and JumpingJack, Mr.Pose will track each repetition provided that the exercise is accurately performed.
For Plank, Mr.Pose will track the time of plank position.

## General Requirements :mage_man:
1. Record a video of person doing a supported exercise.
2. Place the camera horizontally facing the person performing exercise (preferably at eye level).
3. Higher the resolution of video, higher the quality of estimation and measurement.

## Code Requirements :mage_woman:
** TODO **

## How to run :running_man:
```python
python mrpose.py --video <path to video file> --exercise <exercise to be measured>
```
Note:<br />

*<path to video file\>* - Path to the input video file with supported exercise<br />
*<exercise to be measured\>* - Exercise in input video file<br />

**Optional Argument:**<br />
*--exercise* - Choices are pushup, plank, squat, jumpingjack <br />
If argument is not provided, then Mr.Pose will **predict** the exercise done in the video. If argument is provided, then Mr.Pose will measure the exercise mentioned.

## Results :bar_chart:

<p align="center"> :star: <b> Pushups </b> :star: </p>

```python
python mrpose.py --video <path to video file> --exercise pushup
```

<br />

https://user-images.githubusercontent.com/36563521/188864300-bc43d096-c98f-48e4-b9cb-6bef6937f1ca.mp4

Input video source [here](https://www.pexels.com/video/woman-doing-push-ups-8472764/).
---

<p align="center"> :star: <b> Squats </b> :star: </p>

```python
python mrpose.py --video <path to video file> --exercise squat
```

<br />

https://user-images.githubusercontent.com/36563521/188866384-a19d3bb7-d6d2-47be-a71b-d27166dea395.mp4

Input video source [here](https://www.pexels.com/video/woman-exercising-while-wearing-a-face-mask-4265287/).
---

<p align="center"> :star: <b> Plank </b> :star: </p>

```python
python mrpose.py --video <path to video file> --exercise plank
```

<br />

https://user-images.githubusercontent.com/36563521/188879832-825cfd00-cfd3-4b9c-9d99-b73c4a06b8a9.mp4

Input video source [here](https://www.pexels.com/video/female-doing-planks-by-the-balcony-6152665/).
---

## References :page_facing_up:

* [Mediapipe by Google](https://github.com/google/mediapipe).
* [GymLytics](https://github.com/akshaybahadur21/GymLytics).

Happy Learning! 😄
