# from flask import Flask, render_template,Response
# import cv2
# import mediapipe as mp
# import numpy as np
# import pandas as pd
# import time
# import datetime
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import math
# from mpl_toolkits import mplot3d
# from celluloid import Camera
# from scipy import spatial
# import pyshine as ps

# from calculate import Average,dif_compare,diff_compare_angle,convert_data,calculateAngle
# from classify_pose import classifyPose
# from comparePose import compare_pose
# from detect_pose import detectPose
# from extract_key_points import extractKeypoint

# app = Flask(__name__)
# camera = cv2.VideoCapture(0)

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
# # cap = cv2.VideoCapture(0)
# path = "E:\\psg\\Sem 5\\yogatrainer-main\\video\\yoga25.jpg"
# x = extractKeypoint(path)
# dim = (960, 760)
# resized = cv2.resize(x[3], dim, interpolation = cv2.INTER_AREA)
# # cv2.imshow('target',resized)
# angle_target = x[2]
# point_target = x[1]

# def generate_frames():
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             # cv2.putText(frame, str("PERFECT"), (170,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

# def checker():
#     with mp_pose.Pose(min_detection_confidence =0.5, min_tracking_confidence = 0.5) as pose:

#         while camera.isOpened():
#         # while True:
#             ret,frame= camera.read()

#             if not ret:
#                 break

#             else:
#                 image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 image.flags.writeable = False
#                 results = pose.process(image)

#                 image.flags.writeable = True
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#                 image_height, image_width, _ = image.shape
#                 image = cv2.resize(image, (int(image_width * (860 / image_height)), 860))

#                 try:
#                     landmarks = results.pose_landmarks.landmark

#                     shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
#                                   round(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility*100, 2)]
#                     elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
#                                   round(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility*100, 2)]
#                     wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
#                                   round(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility*100, 2)]

#                     angle_point = []

#                     right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
#                     angle_point.append(right_elbow)

#                     left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#                     angle_point.append(left_elbow)

#                     right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#                     angle_point.append(right_shoulder)

#                     left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#                     angle_point.append(left_shoulder)

#                     right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

#                     left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

#                     right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#                     angle_point.append(right_hip)

#                     left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#                     angle_point.append(left_hip)

#                     right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
#                     angle_point.append(right_knee)

#                     left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
#                     angle_point.append(left_knee)
#                     right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

#                     left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]


#                     keypoints = []
#                     for point in landmarks:
#                         keypoints.append({
#                              'X': point.x,
#                              'Y': point.y,
#                              'Z': point.z,
#                              })


#                     p_score = dif_compare(keypoints, point_target)

#                     angle = []

#                     angle1 = calculateAngle(right_shoulder, right_elbow, right_wrist)
#                     angle.append(int(angle1))
#                     angle2 = calculateAngle(left_shoulder, left_elbow, left_wrist)
#                     angle.append(int(angle2))
#                     angle3 = calculateAngle(right_elbow, right_shoulder, right_hip)
#                     angle.append(int(angle3))
#                     angle4 = calculateAngle(left_elbow, left_shoulder, left_hip)
#                     angle.append(int(angle4))
#                     angle5 = calculateAngle(right_shoulder, right_hip, right_knee)
#                     angle.append(int(angle5))
#                     angle6 = calculateAngle(left_shoulder, left_hip, left_knee)
#                     angle.append(int(angle6))
#                     angle7 = calculateAngle(right_hip, right_knee, right_ankle)
#                     angle.append(int(angle7))
#                     angle8 = calculateAngle(left_hip, left_knee, left_ankle)
#                     angle.append(int(angle8))

#                     compare_pose(image, angle_point,angle, angle_target)
#                     a_score = diff_compare_angle(angle,angle_target)

#                     if (p_score >= a_score):
#                         cv2.putText(image, str(int((1 - a_score)*100)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)

#                     else:
#                         cv2.putText(image, str(int((1 - p_score)*100)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)

#                 except:
#                     pass

#                 mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                          mp_drawing.DrawingSpec(color = (0,0,255), thickness = 4, circle_radius = 4),
#                                          mp_drawing.DrawingSpec(color = (0,255,0),thickness = 3, circle_radius = 3)
#                                           )

#             #     cv2.imshow('MediaPipe Feed',image)

#             #     if cv2.waitKey(10) & 0xFF == ord('q'):
#             #         break

#             # cap.release()
#             # cv2.destroyAllWindows()

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

# @app.route("/")
# def home():
#     return render_template('index.html')

# @app.route("/video")
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug = True)




import time
import cv2
from flask import Flask, render_template, Response
import mediapipe as mp




app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
    previous_time = 0
    # creating our model to draw landmarks
    mpDraw = mp.solutions.drawing_utils
    # creating our model to detected our pose
    my_pose = mp.solutions.pose
    pose = my_pose.Pose()

    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        # converting image to RGB from BGR cuz mediapipe only work on RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)
        # print(result.pose_landmarks)
        if result.pose_landmarks:
            mpDraw.draw_landmarks(img, result.pose_landmarks, my_pose.POSE_CONNECTIONS)

        # checking video frame rate
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # Writing FrameRate on video
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        #cv2.imshow("Pose detection", img)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__=="__main__":
    app.run(debug=True)



# > set FLASK_APP=hello.py
# > set FLASK_ENV=development
# > py -m flask run
