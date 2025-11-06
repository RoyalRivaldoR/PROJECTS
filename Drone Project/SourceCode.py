import cv2
import pandas as pd
import time
from ultralytics import YOLO
import numpy as np
import sys
import telepot
from subprocess import call
import cv2
import datetime as dt


try:
    # Load YOLOv8 model for object detection
    model = YOLO('yolov8n.pt', task='segment')
    name = model.names
except Exception as e:
    print("Error loading YOLO model:", e)
    sys.exit(1)

try:
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam")
        sys.exit(1)
except Exception as e:
    print("Error opening webcam:", e)
    sys.exit(1)

time_list = [0, 0, 0, 0]
start_time = time.time()



cap.set(3,480)
cap.set(4,480)

def handle(msg):
    
    print("")



chat_id2 = 234352074
chat_id = -4191992310
def capture_and_send_video():
    bot.sendMessage(chat_id, text="Camera is starting to record")
    time.sleep(0.1)
    bot.sendMessage(chat_id, text="Hold on please for 10 seconds")

    capture_duration = 10

    dim = (480,480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video.mp4', fourcc, 24.0, dim)
    start_time = time.time()
    while int(time.time() - start_time) < capture_duration:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            out.write(resized)
        else:
            bot.sendMessage(chat_id, text="Recording failed")
            break
        cv2.imshow("Img", frame)
        cv2.waitKey(1)
    out.release()
    

    bot.sendMessage(chat_id, text="Recording completed")
    time.sleep(0.1)
    bot.sendMessage(chat_id, text="Uploading video, please be patient")
    time.sleep(0.1)
    bot.sendVideo(chat_id, video=open('./video.mp4', 'rb'))
    bot.sendVideo(chat_id2, video=open('./video.mp4', 'rb'))

bot = telepot.Bot('6990828103:AAFDKW8dlbu-7r6yHQAl4Wn5z31IOUI1t7o')
bot.message_loop(handle)

print('Hello')

bot.sendMessage(chat_id, text="Hello")
bot.sendMessage(chat_id2, text="Hello")
i = 0

video_send=0

while True:
    try:
        this_frame_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam")
            continue

        if i % 5 == 0 or i == 0:
            results = model.predict(frame, device='mps')
            result = results[0]
            a = pd.DataFrame(results[0].boxes.data.detach().cpu().numpy())
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
            classes = np.array(result.boxes.cls.cpu(), dtype='int')

            for cls, bbox, confidence in zip(classes, bboxes, a[4]):
                confidence = round((confidence * 100) / 10, 0) * 10
                if name[int(cls)] == "person":  # Check if the detected class is "person"
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (250, 250, 256))
                    cv2.putText(frame, name[int(cls)] + "  " + str(confidence) + '%', (x1, y2 - 5), cv2.FONT_HERSHEY_PLAIN, 2,
                                (250, 250, 250), 2)
                    if(video_send==0):
                        capture_and_send_video()
                        video_send+=1
                        break
                    
                        
                    if(video_send>=100):
                        video_send=0
                    else:
                        video_send+=1
                    #break  # Break the loop if person is detected and video captured

        cv2.imshow("Tracking", frame)

        i += 1

        if cv2.waitKey(1) == 27:
            break

    except Exception as e:
        print("Error:", e)

# Release resources
cap.release()
cv2.destroyAllWindows()
