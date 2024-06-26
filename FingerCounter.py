import cv2
import time
import os
import HandTracingModule as yo
import subprocess

wCam, hCam = 640, 480

# # Get the screen resolution
# try:
#     screen_res = subprocess.check_output('xrandr').decode('utf-8')
#     print("Screen resolution output:", screen_res)
#     screen_res = screen_res.split("current")[1].split(",")[0].strip().split(" ")[0]
#     screen_width, screen_height = map(int, screen_res.split("x"))
# except Exception as e:
#     print("Error:", e)
#     screen_width, screen_height = 1920, 1080  # Default resolution

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
myList = sorted(myList, key=lambda x: int(x.split('.')[0]))
print(myList)
overlayList = []

pTime = 0

detector = yo.handDetector(detectionCon=0.75)

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    image = cv2.resize(image, (200, 200))
    overlayList.append(image)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    handList = detector.findPosition(img, draw=False)

    if len(handList) != 0:
        fingers = []

        #for left hand
        if handList[tipIds[0]][1] < handList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #for right hand
        # if handList[tipIds[0]][1] > handList[tipIds[0] - 1][1]:
        #     fingers.append(1)
        # else:
        #     fingers.append(0)

        #4 Fingers
        for id in range(1, 5):
            if handList[tipIds[id]][2] < handList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]

        cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (48, 390), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (420, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Resize the window to full screen
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
