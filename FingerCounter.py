import cv2
import time
import os
import HandTracingModule as htm

# Set the camera dimensions
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)  # Changed to 0 for default camera
cap.set(3, wCam)
cap.set(4, hCam)

# Folder path where finger images are stored
folderPath = "FingerImages"
myList = os.listdir(folderPath)
print("Loaded images:", myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print("Number of overlay images loaded:", len(overlayList))

# Initialize variables
pTime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:  # Compare x-coordinates for thumb
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:  # Compare y-coordinates for other fingers
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        print("Fingers array:", fingers)
        print("Total fingers:", totalFingers)

        if 0 <= totalFingers <= len(overlayList):
            overlayImg = overlayList[totalFingers - 1]
            h, w, c = overlayImg.shape
            if h > hCam or w > wCam:
                scale = min(hCam / h, wCam / w)
                newH, newW = int(h * scale), int(w * scale)
                overlayImg = cv2.resize(overlayImg, (newW, newH))
                h, w, c = overlayImg.shape
            img[0:h, 0:w] = overlayImg

        # Display the total fingers count
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)
