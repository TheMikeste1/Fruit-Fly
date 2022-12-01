import cv2


def findFace(img):
    face_cascade = cv2.CascadeClassifier("./models/cascade/curated_set_cascade.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # get centerx and centery values (center of the detected face)
        cx = x + w // 2
        cy = y + h // 2
        # calculate the total area
        area = w * h
        # draw a circle where the center is
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        # append these values to the lists we created above
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
    # get the index of the max area detected over coming frame
    # so that focus on the bigger face if there are multiple faces
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    img, info = findFace(img)
    print("Center:", info[0], "Area:", info[1])

    cv2.imshow("Output", img)
    # sleep the while loop for 1 milisecond
    cv2.waitKey(1)
