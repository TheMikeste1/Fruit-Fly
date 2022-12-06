import cv2


def find_apple(img):
    apple_cascade = cv2.CascadeClassifier("./models/cascade/curated_set_cascade.xml")
    apples = apple_cascade.detectMultiScale(img)
    apple_centers = []
    apple_areas = []

    for (x, y, w, h) in apples:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # get centerx and centery values
        cx = x + w // 2
        cy = y + h // 2
        # calculate the total area
        area = w * h
        # draw a circle where the center is
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        # append these values to the lists we created above
        apple_centers.append([cx, cy])
        apple_areas.append(area)
    # get the index of the max area detected over coming frame
    # so that focus on the bigger face if there are multiple faces
    if apple_areas:
        i = apple_areas.index(max(apple_areas))
        return img, [apple_centers[i], apple_areas[i]]
    else:
        return img, [[0, 0], 0]


cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    img, info = find_apple(img)
    print("Center:", info[0], "Area:", info[1])

    cv2.imshow("Output", img)
    # sleep the while loop for 1 milisecond
    cv2.waitKey(1)
