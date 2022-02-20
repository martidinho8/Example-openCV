import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    success, img = cap.read()
    cv.imshow("Video", img)
    if cv.waitKey(1) & 0xFF ==ord("q"):
        break



# print("Package imported")

# img = cv.imread("/home/martijn/Documenten/develop/object_tracking/Donald_trump.jpg")

# cv.imshow("Donald trump", img)
# cv.waitKey(0)
