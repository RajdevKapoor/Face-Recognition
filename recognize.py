import cv2


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)

        id, _ = clf.predict(gray_img[y:y+h, x:x+w])

        if id==1:
            cv2.putText(img, "Rajdev", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords


def recognize(img, clf, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color["white"], "Face", clf)
    return img



faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")


video_capture = cv2.VideoCapture(0)

while True:

    _, img = video_capture.read()

    img = recognize(img, clf, faceCascade)

    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()