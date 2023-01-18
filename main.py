import cv2
from gui_buttons import Buttons

button = Buttons()
button.add_button('person', 20, 20)
button.add_button('cell phone', 20, 100)
button.add_button('remote', 20, 180)
button.add_button('scissors', 20, 260)
button.add_button('backpack', 20, 340)


net = cv2.dnn.readNet('dnn_model/yolov4-tiny.weights', 'dnn_model/yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

classes = []
with open('dnn_model/classes.txt', 'r') as obj:
    for class_name in obj.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


def click_button(event, x, y, flags, params):
    global button_person
    if event ==cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', click_button)

while True:
    ret, frame = cap.read()
    active_buttons = button.active_buttons_list()


    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 3, (50, 20, 200), 3)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 20, 200), 3)

    button.display_buttons(frame)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()