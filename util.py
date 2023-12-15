import cv2
from ultralytics import YOLO


def run_detection(model: YOLO):
  cap = cv2.VideoCapture(0)
  while True:
    if cv2.waitKey(1) == ord('q'):
        break
    _, image = cap.read()
    for result in model.predict(image, True):
      cv2.imshow('Results', result.plot())

  # Terminate
  cap.release()
  cv2.destroyAllWindows()
