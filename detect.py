import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train-7/weights/best.pt")

image = cv2.imread('./test/images/dog.jpg')

results = model(image,conf = 0.8)

for result in results:
    for box in result.boxes:
        x1,y1,x2,y2 = map(int,box.xyxy[0])

        confidence = float(box.conf[0]) *100

        cv2.rectangle(image, (x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(image,f"Dog Face:{confidence:.2f}%", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))



cv2.imshow("Dog Face Detection",image)

cv2.waitKey(0)


cv2.destroyAllWindows()
                    

