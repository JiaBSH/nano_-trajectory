from ultralytics import YOLO
import cv2

#model = YOLO('yolov8n.pt')
model = YOLO(r'E:\cq_instance_post\yolo_code\runs\detect\train5\weights\best.pt')
print('kk')
img = cv2.imread(r'E:\cq_instance_post\big_data\yolo_dataset\images\val\20250107-2.jpg')
print('k')
#img = cv2.flip(img, 1)  # 0=上下翻转，1=左右翻转
print('hh')
results = model.predict(source=img, save=True,line_thickness=1 )
print('ss')
results[0].show()