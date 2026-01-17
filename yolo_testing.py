from ultralytics import YOLO


model = YOLO('models/best2.pt')

result = model.predict('vedios/08fd33_4.mp4',save=True)
print(result[0])
print('=====================================')
for box in result[0].boxes:
    print(box)