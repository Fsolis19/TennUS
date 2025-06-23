from ultralytics import YOLO

model = YOLO('models_weigths/keypoint_detection/best.pt')  

results = model.predict('input_video/videoplayback.mp4', save = True)
print(results[0])

print('----------------------------------------------')

for box in results[0].boxes:
    print(box)