from ultralytics import YOLO
import cv2
import os

model = YOLO('models_weigths/segmentation/best.pt')
model2 = YOLO('models_weigths/keypoint_detection/best.pt')

os.makedirs('output_video/segmentation_keypoint_detection', exist_ok=True)
output_path = 'output_video/segmentation_keypoint_detection/models_combination_good.mp4'

cap = cv2.VideoCapture('input_video/videoplayback.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result_seg = model.predict(frame, verbose=False)[0]
    result_kp = model2.predict(frame, verbose=False)[0]

    seg_vis = result_seg.plot(labels=False, boxes=False, line_width=15)
    kp_vis  = result_kp.plot(labels=False, boxes=False, line_width=15)

    frame_combined = cv2.addWeighted(seg_vis, 0.7, kp_vis, 0.7, 0)

    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

    out.write(frame_combined)

cap.release()
out.release()
print(f"Video final guardado en: {output_path}")
