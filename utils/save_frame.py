import os
import cv2

filename = "../texture_video.avi"
output_dir = "../texture/input"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(filename)
retval = cap.isOpened()
print(retval)
frame_index = 0
save_index = 0

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        break
    if frame_index %10 == 0:
        frame = cv2.resize(frame, (512, 512))
        save_path = os.path.join(output_dir, f"input_{save_index}.png")
        cv2.imwrite(save_path, frame)
        print(f"Save: {save_path}")
        save_index += 1
    frame_index += 1