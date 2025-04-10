import cv2
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ✅ Load YOLO model
model = YOLO("C:/Users/www.abcom.in/Desktop/projects/EggCounting/model/best.pt")

# ✅ Load video
video_path = "C:/Users/www.abcom.in/Desktop/projects/EggCounting/video/20180910_144521.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception("❌ Failed to open video.")

# ✅ Get video properties
ret, first_frame = cap.read()
if not ret:
    raise Exception("❌ Failed to read first frame.")

height, width = first_frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30

print(f"Resolution: {width}x{height}, FPS: {fps}")

# ✅ Output video writer
out = cv2.VideoWriter("output_egg_counter.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# ✅ Counting setup
line_y = int(height * 0.6)  # Adjust this if needed
egg_count = 0
counted_ids = set()
prev_positions = {}
frame_number = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Run object tracking
    results = model.track(source=frame, persist=True, conf=0.5, verbose=False)
    boxes = results[0].boxes
    if boxes is None:
        continue

    for box in boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        prev_cy = prev_positions.get(track_id, None)

        if prev_cy is not None:
            if prev_cy < line_y and cy >= line_y and track_id not in counted_ids:
                egg_count += 1
                counted_ids.add(track_id)

        prev_positions[track_id] = cy

        # ✅ Draw detection and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # ✅ Draw counting line and total
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
    cv2.putText(frame, f"Egg Count: {egg_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    out.write(frame)

    if frame_number % 10 == 0:
        cv2.imshow("Egg Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_number += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Done! Total eggs counted: {egg_count}")
print("🎞 Output saved as: output_egg_counter.mp4")
