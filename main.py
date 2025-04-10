import cv2
from ultralytics import YOLO
import os

# ✅ Load the trained model
model = YOLO("C:/Users/www.abcom.in/Desktop/projects/EggCounting/model/best.pt")  # Update path if needed

# ✅ Video input and output setup
video_path = "C:/Users/www.abcom.in/Desktop/projects/EggCounting/video/20180910_144521.mp4"  # Your input video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception("❌ Failed to open video.")

# Read first frame to set size
ret, first_frame = cap.read()
if not ret:
    raise Exception("❌ Couldn't read the first frame.")

first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
height, width = first_frame.shape[:2]

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30

# Output video writer
out = cv2.VideoWriter("output_egg_counter.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Reset to frame 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Line and counting setup
line_y = int(height * 0.6)
egg_count = 0
counted_ids = set()
prev_positions = {}
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate if needed
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Run tracking
    results = model.track(source=frame, persist=True, conf=0.5, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        prev_cy = prev_positions.get(track_id, None)

        if track_id not in counted_ids:
            if prev_cy is not None:
                if prev_cy > line_y and cy <= line_y:
                    egg_count += 1
                    counted_ids.add(track_id)

        prev_positions[track_id] = cy

        # Draw box, ID, center
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Draw line and count
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
    cv2.putText(frame, f"Egg Count: {egg_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    # Show frame live
    cv2.imshow("Egg Counter", frame)

    # Press Q to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    out.write(frame)
    frame_number += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Done! Total eggs counted: {egg_count}")
print("🎞 Output video saved as: output_egg_counter.mp4")
