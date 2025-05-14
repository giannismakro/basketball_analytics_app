import cv2
from yolov8 import YOLODetector
from deep_sort.tracker import DeepSORTTracker
from utils import compute_speed
import torch.serialization
from torch.nn.modules.container import Sequential, ModuleList
from ultralytics.nn.modules import Conv, C2f, Bottleneck, SPPF, Concat, Detect, DFL
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from ultralytics.nn.tasks import DetectionModel

# --- Video Setup ---
video_path = "input/sample_game.mp4"
cap = cv2.VideoCapture(video_path)

torch.serialization.add_safe_globals([
    DetectionModel,
    Sequential,
    Upsample,
    Concat,
    Conv,
    Conv2d,
    BatchNorm2d,
    Detect,
    SiLU,
    C2f,
    DFL,
    ModuleList,
    MaxPool2d,
    Bottleneck,
    SPPF
])

detector = YOLODetector()
tracker = DeepSORTTracker()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

positions = {}

# --- Create OpenCV window with trackbar ---
cv2.namedWindow("Basketball Analytics")
cv2.createTrackbar("Frame", "Basketball Analytics", 0, total_frames - 1, lambda x: None)

last_slider_pos = 0

while True:
    # Handle seeking
    slider_pos = cv2.getTrackbarPos("Frame", "Basketball Analytics")
    if abs(slider_pos - last_slider_pos) > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, slider_pos)
        last_slider_pos = slider_pos

    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks = tracker.update_tracks(detections, frame)
    h, w, _ = frame.shape

    for track in tracks:

        l, t, r, b = track.to_ltrb()

        # Clip to image size
        l = max(0, min(int(l), w - 1))
        r = max(0, min(int(r), w - 1))
        t = max(0, min(int(t), h - 1))
        b = max(0, min(int(b), h - 1))

        # Optional shrink
        bbox_w = r - l
        bbox_h = b - t
        shrink = 0.1
        l += int(bbox_w * shrink / 2)
        r -= int(bbox_w * shrink / 2)
        t += int(bbox_h * shrink / 2)
        b -= int(bbox_h * shrink / 2)

        x = (l + r) / 2
        y = (t + b) / 2

        # Draw box and label
        print(track.track_id)
        if track.track_id == "4":
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{track.track_id}', (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Speed and distance calculations
        if track.track_id in positions:
            last_pos, last_frame = positions[track.track_id]
            speed = compute_speed(last_pos, (x, y), 1 / fps)
            distance = compute_speed((0, 0), (x - last_pos[0], y - last_pos[1]), 1)
        else:
            speed = 0
            distance = 0

        positions[track.track_id] = ((x, y), cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Update slider to match current frame
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos("Frame", "Basketball Analytics", current_frame)

    cv2.imshow("Basketball Analytics", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
