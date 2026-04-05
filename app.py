# this code has streamlit, kakfa, and postgres dependencies. Make sure to install them before running:
import cv2
import numpy as np
from ultralytics import YOLO
import json
import datetime
import streamlit as st
import psycopg2
# --- DATABASE CONFIGURATION ---
DB_CONFIG = {
    "dbname": "postgres",        # Must match your screenshot
    "user": "postgres",          # Default username
    "password": "1234",  # Your actual database password
    "host": "localhost",
    "port": "5432"
}

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(page_title="Mining Fleet UI", layout="wide")
st.title("🚜 Production Mining Tracker & Utilization Dashboard")

# Create two columns: Left for Video (larger), Right for Dashboard (smaller)
col1, col2 = st.columns([2, 1])
video_placeholder = col1.empty()

with col2:
    st.subheader("Live Utilization Dashboard")
    metric_activity = st.empty()
    metric_loads = st.empty()
    metric_utilization = st.empty()
    metric_active_time = st.empty()
    metric_idle_time = st.empty()
    st.markdown("---")
    st.subheader("Live Kafka Payload")
    json_placeholder = st.empty()

# --- CONFIGURATION ---
MODEL_V3_PATH = r"C:\Users\Yousef Ibrahim\Downloads\Equipment Utilization & Activity Classification\runs\segment\excavator_v3_master\weights\best.pt"
MODEL_COCO_PATH = "yolov8s.pt"
VIDEO_PATH = r"C:\Users\Yousef Ibrahim\Downloads\old Liebherr R9350 Excavator loads a Caterpillar 777e truck at full power _ Mini.mp4"
CONF_V3 = 0.15
CONF_COCO = 0.15

# --- HELPER FUNCTIONS & CLASSES ---


def boxes_overlap(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])


def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def get_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


class GlobalTracker:
    def __init__(self, max_age=45, max_dist=150):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.max_dist = max_dist

    def update(self, current_boxes):
        matched_ids = set()
        for box in current_boxes:
            cx, cy = get_center(box)
            best_id, min_dist = None, float('inf')
            for tid, trk in self.tracks.items():
                if tid in matched_ids:
                    continue
                dist = get_distance((cx, cy), trk['center'])
                if boxes_overlap(box, trk['box']) or dist < self.max_dist:
                    if dist < min_dist:
                        min_dist, best_id = dist, tid
            if best_id is not None:
                self.tracks[best_id]['box'], self.tracks[best_id]['center'], self.tracks[best_id]['age'] = box, (
                    cx, cy), 0
                matched_ids.add(best_id)
            else:
                self.tracks[self.next_id] = {'box': box, 'center': (
                    cx, cy), 'age': 0, 'status': 'WAITING'}
                matched_ids.add(self.next_id)
                self.next_id += 1
        to_delete = [tid for tid in self.tracks if tid not in matched_ids and (self.tracks[tid].__setitem__(
            'age', self.tracks[tid]['age'] + 1) or self.tracks[tid]['age'] > self.max_age)]
        for tid in to_delete:
            del self.tracks[tid]
        return self.tracks


# Load Models
model_v3 = YOLO(MODEL_V3_PATH)
model_coco = YOLO(MODEL_COCO_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0

truck_tracker = GlobalTracker(max_age=60, max_dist=150)
excavator_tracker = GlobalTracker(max_age=9000, max_dist=400)

# --- PAYLOAD & STATE VARIABLES ---
total_loads = 0
is_dumping = False
dumping_frames = 0
REQUIRED_DUMP_FRAMES = 1
last_known_bucket = None
bucket_missing_frames = 0
load_cooldown_timer = 0
COOLDOWN_MAX = 90
current_activity = "IDLE"
idle_timer = 0
IDLE_THRESHOLD = 300
frame_count = 0
total_active_frames = 0
total_idle_frames = 0

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    annotated_frame = frame.copy()
    frame_area = frame.shape[0] * frame.shape[1]
    frame_height = frame.shape[0]

    raw_excavator_boxes, raw_truck_boxes = [], []
    bucket_center = None
    if load_cooldown_timer > 0:
        load_cooldown_timer -= 1

    # 1. RUN V3 SPECIALIST
    res_v3 = model_v3.predict(frame, conf=CONF_V3, verbose=False)
    if res_v3[0].boxes is not None:
        v3_boxes, v3_clss = res_v3[0].boxes.xyxy.cpu(
        ).numpy(), res_v3[0].boxes.cls.cpu().numpy()
        for box, cls in zip(v3_boxes, v3_clss):
            x1, y1, x2, y2 = [int(x) for x in box]
            if int(cls) == 1 and ((x2 - x1) * (y2 - y1)) > (frame_area * 0.03):
                raw_excavator_boxes.append([x1, y1, x2, y2])
            elif int(cls) == 2:
                raw_truck_boxes.append([x1, y1, x2, y2])

        bucket_found_this_frame = False
        if res_v3[0].masks is not None:
            for i, cls in enumerate(v3_clss):
                if int(cls) == 0:
                    mask = cv2.resize(res_v3[0].masks.data[i].cpu(
                    ).numpy(), (frame.shape[1], frame.shape[0]))
                    M = cv2.moments(mask)
                    if M["m00"] != 0:
                        bucket_center = (
                            int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                        last_known_bucket, bucket_missing_frames, bucket_found_this_frame = bucket_center, 0, True
                        cv2.circle(annotated_frame, bucket_center,
                                   8, (0, 0, 255), -1)
                        cv2.putText(
                            annotated_frame, "BUCKET", (bucket_center[0]+10, bucket_center[1]), 0, 0.6, (0, 0, 255), 2)
                    break

        # THE FIX: Ghost memory strictly set to 15 frames for instant snapping
        if not bucket_found_this_frame and last_known_bucket is not None:
            bucket_missing_frames += 1
            if bucket_missing_frames < 15:
                bucket_center = last_known_bucket
                cv2.circle(annotated_frame, bucket_center,
                           8, (0, 165, 255), -1)
            else:
                last_known_bucket = None

    # 2. EXCAVATOR TRACKER
    glued_excavator_boxes = []
    for raw_box in raw_excavator_boxes:
        merged = False
        for i, g_box in enumerate(glued_excavator_boxes):
            if boxes_overlap(raw_box, g_box) or get_distance(get_center(raw_box), get_center(g_box)) < 400:
                glued_excavator_boxes[i] = [min(raw_box[0], g_box[0]), min(
                    raw_box[1], g_box[1]), max(raw_box[2], g_box[2]), max(raw_box[3], g_box[3])]
                merged = True
                break
        if not merged:
            glued_excavator_boxes.append(raw_box)

    active_excavators = excavator_tracker.update(glued_excavator_boxes)
    current_excavator_zones, primary_excavator_id = [], 1
    for ex_id, ex_data in active_excavators.items():
        primary_excavator_id = ex_id
        ex1, ey1, ex2, ey2 = ex_data['box']
        current_excavator_zones.append(ex_data['box'])
        cv2.rectangle(annotated_frame, (ex1, ey1), (ex2, ey2), (255, 0, 0), 2)
        cv2.rectangle(annotated_frame, (ex1, ey1-35),
                      (ex1+200, ey1), (255, 0, 0), -1)
        cv2.putText(
            annotated_frame, f"EXCAVATOR {ex_id}", (ex1+5, ey1-5), 0, 0.7, (255, 255, 255), 2)

    # 3. RUN COCO TRUCKS
    res_coco = model_coco.predict(
        frame, classes=[7], conf=CONF_COCO, verbose=False)
    if res_coco[0].boxes is not None:
        for box in res_coco[0].boxes.xyxy.cpu().numpy():
            cx1, cy1, cx2, cy2 = [int(x) for x in box]
            coco_rect = [cx1, cy1, cx2, cy2]
            box_area = (cx2 - cx1) * (cy2 - cy1)
            if box_area < (frame_area * 0.001) or box_area > (frame_area * 0.05):
                continue

            is_hallucination = any(boxes_overlap(coco_rect, ex_zone) for ex_zone in current_excavator_zones) or any(
                boxes_overlap(coco_rect, v3_box) for v3_box in raw_truck_boxes)
            if not is_hallucination:
                raw_truck_boxes.append(coco_rect)

    # 4. TRUCK TRACKER
    glued_truck_boxes = []
    for raw_box in raw_truck_boxes:
        merged = False
        for i, g_box in enumerate(glued_truck_boxes):
            if boxes_overlap(raw_box, g_box) and get_distance(get_center(raw_box), get_center(g_box)) < 200:
                glued_truck_boxes[i] = [min(raw_box[0], g_box[0]), min(
                    raw_box[1], g_box[1]), max(raw_box[2], g_box[2]), max(raw_box[3], g_box[3])]
                merged = True
                break
        if not merged:
            glued_truck_boxes.append(raw_box)

    active_global_tracks = truck_tracker.update(glued_truck_boxes)
    active_truck_box = None
    for track_id, track_data in active_global_tracks.items():
        tx1, ty1, tx2, ty2 = track_data['box']
        trk_rect = [tx1, ty1, tx2, ty2]
        is_active_zone = ((tx2 - tx1) * (ty2 - ty1)) > (frame_area * 0.05)

        if not is_active_zone:
            for ex1, ey1, ex2, ey2 in current_excavator_zones:
                if boxes_overlap(trk_rect, [ex1 - 250, ey1 - 150, ex2 + 250, ey2 + 150]):
                    is_active_zone = True
                    break

        track_data['status'] = 'ACTIVE' if is_active_zone else 'WAITING'
        if is_active_zone and track_data['age'] == 0:
            active_truck_box = track_data['box']

        color = (0, 255, 0) if is_active_zone else (0, 255, 255)
        if track_data['age'] > 0:
            color = (0, 100, 0) if is_active_zone else (0, 100, 100)

        cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2),
                      color, 3 if is_active_zone else 2)
        cv2.rectangle(annotated_frame, (tx1, ty1-35),
                      (tx1+200, ty1), color, -1)
        cv2.putText(annotated_frame, f"{track_data['status']} TRUCK {track_id}", (
            tx1+5, ty1-5), 0, 0.6, (0, 0, 0), 2)

    # 5. PAYLOAD MATH
    activity_this_frame = None
    if bucket_center:
        bx, by = bucket_center
        if active_truck_box:
            tx1, ty1, tx2, ty2 = active_truck_box

            # The tight box you requested!
            dump_top = ty1 - 100
            dump_left = tx1 - 100
            dump_right = tx2 + 100

            if dump_left < bx < dump_right and dump_top < by < ty2:
                dumping_frames += 1
                if dumping_frames >= REQUIRED_DUMP_FRAMES:
                    is_dumping, activity_this_frame = True, "DUMPING PAYLOAD"
                else:
                    activity_this_frame = "SWINGING"
            else:
                activity_this_frame = "DIGGING" if by > (
                    ty1 + 50) else "SWINGING"
                if is_dumping:
                    if load_cooldown_timer == 0:
                        total_loads += 1
                        load_cooldown_timer = COOLDOWN_MAX
                    is_dumping = False
                dumping_frames = 0
        else:
            activity_this_frame = "DIGGING" if by > (
                frame_height * 0.6) else "SWINGING"
    else:
        if is_dumping:
            if load_cooldown_timer == 0:
                total_loads += 1
                load_cooldown_timer = COOLDOWN_MAX
            is_dumping = False
        dumping_frames = 0

    if activity_this_frame is not None:
        current_activity, idle_timer = activity_this_frame, 0
    else:
        idle_timer += 1
        if idle_timer > IDLE_THRESHOLD:
            current_activity = "IDLE"

    # 6. JSON TELEMETRY & STREAMLIT UPDATES
    if current_activity == "IDLE":
        total_idle_frames += 1
        status_state = "IDLE"
    else:
        total_active_frames += 1
        status_state = "ACTIVE"

    # --- PUSH TO STREAMLIT UI EVERY FRAME ---
    # Convert BGR (OpenCV) to RGB (Streamlit) so the colors don't look weird
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB",
                            use_container_width=True)

    if frame_count % int(fps) == 0:
        total_seconds = frame_count / fps
        active_seconds = total_active_frames / fps
        idle_seconds = total_idle_frames / fps
        utilization_percent = (total_active_frames /
                               frame_count) * 100 if frame_count > 0 else 0.0

        # Update Sidebar Metrics dynamically!
        metric_activity.metric("Current Activity", current_activity)
        metric_loads.metric("Total Loads Extracted", total_loads)
        metric_utilization.metric(
            "Utilization %", f"{utilization_percent:.1f}%")
        metric_active_time.metric("Working Time", f"{active_seconds:.1f} sec")
        metric_idle_time.metric("Idle Time", f"{idle_seconds:.1f} sec")

        payload = {
            "frame_id": frame_count,
            "equipment_id": f"EX-{primary_excavator_id:03d}",
            "equipment_class": "excavator",
            "timestamp": str(datetime.timedelta(seconds=int(total_seconds))) + f".{int((total_seconds % 1) * 1000):03d}",
            "utilization": {
                "current_state": status_state,
                "current_activity": current_activity,
                "motion_source": "arm_only"
            },
            "time_analytics": {
                "total_tracked_seconds": round(total_seconds, 1),
                "total_active_seconds": round(active_seconds, 1),
                "total_idle_seconds": round(idle_seconds, 1),
                "utilization_percent": round(utilization_percent, 1)
            }
        }

        # Display JSON live on the dashboard
        json_placeholder.json(payload)
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            insert_query = """
                INSERT INTO equipment_utilization (
                    frame_id, equipment_id, equipment_class, timestamp, 
                    current_state, current_activity, motion_source, 
                    total_tracked_seconds, total_active_seconds, total_idle_seconds, utilization_percent
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            data = (
                payload["frame_id"], payload["equipment_id"], payload["equipment_class"], payload["timestamp"],
                payload["utilization"]["current_state"], payload["utilization"]["current_activity"], payload["utilization"]["motion_source"],
                payload["time_analytics"]["total_tracked_seconds"], payload["time_analytics"]["total_active_seconds"],
                payload["time_analytics"]["total_idle_seconds"], payload["time_analytics"]["utilization_percent"]
            )
            cursor.execute(insert_query, data)
            conn.commit()
            cursor.close()
            conn.close()
            print(f"✅ DB STREAM SUCCESS: Frame {frame_count} inserted.")
        except Exception as e:
            print(
                f"❌ DB CONNECTION ERROR: Make sure PostgreSQL is running and your password is correct! Error: {e}")

cap.release()
