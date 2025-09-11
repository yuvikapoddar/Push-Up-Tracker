# import time
# from dataclasses import dataclass
# from pathlib import Path

# import cv2
# import mediapipe as mp
# import numpy as np
# import pandas as pd


# # ---------- Geometry helpers ----------
# def angle(a, b, c):
#     """Angle at point b (in degrees) given three (x,y) points."""
#     a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
#     ba, bc = a - b, c - b
#     den = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
#     cosang = np.clip(np.dot(ba, bc) / den, -1.0, 1.0)
#     return float(np.degrees(np.arccos(cosang)))


# @dataclass
# class RepStats:
#     start_frame: int
#     min_angle: float = 999.0
#     max_angle: float = 0.0
#     ecc_time_s: float = 0.0  # lowering (when stage == "up")
#     con_time_s: float = 0.0  # rising (when stage == "down")


# class PushupCounter:
#     """
#     Simple state machine:
#       stage == "up"   : elbows near lockout (angle high) → count eccentric time
#       stage == "down" : elbows bent (angle low)          → count concentric time
#     A rep is completed on transition down -> up.
#     """

#     def __init__(self, up_thr=160, down_thr=90):
#         self.up_thr = up_thr
#         self.down_thr = down_thr
#         self.stage = None            # "up" or "down"
#         self.reps = 0
#         self.curr: RepStats | None = None
#         self.last_time = None
#         self.completed: list[dict] = []

#     def update(self, ang: float, frame_idx: int, now: float):
#         if self.last_time is None:
#             self.last_time = now
#         dt = max(0.0, now - self.last_time)
#         self.last_time = now

#         if self.curr is None:
#             self.curr = RepStats(frame_idx)

#         # Track min/max and phase durations
#         self.curr.min_angle = min(self.curr.min_angle, ang)
#         self.curr.max_angle = max(self.curr.max_angle, ang)
#         if self.stage == "up":
#             self.curr.ecc_time_s += dt
#         elif self.stage == "down":
#             self.curr.con_time_s += dt

#         rep_out = None

#         # Initialize stage on first frames
#         if self.stage is None:
#             self.stage = "up" if ang >= self.up_thr else "down"

#         # Transitions
#         if ang <= self.down_thr and self.stage == "up":
#             self.stage = "down"
#         elif ang >= self.up_thr and self.stage == "down":
#             self.stage = "up"
#             self.reps += 1
#             rep_out = self.finalize(frame_idx)

#         return self.reps, self.stage, rep_out

#     def finalize(self, frame_idx: int):
#         span = 115.0  # rough elbow ROM span for push-ups
#         rom_pct = (self.curr.max_angle - self.curr.min_angle) / span * 100.0

#         rep = {
#             "rep_index": self.reps,
#             "min_angle": round(self.curr.min_angle, 1),
#             "max_angle": round(self.curr.max_angle, 1),
#             "rom_pct": round(rom_pct, 1),
#             "ecc_time_s": round(self.curr.ecc_time_s, 2),
#             "con_time_s": round(self.curr.con_time_s, 2),
#             "rep_time_s": round(self.curr.ecc_time_s + self.curr.con_time_s, 2),
#         }
#         self.completed.append(rep)
#         # start a new rep window from current frame
#         self.curr = RepStats(frame_idx)
#         return rep


# # ---------- MediaPipe helpers ----------
# mp_pose = mp.solutions.pose
# mp_draw = mp.solutions.drawing_utils


# def _xy(lms, w, h, idx):
#     p = lms[idx]
#     return int(p.x * w), int(p.y * h)


# def elbow_angle_min_side(lms, w, h):
#     # Left arm
#     LSH, LEL, LWR = _xy(lms, w, h, 11), _xy(lms, w, h, 13), _xy(lms, w, h, 15)
#     # Right arm
#     RSH, REL, RWR = _xy(lms, w, h, 12), _xy(lms, w, h, 14), _xy(lms, w, h, 16)
#     return min(angle(LSH, LEL, LWR), angle(RSH, REL, RWR))


# # ---------- Main entry (used by server.py) ----------
# def run(input_path, use_webcam, out_dir: Path, speak: bool = False, frame_csv: bool = False, no_window: bool = True):
#     """
#     Processes a video/webcam stream and writes:
#       - annotated video to <out_dir>/annotated.mp4
#       - per_rep.csv (always if any reps)
#       - per_frame.csv (when frame_csv=True)
#     """
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_video_path = out_dir / "annotated.mp4"

#     # Open capture
#     cap = cv2.VideoCapture(0 if use_webcam else str(input_path))
#     if not cap.isOpened():
#         raise RuntimeError(f"Could not open input: {input_path}")

#     # Read one frame to get size reliably
#     ok, first = cap.read()
#     if not ok:
#         cap.release()
#         raise RuntimeError("No frames in the video")

#     h, w = first.shape[:2]
#     # Try to get FPS; if 0, use a sane default
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if not fps or fps <= 1e-3:
#         fps = 25.0

#     # H.264 writer (browser-friendly)
#     fourcc = cv2.VideoWriter_fourcc(*"avc1")
#     out_writer = cv2.VideoWriter(str(out_video_path), fourcc, float(fps), (w, h))
#     if not out_writer.isOpened():
#         cap.release()
#         raise RuntimeError("VideoWriter failed to open. H.264 may be unavailable.")

#     counter = PushupCounter()
#     per_frame, per_rep = [], []

#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         frame_idx = 0

#         # process first frame (we already read it)
#         frames_iter = [(True, first)]
#         # then the rest
#         def rest():
#             while True:
#                 ok2, fr = cap.read()
#                 if not ok2:
#                     break
#                 yield (ok2, fr)

#         for ok, frame in list(frames_iter) + list(rest()):
#             if not ok:
#                 break

#             # mirror for user experience
#             frame = cv2.flip(frame, 1)
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             res = pose.process(rgb)

#             ang = np.nan
#             if res.pose_landmarks:
#                 ang = elbow_angle_min_side(res.pose_landmarks.landmark, w, h)

#             reps, stage, done = counter.update(float(ang) if not np.isnan(ang) else 0.0, frame_idx, time.time())

#             # draw overlays
#             vis = frame.copy()
#             if res.pose_landmarks:
#                 mp_draw.draw_landmarks(vis, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#             cv2.putText(vis, f"Reps: {reps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             if not np.isnan(ang):
#                 cv2.putText(vis, f"Elbow: {int(round(ang))}°", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#             if stage:
#                 cv2.putText(vis, f"Stage: {stage}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#             # write frame
#             out_writer.write(vis)

#             # logs
#             if frame_csv:
#                 per_frame.append({"frame": frame_idx, "angle": float(ang) if not np.isnan(ang) else None,
#                                   "reps": reps, "stage": stage})
#             if done:
#                 per_rep.append(done)

#             # optional preview window for local testing
#             if not no_window:
#                 cv2.imshow("Push-Up Counter", vis)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#             frame_idx += 1

#     cap.release()
#     out_writer.release()
#     cv2.destroyAllWindows()

#     # write CSVs
#     if frame_csv and len(per_frame):
#         pd.DataFrame(per_frame).to_csv(out_dir / "per_frame.csv", index=False)
#     if len(per_rep):
#         pd.DataFrame(per_rep).to_csv(out_dir / "per_rep.csv", index=False)



import cv2, mediapipe as mp, numpy as np, time, pandas as pd, pyttsx3
from pathlib import Path
from dataclasses import dataclass

# ---- Text-to-Speech Setup ----
engine = pyttsx3.init()
engine.setProperty("rate", 160)   # speed
engine.setProperty("volume", 1.0) # full volume

def speak(msg):
    try:
        engine.say(msg)
        engine.runAndWait()
    except:
        pass

# ---- Angle Calculation ----
def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

@dataclass
class RepStats:
    start_frame: int
    min_angle: float = 999.0
    max_angle: float = 0.0
    ecc_time_s: float = 0.0
    con_time_s: float = 0.0

class PushupCounter:
    def __init__(self, up_thr=160, down_thr=90, speak_feedback=False):
        self.up_thr, self.down_thr = up_thr, down_thr
        self.stage, self.reps, self.curr = None, 0, None
        self.last_time = None
        self.completed = []
        self.speak_feedback = speak_feedback

    def update(self, ang, frame_idx, now):
        if self.last_time is None: self.last_time = now
        dt = now - self.last_time; self.last_time = now
        if self.curr is None: self.curr = RepStats(frame_idx)

        self.curr.min_angle = min(self.curr.min_angle, ang)
        self.curr.max_angle = max(self.curr.max_angle, ang)
        if self.stage == "up": self.curr.ecc_time_s += dt
        else: self.curr.con_time_s += dt

        rep_out = None
        if ang <= self.down_thr and self.stage == "up":
            self.stage = "down"
        elif ang >= self.up_thr and self.stage == "down":
            self.stage = "up"; self.reps += 1
            rep_out = self.finalize(frame_idx)
            if self.speak_feedback:
                speak(f"Rep {self.reps} completed")
        elif self.stage is None:
            self.stage = "up" if ang >= self.up_thr else "down"
        return self.reps, self.stage, rep_out

    def finalize(self, frame_idx):
        span = 115  # rough elbow ROM span
        rom_pct = (self.curr.max_angle - self.curr.min_angle) / span * 100
        rep = {
            "rep_index": self.reps,
            "min_angle": round(self.curr.min_angle, 1),
            "max_angle": round(self.curr.max_angle, 1),
            "rom_pct": round(rom_pct, 1),
            "ecc_time_s": round(self.curr.ecc_time_s, 2),
            "con_time_s": round(self.curr.con_time_s, 2),
            "rep_time_s": round(self.curr.ecc_time_s + self.curr.con_time_s, 2)
        }
        self.completed.append(rep)
        self.curr = RepStats(frame_idx)
        return rep

mp_pose, mp_draw = mp.solutions.pose, mp.solutions.drawing_utils

def get_xy(lm, w, h, idx):
    p = lm[idx]; return (int(p.x * w), int(p.y * h))

def elbow_angle_min_side(lm, w, h):
    LSH, LEL, LWR = get_xy(lm, w,h,11), get_xy(lm, w,h,13), get_xy(lm, w,h,15)
    RSH, REL, RWR = get_xy(lm, w,h,12), get_xy(lm, w,h,14), get_xy(lm, w,h,16)
    return min(angle(LSH, LEL, LWR), angle(RSH, REL, RWR))

def run(input_path, use_webcam, out_dir: Path, speak_feedback=True, frame_csv=False, no_window=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(0 if use_webcam else input_path)
    fps, width, height = cap.get(cv2.CAP_PROP_FPS) or 30, int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec (browser-friendly)
    out_video = out_dir / "annotated.mp4"
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

    counter, frame_idx = PushupCounter(speak_feedback=speak_feedback), 0
    per_frame, per_rep = [], []
    with mp_pose.Pose() as pose:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1); rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb); ang = np.nan
            if res.pose_landmarks: ang = elbow_angle_min_side(res.pose_landmarks.landmark, width, height)
            reps, stage, done = counter.update(float(ang), frame_idx, time.time())
            vis = frame.copy()
            if res.pose_landmarks: mp_draw.draw_landmarks(vis, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(vis,f"Reps:{reps}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            writer.write(vis)
            if frame_csv: per_frame.append({"frame":frame_idx,"angle":ang,"reps":reps,"stage":stage})
            if done: per_rep.append(done)
            frame_idx += 1
    cap.release(); writer.release()
    if frame_csv: pd.DataFrame(per_frame).to_csv(out_dir/"per_frame.csv",index=False)
    if per_rep: pd.DataFrame(per_rep).to_csv(out_dir/"per_rep.csv",index=False)
