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
