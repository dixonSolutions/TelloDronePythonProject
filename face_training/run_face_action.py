import os
import sys
import json
import time
import argparse
from typing import Tuple, List

import cv2
import numpy as np
try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None


HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.append(ROOT)

from djitellopy import Tello  # type: ignore
from tell_video import TelloVideo  # type: ignore


MODELS_ROOT = os.path.join(HERE, 'models')

def _list_model_dirs() -> List[str]:
    try:
        return [d for d in os.listdir(MODELS_ROOT) if os.path.isdir(os.path.join(MODELS_ROOT, d))]
    except Exception:
        return []

def _resolve_model_dir(name: str | None) -> str:
    if name:
        cand = os.path.join(MODELS_ROOT, name)
        if os.path.isdir(cand):
            return cand
        raise FileNotFoundError(f"Model '{name}' not found under {MODELS_ROOT}")
    candidates = _list_model_dirs()
    if len(candidates) == 1:
        return os.path.join(MODELS_ROOT, candidates[0])
    if len(candidates) == 0:
        return HERE
    raise RuntimeError(f"Multiple models found {candidates}. Specify --model <name> or set FACE_MODEL_NAME env var.")

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', dest='model', default=os.environ.get('FACE_MODEL_NAME'))
    return p.parse_args()

ARGS = _parse_args()
MODEL_DIR = _resolve_model_dir(ARGS.model)
MODEL_PATH = os.path.join(MODEL_DIR, 'lbph.yml')
MODEL_ORB_PATH = os.path.join(MODEL_DIR, 'orb.npz')
META_PATH = os.path.join(MODEL_DIR, 'meta.json')
MODEL_MP_PATH = os.path.join(MODEL_DIR, 'mp.npz')


def load_model():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError("Meta not found. Run train_face_model.py first.")
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    algo = meta.get('algorithm', 'lbph')
    if algo == 'mp':
        if not os.path.exists(MODEL_MP_PATH):
            raise FileNotFoundError("MediaPipe model file missing.")
        data = np.load(MODEL_MP_PATH)
        centroid = data['centroid']
        thr = float(data['thr'])
        return ('mp', (centroid, thr), meta)
    elif algo == 'lbph':
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()  # type: ignore[attr-defined]
        except Exception:
            raise RuntimeError("LBPH model selected but cv2.face is unavailable. Re-train to use ORB fallback.")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("LBPH model file missing.")
        recognizer.read(MODEL_PATH)
        return ('lbph', recognizer, meta)
    else:
        if not os.path.exists(MODEL_ORB_PATH):
            raise FileNotFoundError("ORB model file missing.")
        data = np.load(MODEL_ORB_PATH)
        train_desc = data['train_desc']
        return ('orb', train_desc, meta)


def mp_embed_from_bgr(img_bgr: np.ndarray):
    if mp is None:
        return None
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False) as fm:
            res = fm.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lms = res.multi_face_landmarks[0].landmark
            coords = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
            coords = coords - coords.mean(axis=0, keepdims=True)
            denom = coords.std() + 1e-6
            coords = coords / denom
            emb = coords.flatten()
            return emb
    except Exception:
        return None


def main() -> None:
    algo, model, meta = load_model()
    threshold = float(min(max(meta.get('threshold', 80.0), 70.0), 90.0)) if meta.get('algorithm', 'lbph') == 'lbph' else None
    distance_thresh = int(meta.get('distance_thresh', 32)) if meta.get('algorithm', 'lbph') != 'lbph' else None
    min_good = int(meta.get('min_good', 25)) if meta.get('algorithm', 'lbph') != 'lbph' else None
    face_w, face_h = meta.get('face_size', [200, 200])
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    tello = Tello()
    tello.connect()
    time.sleep(1)
    video = TelloVideo(tello)
    video.start()
    print("Connected. Press Ctrl+C to stop. Takeoff manually, or adapt to auto-takeoff if desired.")

    last_action_ts = 0.0
    ACTION_COOLDOWN = 5.0

    try:
        while True:
            frame = video.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            match_ok = False
            if algo == 'mp':
                emb = mp_embed_from_bgr(frame)
                if emb is not None:
                    centroid, thr = model  # type: ignore[misc]
                    dist = float(np.linalg.norm(emb - centroid))
                    match_ok = dist <= thr
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dets = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
                if len(dets) > 0:
                    x, y, w, h = sorted(dets, key=lambda d: d[2] * d[3], reverse=True)[0]
                    pad = int(0.15 * max(w, h))
                    x0 = max(0, x - pad)
                    y0 = max(0, y - pad)
                    x1 = min(gray.shape[1], x + w + pad)
                    y1 = min(gray.shape[0], y + h + pad)
                    face = gray[y0:y1, x0:x1]
                    face = cv2.resize(face, (face_w, face_h), interpolation=cv2.INTER_AREA)
                    if algo == 'lbph':
                        try:
                            label, conf = model.predict(face)  # type: ignore[union-attr]
                        except Exception:
                            label, conf = (-1, 1e9)
                        match_ok = (label == meta.get('label', 1) and conf <= (threshold or 80.0))
                    else:
                        orb = cv2.ORB_create(nfeatures=1000)
                        _, qdesc = orb.detectAndCompute(face, None)
                        if qdesc is not None and len(qdesc) > 0:
                            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                            knn = bf.knnMatch(qdesc, model, k=2)  # type: ignore[arg-type]
                            good = []
                            for m, n in knn:
                                if m.distance < 0.75 * n.distance and m.distance <= (distance_thresh or 32):
                                    good.append(m)
                            match_ok = len(good) >= (min_good or 25)
                if algo == 'lbph':
                    try:
                        label, conf = model.predict(face)  # type: ignore[union-attr]
                    except Exception:
                        label, conf = (-1, 1e9)
                    match_ok = (label == meta.get('label', 1) and conf <= (threshold or 80.0))
                else:
                    orb = cv2.ORB_create(nfeatures=1000)
                    _, qdesc = orb.detectAndCompute(face, None)
                    if qdesc is not None and len(qdesc) > 0:
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                        knn = bf.knnMatch(qdesc, model, k=2)  # type: ignore[arg-type]
                        good = []
                        for m, n in knn:
                            if m.distance < 0.75 * n.distance and m.distance <= (distance_thresh or 32):
                                good.append(m)
                        match_ok = len(good) >= (min_good or 25)
                if match_ok:
                    now = time.time()
                    if now - last_action_ts > ACTION_COOLDOWN:
                        try:
                            print("Recognized. Moving forward 50cm.")
                            tello.move_forward(50)
                        except Exception:
                            pass
                        last_action_ts = now

            # light loop delay
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            video.stop()
        except Exception:
            pass
        try:
            tello.end()
        except Exception:
            pass


if __name__ == '__main__':
    main()


