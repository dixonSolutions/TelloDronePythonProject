import os
import sys
import json
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None


HERE = os.path.dirname(__file__)

def _project_root() -> str:
    # Simple: base everything on the package directory
    try:
        return os.path.abspath(os.path.dirname(__file__))
    except Exception:
        return os.path.abspath(os.getcwd())

FACE_TRAINING_ROOT = os.path.join(_project_root(), 'face_training')
MODELS_ROOT = os.path.join(FACE_TRAINING_ROOT, 'models')

try:
    os.makedirs(MODELS_ROOT, exist_ok=True)
except Exception:
    pass


def list_models() -> List[str]:
    try:
        return [d for d in os.listdir(MODELS_ROOT) if os.path.isdir(os.path.join(MODELS_ROOT, d))]
    except Exception:
        return []


def model_paths(name: str) -> Tuple[str, str, str, str]:
    base = os.path.join(MODELS_ROOT, name)
    return (
        os.path.join(base, 'lbph.yml'),
        os.path.join(base, 'orb.npz'),
        os.path.join(base, 'mp.npz'),
        os.path.join(base, 'meta.json'),
    )


def load_model(name: str):
    lbph_path, orb_path, mp_path, meta_path = model_paths(name)
    if not os.path.exists(meta_path):
        raise FileNotFoundError('Meta not found for model %s' % name)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    algo = meta.get('algorithm', 'lbph')
    if algo == 'mp':
        if not os.path.exists(mp_path):
            raise FileNotFoundError('mp model not found')
        data = np.load(mp_path)
        centroid = data['centroid']
        thr = float(data['thr'])
        return ('mp', (centroid, thr), meta)
    elif algo == 'lbph':
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()  # type: ignore[attr-defined]
        except Exception:
            raise RuntimeError('LBPH selected but cv2.face unavailable')
        if not os.path.exists(lbph_path):
            raise FileNotFoundError('lbph model not found')
        recognizer.read(lbph_path)
        return ('lbph', recognizer, meta)
    else:
        if not os.path.exists(orb_path):
            raise FileNotFoundError('orb model not found')
        data = np.load(orb_path)
        train_desc = data['train_desc']
        return ('orb', train_desc, meta)


def mp_embed_from_bgr(img_bgr: np.ndarray) -> Optional[np.ndarray]:
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
            # bbox from landmarks
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            h, w = img_bgr.shape[:2]
            bbox = (
                int(x_min * w),
                int(y_min * h),
                max(1, int((x_max - x_min) * w)),
                max(1, int((y_max - y_min) * h)),
            )
            # normalize for embedding
            coords = coords - coords.mean(axis=0, keepdims=True)
            denom = coords.std() + 1e-6
            coords = coords / denom
            emb = coords.flatten()
            return emb, bbox  # type: ignore[return-value]
    except Exception:
        return None


_haar = None


def _haar_cascade():
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return _haar


def detect_face_bbox(gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    try:
        dets = _haar_cascade().detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        if len(dets) == 0:
            return None
        x, y, w, h = sorted(dets, key=lambda d: d[2] * d[3], reverse=True)[0]
        pad = int(0.15 * max(w, h))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(gray.shape[1], x + w + pad)
        y1 = min(gray.shape[0], y + h + pad)
        return (x0, y0, x1 - x0, y1 - y0)
    except Exception:
        return None


def recognize_in_frame(model_info, frame_bgr: np.ndarray):
    algo, model, meta = model_info
    name = meta.get('name', 'face')
    if algo == 'mp':
        res = mp_embed_from_bgr(frame_bgr)
        if res is None:
            return (False, None, name, None)
        emb, bbox = res
        centroid, thr = model
        dist = float(np.linalg.norm(emb - centroid))
        return (dist <= thr, bbox, name, dist)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    bbox = detect_face_bbox(gray)
    if bbox is None:
        return (False, None, name, None)
    x, y, w, h = bbox
    face = gray[y:y + h, x:x + w]
    face_size = meta.get('face_size', [200, 200])
    face = cv2.resize(face, (face_size[0], face_size[1]), interpolation=cv2.INTER_AREA)
    if algo == 'lbph':
        try:
            label, conf = model.predict(face)  # type: ignore[union-attr]
        except Exception:
            label, conf = (-1, 1e9)
        thr = float(min(max(meta.get('threshold', 80.0), 70.0), 90.0))
        ok = (label == meta.get('label', 1) and conf <= thr)
        return (ok, bbox, name, conf)
    # ORB
    orb = cv2.ORB_create(nfeatures=1000)
    _, qdesc = orb.detectAndCompute(face, None)
    if qdesc is None or len(qdesc) == 0:
        return (False, None, name, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(qdesc, model, k=2)  # type: ignore[arg-type]
    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance and m.distance <= int(meta.get('distance_thresh', 32)):
            good.append(m)
    ok = len(good) >= int(meta.get('min_good', 25))
    return (ok, bbox, name, float(len(good)))


def find_samples_for_model(name: str) -> List[str]:
    # Guess images folder by name + '_images', else any dir containing name and 'images'
    candidates = []
    try:
        base_dirs = [FACE_TRAINING_ROOT]
        for bd in base_dirs:
            for d in os.listdir(bd):
                p = os.path.join(bd, d)
                if not os.path.isdir(p):
                    continue
                dn = d.lower()
                if dn == f'{name.lower()}_images' or (name.lower() in dn and 'images' in dn):
                    candidates.append(p)
    except Exception:
        pass
    images: List[str] = []
    for c in candidates:
        try:
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                images.extend([os.path.join(c, f) for f in os.listdir(c) if f.lower().endswith(ext.split('*.')[-1])])
        except Exception:
            pass
    return images


