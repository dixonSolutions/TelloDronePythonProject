import os
import sys
import json
import glob
from typing import List, Tuple

import cv2
import numpy as np
try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None  # optional


SCRIPT_DIR = os.path.dirname(__file__)
MODELS_ROOT = os.path.join(SCRIPT_DIR, 'models')

# Resolve images directory: CLI arg or interactive prompt (name or path)
def _auto_candidates():
    try:
        c = []
        for d in os.listdir(SCRIPT_DIR):
            p = os.path.join(SCRIPT_DIR, d)
            if os.path.isdir(p) and d.lower().endswith('_images'):
                # must contain at least one image
                has_img = any(f.lower().endswith((".jpg",".jpeg",".png",".bmp")) for f in os.listdir(p))
                if has_img:
                    c.append(d)
        return c
    except Exception:
        return []

def _normalize_dir(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(SCRIPT_DIR, path_str)

def _is_under_models_root(path_str: str) -> bool:
    try:
        ap = os.path.abspath(path_str)
        mr = os.path.abspath(MODELS_ROOT)
        return ap == mr or ap.startswith(mr + os.sep)
    except Exception:
        return False

if len(sys.argv) > 1:
    cand_input = sys.argv[1].strip()
    cand_path = _normalize_dir(cand_input)
    if _is_under_models_root(cand_path):
        print("You entered the models directory. Please pass the IMAGES folder (e.g., '<name>_images').")
        raise SystemExit(1)
    _input_dir = cand_input
else:
    # Always prompt user to choose/provide an IMAGES folder; show detected candidates, but do not auto-pick
    while True:
        cand = _auto_candidates()
        if cand:
            print("Detected *_images folders:")
            for d in cand:
                print(f" - {d}")
        try:
            _input_dir = input("Enter IMAGES folder (name or path): ").strip()
        except Exception:
            _input_dir = ''
        if not _input_dir:
            print("Please provide an images folder name or path (not the models folder).")
            continue
        cand_path = _normalize_dir(_input_dir)
        if _is_under_models_root(cand_path):
            print("That path is under 'models/'. Please enter the IMAGES folder (e.g., '<name>_images').")
            continue
        break

DATA_DIR = _normalize_dir(_input_dir)

if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Created images folder: {DATA_DIR}. Add 5-10 face photos and run again.")
        # exit early so user can add images
        raise SystemExit(0)
    except Exception:
        pass

# Resolve person name from images folder (strip optional _images suffix)
_base = os.path.basename(DATA_DIR.rstrip('/'))
_name = _base[:-7] if _base.endswith('_images') else _base
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models', _name)
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, 'lbph.yml')
META_PATH = os.path.join(MODELS_DIR, 'meta.json')
MODEL_ORB_PATH = os.path.join(MODELS_DIR, 'orb.npz')
MODEL_MP_PATH = os.path.join(MODELS_DIR, 'mp.npz')

FACE_SIZE = (200, 200)


def load_images_with_faces(image_paths: List[str]) -> List[np.ndarray]:
    faces: List[np.ndarray] = []
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Preprocess to improve consistency
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            try:
                gray = cv2.equalizeHist(gray)
            except Exception:
                pass
        dets = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        if len(dets) == 0:
            # fallback: take center crop
            h, w = gray.shape[:2]
            y0 = max(0, h // 4)
            y1 = min(h, 3 * h // 4)
            x0 = max(0, w // 4)
            x1 = min(w, 3 * w // 4)
            crop = gray[y0:y1, x0:x1]
        else:
            # largest face
            x, y, w, h = sorted(dets, key=lambda d: d[2] * d[3], reverse=True)[0]
            pad = int(0.15 * max(w, h))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(gray.shape[1], x + w + pad)
            y1 = min(gray.shape[0], y + h + pad)
            crop = gray[y0:y1, x0:x1]
        face = cv2.resize(crop, FACE_SIZE, interpolation=cv2.INTER_AREA)
        faces.append(face)
    return faces


def compute_threshold(recognizer, faces: List[np.ndarray], label: int) -> float:
    # Evaluate confidence on training faces; LBPH confidence: lower is better
    confidences: List[float] = []
    for f in faces:
        try:
            _, conf = recognizer.predict(f)
            confidences.append(float(conf))
        except Exception:
            pass
    if not confidences:
        return 80.0
    avg = float(np.mean(confidences))
    std = float(np.std(confidences))
    # Use avg + 1.5*std to allow variation; clamp to 70–90 per defaults
    thr = avg + 1.5 * max(std, 5.0)
    return float(np.clip(thr, 70.0, 90.0))


def train_orb(faces: List[np.ndarray]) -> None:
    # ORB-based descriptor aggregation fallback (no opencv-contrib required)
    orb = cv2.ORB_create(nfeatures=1000)
    descs = []
    lens = []
    for f in faces:
        kp, des = orb.detectAndCompute(f, None)
        if des is None or len(des) == 0:
            continue
        descs.append(des)
        lens.append(len(des))
    if not descs:
        raise RuntimeError("ORB: could not extract descriptors from training faces.")
    train_desc = np.vstack(descs)
    avg_kp = float(np.mean(lens))
    # distance threshold for Hamming; good matches are <= 32
    distance_thresh = 32
    # minimum number of good matches to accept recognition
    min_good = max(25, int(0.5 * avg_kp))
    np.savez_compressed(MODEL_ORB_PATH, train_desc=train_desc)
    with open(META_PATH, 'w') as f:
        json.dump({
            "algorithm": "orb",
            "name": _name,
            "distance_thresh": distance_thresh,
            "min_good": min_good,
            "face_size": FACE_SIZE,
        }, f)
    print(f"ORB model saved: {MODEL_ORB_PATH}")
    print(f"Meta saved: {META_PATH} (min_good={min_good}, dist<= {distance_thresh})")


def extract_mp_embedding_from_bgr(img_bgr: np.ndarray):
    if mp is None:
        return None
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False) as fm:
            res = fm.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lms = res.multi_face_landmarks[0].landmark
            coords = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)
            # normalize: center and scale to unit variance
            coords = coords - coords.mean(axis=0, keepdims=True)
            denom = coords.std() + 1e-6
            coords = coords / denom
            emb = coords.flatten()
            return emb
    except Exception:
        return None


def train_mediapipe(image_paths: List[str]) -> bool:
    if mp is None:
        return False
    embs = []
    for p in image_paths:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        emb = extract_mp_embedding_from_bgr(bgr)
        if emb is not None:
            embs.append(emb)
    if len(embs) < 3:
        return False
    embs_np = np.stack(embs, axis=0)
    centroid = embs_np.mean(axis=0)
    dists = np.linalg.norm(embs_np - centroid, axis=1)
    mu = float(dists.mean())
    sigma = float(dists.std())
    thr = mu + 2.5 * max(sigma, 0.05)
    np.savez_compressed(MODEL_MP_PATH, centroid=centroid, thr=thr)
    with open(META_PATH, 'w') as f:
        json.dump({
            "algorithm": "mp",
            "name": _name,
            "face_size": FACE_SIZE,
            "mp_model": os.path.basename(MODEL_MP_PATH),
            "mp_thr": thr,
        }, f)
    print(f"MediaPipe geometry model saved: {MODEL_MP_PATH} (thr={thr:.3f})")
    return True


def augment_face(face: np.ndarray) -> List[np.ndarray]:
    # Basic augmentation: small rotations and brightness/contrast jitter
    aug: List[np.ndarray] = []
    h, w = face.shape[:2]
    center = (w // 2, h // 2)
    for angle in (-12, -6, 0, 6, 12):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        for alpha, beta in ((1.0, 0), (1.1, 5), (0.9, -5)):
            jitter = cv2.convertScaleAbs(rotated, alpha=alpha, beta=beta)
            aug.append(jitter)
    return aug


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_paths.extend(glob.glob(os.path.join(DATA_DIR, ext)))
    if len(image_paths) < 5:
        print(f"Not enough images in {DATA_DIR}. Provide at least 5.")
        return

    base_faces = load_images_with_faces(image_paths)
    if len(base_faces) == 0:
        print("No faces could be extracted from the images.")
        return
    # Augment
    faces: List[np.ndarray] = []
    for f in base_faces:
        faces.extend(augment_face(f))

    labels = np.full((len(faces),), 1, dtype=np.int32)
    # Prefer MediaPipe geometry if available and successful
    if train_mediapipe(image_paths):
        return

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)  # type: ignore[attr-defined]
        recognizer.train(faces, labels)
        recognizer.write(MODEL_PATH)
        # Compute threshold on original (non-aug) faces to avoid overfitting to aug set
        threshold = compute_threshold(recognizer, base_faces, 1)
        with open(META_PATH, 'w') as f:
            json.dump({
                "algorithm": "lbph",
                "label": 1,
                "name": _name,
                "threshold": threshold,
                "face_size": FACE_SIZE
            }, f)
        print(f"LBPH model saved: {MODEL_PATH}")
        print(f"Meta saved: {META_PATH} (threshold={threshold:.1f})")
    except Exception:
        print("LBPH unavailable; falling back to ORB descriptors.")
        train_orb(faces)


if __name__ == '__main__':
    main()


