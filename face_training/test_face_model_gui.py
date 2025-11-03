import os
import sys
import json
import tkinter as tk
import tkinter.filedialog as fd
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None


HERE = os.path.dirname(__file__)
MODELS_ROOT = os.path.join(HERE, 'models')

def _list_model_dirs():
    try:
        result = []
        for d in os.listdir(MODELS_ROOT):
            p = os.path.join(MODELS_ROOT, d)
            if not os.path.isdir(p):
                continue
            meta = os.path.join(p, 'meta.json')
            if not os.path.exists(meta):
                continue
            # must have at least one model file
            if any(os.path.exists(os.path.join(p, fn)) for fn in ('lbph.yml', 'orb.npz', 'mp.npz')):
                result.append(d)
        return result
    except Exception:
        return []

MODEL_DIR = HERE
MODEL_PATH = os.path.join(MODEL_DIR, 'lbph.yml')
MODEL_ORB_PATH = os.path.join(MODEL_DIR, 'orb.npz')
META_PATH = os.path.join(MODEL_DIR, 'meta.json')


def load_model():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError("Meta not found. Train the model first.")
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    algo = meta.get('algorithm', 'lbph')
    if algo == 'mp':
        data = np.load(os.path.join(MODEL_DIR, 'mp.npz'))
        return ('mp', (data['centroid'], float(data['thr'])), meta)
    elif algo == 'lbph':
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()  # type: ignore[attr-defined]
        except Exception:
            raise RuntimeError("LBPH meta found but cv2.face is unavailable. Re-train to use ORB.")
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


def detect_and_crop(gray: np.ndarray) -> np.ndarray:
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    dets = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(dets) == 0:
        # fallback center crop
        h, w = gray.shape[:2]
        y0 = max(0, h // 4)
        y1 = min(h, 3 * h // 4)
        x0 = max(0, w // 4)
        x1 = min(w, 3 * w // 4)
        return gray[y0:y1, x0:x1]
    x, y, w, h = sorted(dets, key=lambda d: d[2] * d[3], reverse=True)[0]
    pad = int(0.15 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    return gray[y0:y1, x0:x1]


def mp_embed_from_bgr(img_bgr: np.ndarray):
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
            coords = coords - coords.mean(axis=0, keepdims=True)
            denom = coords.std() + 1e-6
            coords = coords / denom
            emb = coords.flatten()
            return emb
    except Exception:
        return None


def main() -> None:
    # Console input first: ask for model folder name under face_training/models
    global MODEL_DIR, MODEL_PATH, MODEL_ORB_PATH, META_PATH
    name_arg = sys.argv[1].strip() if len(sys.argv) > 1 else ''
    if not name_arg:
        try:
            name_arg = input("Enter model name under 'face_training/models': ").strip()
        except Exception:
            name_arg = ''
    if not name_arg:
        print("No model name provided. Available models:")
        for d in _list_model_dirs():
            print(f" - {d}")
        return
    candidate = os.path.join(MODELS_ROOT, name_arg)
    if not os.path.isdir(candidate):
        print(f"Model '{name_arg}' not found under {MODELS_ROOT}.")
        print("Available models:")
        for d in _list_model_dirs():
            print(f" - {d}")
        return

    MODEL_DIR = candidate
    MODEL_PATH = os.path.join(MODEL_DIR, 'lbph.yml')
    MODEL_ORB_PATH = os.path.join(MODEL_DIR, 'orb.npz')
    META_PATH = os.path.join(MODEL_DIR, 'meta.json')

    # Defaults before load; select_model will overwrite
    algo = None
    model = None
    meta = {}
    face_w, face_h = (200, 200)
    threshold = None
    distance_thresh = None
    min_good = None

    root = tk.Tk()
    root.title("Face Model Tester")

    # Model chooser
    top = tk.Frame(root)
    top.pack(fill='x')
    tk.Label(top, text='Model:').pack(side='left', padx=6)
    model_var = tk.StringVar()
    model_choices = _list_model_dirs()
    model_box = tk.OptionMenu(top, model_var, *model_choices)
    model_box.pack(side='left')
    tk.Label(top, text='or name:').pack(side='left', padx=6)
    model_entry = tk.Entry(top, width=16)
    model_entry.pack(side='left')
    def load_by_name():
        name = model_entry.get().strip()
        if not name:
            return
        select_model(name)
    tk.Button(top, text='Load', command=load_by_name).pack(side='left', padx=6)

    def select_model(name: str):
        global MODEL_DIR, MODEL_PATH, MODEL_ORB_PATH, META_PATH
        if not name:
            return
        MODEL_DIR = os.path.join(MODELS_ROOT, name)
        MODEL_PATH = os.path.join(MODEL_DIR, 'lbph.yml')
        MODEL_ORB_PATH = os.path.join(MODEL_DIR, 'orb.npz')
        META_PATH = os.path.join(MODEL_DIR, 'meta.json')
        try:
            # reload model with new dir
            nonlocal algo, model, meta, threshold, distance_thresh, min_good, face_w, face_h
            algo, model, meta = load_model()
            face_w, face_h = meta.get('face_size', [200, 200])
            threshold = float(meta.get('threshold', 80.0)) if meta.get('algorithm', 'lbph') == 'lbph' else None
            distance_thresh = int(meta.get('distance_thresh', 32)) if meta.get('algorithm', 'lbph') != 'lbph' else None
            min_good = int(meta.get('min_good', 25)) if meta.get('algorithm', 'lbph') != 'lbph' else None
            result_var.set(f"Loaded model '{name}'")
        except Exception as e:
            result_var.set(f"Load error: {e}")

    if model_choices:
        # set initial selection to the console-provided name if present
        initial = name_arg if name_arg in model_choices else model_choices[0]
        model_var.set(initial)
        try:
            select_model(initial)
        except Exception:
            pass
    else:
        result_var = tk.StringVar()
        # show message; user can type a name path
        tk.Label(root, text='No models found in face_training/models').pack()
    model_var.trace_add('write', lambda *_: select_model(model_var.get()))

    img_label = tk.Label(root)
    img_label.pack(padx=10, pady=10)

    result_var = tk.StringVar(value="Open an image to test.")
    tk.Label(root, textvariable=result_var).pack(pady=6)

    # No sensitivity control; use default thresholds

    def open_image():
        path = fd.askopenfilename(title='Choose an image', filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp')])
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            result_var.set("Failed to load image.")
            return
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Match training preprocessing
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            try:
                gray = cv2.equalizeHist(gray)
            except Exception:
                pass
        face = detect_and_crop(gray)
        face = cv2.resize(face, (face_w, face_h), interpolation=cv2.INTER_AREA)

        recognized = False
        detail = ''
        if algo == 'mp':
            emb = mp_embed_from_bgr(bgr)
            if emb is None:
                recognized = False
                detail = 'no face landmarks'
            else:
                centroid, thr = model  # type: ignore[misc]
                dist = float(np.linalg.norm(emb - centroid))
                recognized = dist <= thr
                detail = f"dist={dist:.3f} thr={thr:.3f}"
        elif algo == 'lbph':
            try:
                label, conf = model.predict(face)  # type: ignore[union-attr]
            except Exception:
                label, conf = (-1, 1e9)
            thr = float(min(max((threshold or 80.0), 70.0), 90.0))
            recognized = (label == meta.get('label', 1) and conf <= thr)
            detail = f"conf={conf:.1f} threshold={thr:.1f}"
        else:
            orb = cv2.ORB_create(nfeatures=1000)
            _, qdesc = orb.detectAndCompute(face, None)
            if qdesc is not None and len(qdesc) > 0:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                knn = bf.knnMatch(qdesc, model, k=2)  # type: ignore[arg-type]
                dist = (distance_thresh or 32)
                good = []
                for m, n in knn:
                    if m.distance < 0.75 * n.distance and m.distance <= dist:
                        good.append(m)
                mg = (min_good or 25)
                recognized = len(good) >= mg
                detail = f"good={len(good)} min_good={mg} dist<= {dist}"
            else:
                recognized = False
                detail = "no descriptors"

        # show image scaled
        disp = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = disp.shape[:2]
        max_w = 640
        scale = min(1.0, max_w / float(w))
        disp = cv2.resize(disp, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(disp))
        img_label.configure(image=imgtk)
        img_label.image = imgtk

        result_var.set(f"Recognized: {'YES' if recognized else 'NO'} ({detail})")

    tk.Button(root, text='Open Image...', command=open_image).pack(pady=8)
    root.mainloop()


if __name__ == '__main__':
    main()


