import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkfont
import customtkinter as ctk
import tkinter.filedialog as fd
from PIL import Image, ImageTk
import cv2
import os
import time
import random
import sys
import shutil
from djitellopy import Tello
from tell_video import TelloVideo
from free_fly import FreeFlyController
from models import list_models, load_model, recognize_in_frame, find_samples_for_model
from ui import BlueButton, ToggleSwitch, Tabs, Dialog, Theme

# Connection state (lazy connect)
tello = None
video = None
is_connected = False
video_started = False
last_video_ts = 0.0

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.title("Yo, ServiceFuzz!")

# Fedora-like blue/white styling (rounded-like via padding)
FEDORA_BLUE = '#3c6eb4'
BG = '#f5f7fa'
FG = '#0f1b2a'
style = ttk.Style()
try:
    style.theme_use('clam')
except Exception:
    pass
try:
    style.configure('.', background=BG, foreground=FG, font=('Sans', 10))
    style.configure('TFrame', background=BG)
    style.configure('TLabel', background=BG, foreground=FG, font=('Sans', 11))
    style.configure('TNotebook', background=BG, tabmargins=[6, 6, 6, 0])
    style.configure('TNotebook.Tab', background=BG, foreground=FG, padding=[16, 10], font=('Sans', 11, 'bold'))
    style.map('TNotebook.Tab', background=[('selected', FEDORA_BLUE)], foreground=[('selected', 'white')])
    style.configure('TButton', padding=[14, 10], font=('Sans', 11, 'bold'))
    style.configure('Big.TButton', padding=[18, 12], font=('Sans', 12, 'bold'))
    style.configure('TCheckbutton', background=BG, foreground=FG, padding=[8, 8])
except Exception:
    pass
try:
    root.configure(bg=BG)
except Exception:
    pass

video_label = ctk.CTkLabel(root)

SPEED = 50

# Header with status and battery
header = ctk.CTkFrame(root)
status_var = tk.StringVar(value="Connecting...")
status_label = ctk.CTkLabel(header, textvariable=status_var)
status_label.pack(side='left', padx=8)


# Live stats (height, attitude)
height_var = tk.StringVar(value="H: - cm")
tilt_var = tk.StringVar(value="Att: P -° R -° Y -°")
height_label = ctk.CTkLabel(header, textvariable=height_var)
height_label.pack(side='left', padx=8)
tilt_label = ctk.CTkLabel(header, textvariable=tilt_var)
tilt_label.pack(side='left', padx=8)
battery_var = tk.StringVar(value="Battery: -")
battery_label = ctk.CTkLabel(header, textvariable=battery_var)
battery_label.pack(side='right', padx=8)

# Quit button (red) on header, with 'q' shortcut
quit_btn = ctk.CTkButton(header, text='Quit', fg_color='#c62828', hover_color='#b71c1c', command=lambda: on_close())
quit_btn.pack(side='right', padx=8)

# Action spinner (shown during async actions)
action_spinner = ctk.CTkProgressBar(header, width=120, mode='indeterminate')
_active_ops = 0

def _start_activity():
    global _active_ops
    _active_ops += 1
    try:
        action_spinner.pack(side='right', padx=8)
        try:
            action_spinner.start()
        except TypeError:
            # Fallback for ttk Progressbar signature if present
            try:
                action_spinner.start(20)
            except Exception:
                pass
    except Exception:
        pass

def _stop_activity():
    global _active_ops
    _active_ops = max(0, _active_ops - 1)
    if _active_ops == 0:
        try:
            action_spinner.stop()
            action_spinner.pack_forget()
        except Exception:
            pass

# Thread-safe status logger: prints to console and updates UI on main thread
def ui_status(message: str):
    try:
        print(message)
    except Exception:
        pass
    try:
        root.after(0, lambda: status_var.set(message))
    except Exception:
        pass

# Tabs: Controls (default) and Modes using custom ui Tabs
tabs = Tabs(root)
tab_controls = tabs.add('Controls')
tab_modes = tabs.add('Modes')
tab_tricks = tabs.add('Tricks')
tab_models = tabs.add('Models')
tab_settings = tabs.add('Settings')

# Controls tab content
controls = ctk.CTkFrame(tab_controls)
controls.pack(pady=8)

# Control pads (movement, yaw, altitude)
pad = ctk.CTkFrame(controls)
pad.pack(pady=6)

def rc_nudge(lr=0, fb=0, ud=0, yw=0, dur=0.3):
    try:
        if not is_connected or tello is None:
            status_var.set("Not connected")
            return
        t_end = time.time() + dur
        while time.time() < t_end:
            try:
                tello.send_rc_control(lr, fb, ud, yw)
            except Exception:
                break
            time.sleep(0.05)
        try:
            tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass
    except Exception:
        pass

pad_move = ctk.CTkFrame(pad)
pad_move.grid(row=0, column=0, padx=10)
ctk.CTkButton(pad_move, text='↑', width=36, command=lambda: run_async(lambda: rc_nudge(fb=SPEED))).grid(row=0, column=1, pady=2)
ctk.CTkButton(pad_move, text='←', width=36, command=lambda: run_async(lambda: rc_nudge(lr=-SPEED))).grid(row=1, column=0, padx=2)
ctk.CTkButton(pad_move, text='■', width=36, command=lambda: run_async(lambda: rc_nudge(0,0,0,0,0.05))).grid(row=1, column=1)
ctk.CTkButton(pad_move, text='→', width=36, command=lambda: run_async(lambda: rc_nudge(lr=SPEED))).grid(row=1, column=2, padx=2)
ctk.CTkButton(pad_move, text='↓', width=36, command=lambda: run_async(lambda: rc_nudge(fb=-SPEED))).grid(row=2, column=1, pady=2)

pad_turn = ctk.CTkFrame(pad)
pad_turn.grid(row=0, column=1, padx=10)
ctk.CTkButton(pad_turn, text='⟲', width=36, command=lambda: run_async(lambda: rc_nudge(yw=-SPEED))).grid(row=0, column=0, padx=4)
ctk.CTkButton(pad_turn, text='⟳', width=36, command=lambda: run_async(lambda: rc_nudge(yw=SPEED))).grid(row=0, column=1, padx=4)

pad_alt = ctk.CTkFrame(pad)
pad_alt.grid(row=0, column=2, padx=10)
ctk.CTkButton(pad_alt, text='Up', width=48, command=lambda: run_async(lambda: rc_nudge(ud=SPEED))).grid(row=0, column=0, pady=2)
ctk.CTkButton(pad_alt, text='Down', width=48, command=lambda: run_async(lambda: rc_nudge(ud=-SPEED))).grid(row=1, column=0, pady=2)

# Buttons area (use grid inside its own frame to avoid mixing with pack on parent)
controls_buttons = ttk.Frame(controls)
controls_buttons.pack(pady=6)

# Place video after tabs so controls stay visible (packed when overlay hides)

is_flying = False
last_takeoff_attempt_ts = 0.0
pressed = set()
free_fly_active = False
free_fly_controller = None

# Recording state
recording = False
video_writer = None
record_dir = "/var/home/ratrad/Videos/Tello"
ttk.Style().configure('Rec.TLabel', foreground='#c62828')
rec_var = tk.StringVar(value='')
rec_label = ttk.Label(header, textvariable=rec_var, style='Rec.TLabel')
rec_label.pack(side='left', padx=8)


# Face recognition state
recognition_active = False
recognition_model_info = None
recognition_name = None
recognition_last_bbox = None
recognition_last_seen_ts = 0.0
recognition_streak_start_ts = 0.0
recognition_action_done = False
model_settings = {}

def takeoff():
    global is_flying, last_takeoff_attempt_ts
    now = time.time()
    if now - last_takeoff_attempt_ts < 5:
        return
    last_takeoff_attempt_ts = now
    try:
        try:
            batt = tello.get_battery()
            status_var.set(f"Battery {batt}%")
            if isinstance(batt, int) and batt < 15:
                status_var.set("Battery too low for takeoff")
                return
        except Exception:
            pass
        status_var.set("Taking off...")
        # ensure no movement before takeoff
        try:
            tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass
        tello.takeoff()
        is_flying = True
        # Nudge down slightly to keep low
        try:
            tello.move_down(20)
        except Exception:
            pass
        status_var.set("Airborne (low)")
    except Exception:
        # Re-enter SDK mode and retry once, per README simple flow
        try:
            tello.connect()
            time.sleep(1)
            tello.takeoff()
            is_flying = True
            status_var.set("Airborne")
            return
        except Exception:
            status_var.set("Takeoff error")

def land():
    global is_flying
    try:
        status_var.set("Landing...")
        tello.land()
        is_flying = False
        status_var.set("Landed")
    except Exception:
        status_var.set("Land error")

def emergency():
    try:
        tello.emergency()
        status_var.set("EMERGENCY STOP")
    except Exception:
        status_var.set("Emergency error")

def run_async(fn, btn=None):
    def _runner():
        try:
            fn()
        finally:
            try:
                if btn is not None:
                    try:
                        btn.state(['!disabled'])
                    except Exception:
                        pass
                root.after(0, _stop_activity)
            except Exception:
                pass
    if btn is not None:
        try:
            btn.state(['disabled'])
        except Exception:
            pass
    _start_activity()
    import threading
    threading.Thread(target=_runner, daemon=True).start()

btn_takeoff = BlueButton(controls_buttons, text="Takeoff", command=lambda: run_async(takeoff))
btn_land = BlueButton(controls_buttons, text="Land", command=lambda: run_async(land))
btn_emg = BlueButton(controls_buttons, text="EMERGENCY", command=lambda: run_async(emergency))
btn_takeoff.grid(row=0, column=0, padx=4)
btn_land.grid(row=0, column=1, padx=4)
btn_emg.grid(row=0, column=2, padx=4)

def take_picture():
    frame = video.get_frame()
    if frame is not None:
        save_dir = "/var/home/ratrad/Pictures/Tello"
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            pass
        fname = f"picture_{int(time.time())}_{random.randint(1000,9999)}.png"
        path = os.path.join(save_dir, fname)
        try:
            # If recognition is active, draw circle and name tag before saving
            draw = frame.copy()
            try:
                if recognition_active and recognition_model_info is not None:
                    ok, bbox, name, metric = recognize_in_frame(recognition_model_info, frame)
                    if ok and bbox is not None:
                        x, y, w2, h2 = bbox
                        cx = int(x + w2 / 2)
                        cy = int(y + h2 / 2)
                        r = int(max(w2, h2) / 2)
                        cv2.circle(draw, (cx, cy), r, (0, 255, 0), 2)
                        label = recognition_name or name or 'face'
                        cv2.putText(draw, label, (max(0, x), max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception:
                pass
            cv2.imwrite(path, draw)
            print(f"Saved {path}")
            try:
                status_var.set(f"Saved {path}")
            except Exception:
                pass
        except Exception:
            pass

btn_pic = BlueButton(controls_buttons, text="Take Picture", command=lambda: run_async(take_picture))
btn_pic.grid(row=0, column=3, padx=4)

def toggle_free_fly():
    global free_fly_active, free_fly_controller
    try:
        if free_fly_active:
            if free_fly_controller is not None:
                free_fly_controller.stop()
            free_fly_active = False
            try:
                free_fly_btn.configure(text="Free Fly")
            except Exception:
                pass
            status_var.set("Free Fly stopped")
            # Ensure RC stops
            try:
                tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass
            return

        # Start free fly (auto-takeoff if needed)
        if not is_flying:
            takeoff()
        if not is_flying:
            # Takeoff failed; abort enabling mode
            status_var.set("Free Fly aborted: takeoff failed")
            return
        free_fly_controller = FreeFlyController(tello, video, status_var)
        free_fly_controller.start()
        free_fly_active = True
        try:
            free_fly_btn.configure(text="Exit Free Fly")
        except Exception:
            pass
        status_var.set("Free Fly started")
    except Exception:
        status_var.set("Free Fly error")

# Modes tab content
modes = ttk.Frame(tab_modes)
modes.pack(pady=8, fill='x')

# Centered toggles row
modes_toggles = ctk.CTkFrame(modes)
modes_toggles.pack(pady=4)

free_fly_btn = BlueButton(modes_toggles, text="Free Fly", command=lambda: run_async(toggle_free_fly))
free_fly_btn.pack(side='left', padx=8)

fast_mode_on = False

def toggle_fast_mode():
    global fast_mode_on, SPEED
    try:
        if not fast_mode_on:
            try:
                tello.set_speed(100)
            except Exception:
                pass
            SPEED = 100
            fast_btn.configure(text="Fast Mode: ON")
            status_var.set("Fast Mode enabled")
            fast_mode_on = True
        else:
            try:
                tello.set_speed(50)
            except Exception:
                pass
            SPEED = 50
            fast_btn.configure(text="Fast Mode: OFF")
            status_var.set("Fast Mode disabled")
            fast_mode_on = False
    except Exception:
        status_var.set("Fast Mode error")

fast_btn = BlueButton(modes_toggles, text="Fast Mode: OFF", command=lambda: run_async(toggle_fast_mode))
fast_btn.pack(side='left', padx=8)

 # (Voice Mode removed)

# Settings tab content
settings = ttk.Frame(tab_settings)
settings.pack(pady=8)

ttk.Label(settings, text='Video Quality').grid(row=0, column=0, padx=6, pady=4, sticky='e')
quality_var = tk.StringVar(value='Auto')
quality_combo = ttk.Combobox(settings, textvariable=quality_var, values=['Auto', 'Low', 'Medium', 'High'], state='readonly', width=10)
quality_combo.grid(row=0, column=1, padx=6, pady=4)

def apply_quality():
    try:
        if not is_connected or tello is None:
            status_var.set('Not connected')
            return
        sel = quality_var.get()
        # Best-effort mapping; ignore errors if unsupported
        if sel == 'Auto':
            try: tello.set_video_bitrate_auto()
            except Exception: pass
        elif sel == 'Low':
            try: tello.set_video_bitrate(1)
            except Exception: pass
        elif sel == 'Medium':
            try: tello.set_video_bitrate(3)
            except Exception: pass
        elif sel == 'High':
            try: tello.set_video_bitrate(4)
            except Exception: pass
        status_var.set(f'Quality applied: {sel}')
    except Exception:
        status_var.set('Quality apply error')

btn_apply_q = BlueButton(settings, text='Apply Quality', command=lambda: run_async(apply_quality))
btn_apply_q.grid(row=0, column=2, padx=6, pady=4)

# Color space handling
ctk.CTkLabel(settings, text='Color Space').grid(row=1, column=0, padx=6, pady=4, sticky='e')
color_mode_var = tk.StringVar(value='YUV420')
color_combo = ctk.CTkOptionMenu(settings, values=['Auto', 'YUV420'], variable=color_mode_var, cursor='hand2')
color_combo.grid(row=1, column=1, padx=6, pady=4)

def apply_color_mode():
    # No device call needed; handled in frame conversion
    status_var.set(f"Color mode: {color_mode_var.get()}")

btn_apply_c = BlueButton(settings, text='Apply Color', command=lambda: run_async(apply_color_mode))
btn_apply_c.grid(row=1, column=2, padx=6, pady=4)

# Tricks toggle and actions
tricks_on = False

def toggle_tricks():
    global tricks_on
    tricks_on = not tricks_on
    try:
        tricks_btn.configure(text=("Tricks: ON" if tricks_on else "Tricks: OFF"))
        if tricks_on:
            tricks_actions.pack(pady=6)
        else:
            tricks_actions.pack_forget()
        status_var.set("Tricks enabled" if tricks_on else "Tricks disabled")
    except Exception:
        pass

tricks_btn = BlueButton(modes_toggles, text="Tricks: OFF", command=lambda: run_async(toggle_tricks))
tricks_btn.pack_forget()

tricks_actions = ttk.Frame(tab_tricks)
tricks_actions.pack(pady=10)

def do_front_flip():
    if not is_flying:
        status_var.set("Flip aborted: not flying")
        return
    try:
        try:
            tello.flip_forward()
        except Exception:
            try:
                tello.flip('f')
            except Exception:
                raise
        status_var.set("Front flip")
    except Exception:
        status_var.set("Front flip error")

def do_back_flip():
    if not is_flying:
        status_var.set("Flip aborted: not flying")
        return
    try:
        try:
            tello.flip_back()
        except Exception:
            try:
                tello.flip('b')
            except Exception:
                raise
        status_var.set("Back flip")
    except Exception:
        status_var.set("Back flip error")

## Voice Mode removed

btn_flip_f = ttk.Button(tricks_actions, text="Front Flip")
btn_flip_b = ttk.Button(tricks_actions, text="Back Flip")
btn_flip_f.configure(command=lambda b=btn_flip_f: run_async(do_front_flip, b))
btn_flip_b.configure(command=lambda b=btn_flip_b: run_async(do_back_flip, b))
btn_flip_f.pack(side='left', padx=6)
btn_flip_b.pack(side='left', padx=6)

# Debug metrics from Free Fly
debug_var = tk.StringVar(value="")
debug_label = ttk.Label(modes, textvariable=debug_var)
debug_label.pack(pady=4)

# Models tab content
models_frame = ttk.Frame(tab_models)
models_frame.pack(pady=8, fill='both', expand=True)

left_panel = ctk.CTkFrame(models_frame)
left_panel.pack(side='left', fill='both', expand=True, padx=8)
right_panel = ctk.CTkFrame(models_frame)
right_panel.pack(side='right', fill='y', padx=8)

ctk.CTkLabel(left_panel, text='Available Models:').pack(anchor='w', padx=4, pady=4)
model_listbox = tk.Listbox(left_panel, height=10, font=('Sans', 11))
model_listbox.pack(fill='both', expand=True)

def refresh_models_list():
    try:
        model_listbox.delete(0, tk.END)
        for name in list_models():
            model_listbox.insert(tk.END, name)
    except Exception:
        pass

refresh_models_list()

# Left panel now only lists models; all actions live in the sidebar

# Global enable checkbox
recognition_master_var = tk.BooleanVar(value=False)

# Right settings panel
detail_name_var = tk.StringVar(value='')
ttk.Label(right_panel, textvariable=detail_name_var).pack(anchor='w', pady=(4,6))

## CustomTkinter: Switch and buttons

toggle_var = tk.StringVar(value='Off')

def _ensure_selected_name(name: str):
    try:
        for i in range(model_listbox.size()):
            if model_listbox.get(i) == name:
                model_listbox.selection_clear(0, tk.END)
                model_listbox.selection_set(i)
                model_listbox.see(i)
                break
    except Exception:
        pass

def on_toggle_switch(value=None):
    name = detail_name_var.get()
    if not name:
        toggle_var.set('Off')
        status_var.set('Select a model first')
        return
    _ensure_selected_name(name)
    # Toggle using existing logic
    is_on = (value or toggle_var.get()) == 'On'
    if is_on:
        if not (recognition_active and recognition_name == name):
            toggle_recognition()
    else:
        if recognition_active and recognition_name == name:
            toggle_recognition()

enable_row = ttk.Frame(right_panel)
enable_row.pack(anchor='w', pady=6)
ttk.Label(enable_row, text='Enable').pack(side='left', padx=(0,8))
enable_switch = ToggleSwitch(enable_row, value=False, command=lambda v: on_toggle_switch('On' if v else 'Off'))
enable_switch.pack(side='left')

# Action: shows current choice and opens popup
current_action_var = tk.StringVar(value='none')

def open_action_popup():
    name = detail_name_var.get()
    if not name:
        status_var.set('Select a model first')
        return
    win = ctk.CTkToplevel(root)
    win.title('Select Action')
    frm = ttk.Frame(win)
    frm.pack(padx=10, pady=10)
    actions = ['none','takeoff','land','up','down','left','right','forward','back']
    sel = tk.StringVar(value=current_action_var.get())
    for i, a in enumerate(actions):
        ttk.Radiobutton(frm, text=a, value=a, variable=sel).grid(row=i, column=0, sticky='w', pady=2)
    def save_close():
        current_action_var.set(sel.get())
        # persist into model_settings immediately
        try:
            model_settings[name] = {**model_settings.get(name, {}), 'action': sel.get()}
        except Exception:
            model_settings[name] = {'action': sel.get()}
        try:
            action_btn.configure(text=f"Action: {sel.get()}")
        except Exception:
            pass
        win.destroy()
    ctk.CTkButton(frm, text='OK', command=save_close).grid(row=len(actions), column=0, pady=(8,0))

action_btn = BlueButton(right_panel, text='Action: none', command=open_action_popup)
action_btn.pack(anchor='w', pady=4)

apply_btn = BlueButton(right_panel, text='Apply', command=lambda: apply_detail_settings())
apply_btn.pack(anchor='w', pady=8)
update_btn = BlueButton(right_panel, text='Update Model...', command=lambda: open_update_model_popup())
update_btn.pack(anchor='w', pady=2)
new_model_btn = BlueButton(right_panel, text='New Model...', command=lambda: create_new_model())
new_model_btn.pack(anchor='w', pady=2)
delete_btn = BlueButton(right_panel, text='Delete Model', command=lambda: delete_selected_model())
delete_btn.pack(anchor='w', pady=2)

def _select_model_name():
    try:
        idxs = model_listbox.curselection()
        if not idxs:
            return None
        return model_listbox.get(idxs[0])
    except Exception:
        return None

def toggle_recognition():
    global recognition_active, recognition_model_info, recognition_name
    try:
        if recognition_active:
            recognition_active = False
            recognition_model_info = None
            recognition_name = None
        try:
            enable_switch.set(False)
            toggle_var.set('Off')
        except Exception:
            pass
            status_var.set('Recognition stopped')
            return
        name = _select_model_name()
        if not name:
            status_var.set('Select a model first')
            return
        recognition_model_info = load_model(name)
        recognition_name = name
        recognition_active = True
        # reset streak/action
        global recognition_streak_start_ts, recognition_action_done
        recognition_streak_start_ts = 0.0
        recognition_action_done = False
        try:
            enable_switch.set(True)
            toggle_var.set('On')
        except Exception:
            pass
        status_var.set(f'Recognition started: {name}')
    except Exception:
        status_var.set('Recognition error')

 

def open_samples():
    name = _select_model_name()
    if not name:
        status_var.set('Select a model first')
        return
    try:
        paths = find_samples_for_model(name)
        win = ctk.CTkToplevel(root)
        win.title(f'Samples: {name}')
        canvas = tk.Canvas(win)
        canvas.pack(fill='both', expand=True)
        x, y = 10, 10
        for p in paths[:8]:
            try:
                img_bgr = cv2.imread(p)
                if img_bgr is None:
                    continue
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                scale = min(1.0, 200.0 / float(max(w, h)))
                thumb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(thumb))
                lbl = tk.Label(canvas, image=imgtk)
                lbl.image = imgtk
                canvas.create_window(x, y, anchor='nw', window=lbl)
                x += int(w * scale) + 10
                if x > 800:
                    x = 10
                    y += int(h * scale) + 10
            except Exception:
                continue
    except Exception:
        status_var.set('Samples error')

 

def open_update_model_popup():
    name = detail_name_var.get()
    if not name:
        status_var.set('Select a model first')
        return
    win = ctk.CTkToplevel(root)
    win.title(f'Update Model: {name}')
    frm = ttk.Frame(win)
    frm.pack(padx=10, pady=10, fill='both', expand=True)
    img_dir = _model_images_dir(name)
    ttk.Label(frm, text=f'Images folder: {img_dir}').grid(row=0, column=0, columnspan=3, sticky='w', pady=(0,6))
    start_over_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(frm, text='Start over (clear existing samples)', variable=start_over_var).grid(row=1, column=0, columnspan=2, sticky='w')

    def clear_samples():
        try:
            if os.path.isdir(img_dir):
                for fn in os.listdir(img_dir):
                    try:
                        os.remove(os.path.join(img_dir, fn))
                    except Exception:
                        pass
            status_var.set('Samples cleared')
        except Exception:
            status_var.set('Failed to clear samples')

    def capture_once():
        frame = video.get_frame() if video is not None else None
        if frame is None:
            status_var.set('No frame to capture')
            return
        try:
            os.makedirs(img_dir, exist_ok=True)
            fname = f'sample_{int(time.time())}_{random.randint(1000,9999)}.jpg'
            path = os.path.join(img_dir, fname)
            cv2.imwrite(path, frame)
            status_var.set(f'Captured sample to {path}')
        except Exception:
            status_var.set('Capture failed')

    def train_now():
        def _train():
            import subprocess, sys as _sys
            try:
                if start_over_var.get() and os.path.isdir(img_dir):
                    for fn in os.listdir(img_dir):
                        try:
                            os.remove(os.path.join(img_dir, fn))
                        except Exception:
                            pass
                os.makedirs(img_dir, exist_ok=True)
                subprocess.run([_sys.executable, os.path.join(os.path.dirname(__file__), 'face_training', 'train_face_model.py'), img_dir], check=True)
            except Exception:
                pass
            finally:
                root.after(0, refresh_models_list)
        run_async(_train)

    ttk.Button(frm, text='Clear Samples', command=clear_samples).grid(row=1, column=2, sticky='e')
    ttk.Button(frm, text='Capture Photo', command=capture_once).grid(row=2, column=0, pady=6, sticky='w')
    ttk.Button(frm, text='Train Model', command=train_now).grid(row=2, column=1, pady=6, sticky='w')
    ttk.Button(frm, text='Close', command=win.destroy).grid(row=2, column=2, pady=6, sticky='e')

 

def on_model_select_event(evt=None):
    name = _select_model_name()
    if not name:
        return
    detail_name_var.set(name)
    # populate toggle/action from model_settings
    cfg = model_settings.get(name, {})
    try:
        enable_switch.set(bool(recognition_active and (recognition_name == name)))
        toggle_var.set('On' if (recognition_active and (recognition_name == name)) else 'Off')
    except Exception:
        toggle_var.set('Off')
    current_action = cfg.get('action', 'none')
    current_action_var.set(current_action)
    try:
        action_btn.configure(text=f"Action: {current_action}")
    except Exception:
        pass

def apply_detail_settings():
    name = detail_name_var.get()
    if not name:
        return
    model_settings[name] = {**model_settings.get(name, {}), 'action': current_action_var.get()}
    # toggle enable
    if 'enable_switch' in globals() and enable_switch.get():
        # ensure selected
        try:
            idxs = model_listbox.curselection()
            if not idxs:
                # select the row matching name
                for i in range(model_listbox.size()):
                    if model_listbox.get(i) == name:
                        model_listbox.selection_set(i)
                        break
        except Exception:
            pass
        if not recognition_active or recognition_name != name:
            toggle_recognition()
    else:
        if recognition_active and recognition_name == name:
            toggle_recognition()

def delete_selected_model():
    name = detail_name_var.get()
    if not name:
        return
    try:
        base = os.path.join(os.path.dirname(__file__), 'face_training', 'models', name)
        shutil.rmtree(base)
        if recognition_active and recognition_name == name:
            toggle_recognition()
        refresh_models_list()
        detail_name_var.set('')
        status_var.set(f"Deleted model {name}")
    except Exception:
        status_var.set('Delete failed')

def create_new_model():
    win = ctk.CTkToplevel(root)
    win.title('Create New Model')
    frm = ttk.Frame(win)
    frm.pack(padx=10, pady=10)
    ttk.Label(frm, text='Model name:').grid(row=0, column=0, sticky='e', padx=4, pady=4)
    name_var = tk.StringVar(value='')
    name_entry = ttk.Entry(frm, textvariable=name_var, width=20)
    name_entry.grid(row=0, column=1, padx=4, pady=4)

    info_var = tk.StringVar(value='Provide a unique name, then capture photos and train.')
    ttk.Label(frm, textvariable=info_var).grid(row=1, column=0, columnspan=2, sticky='w')

    def _img_dir_for(name: str) -> str:
        return os.path.join(os.path.dirname(__file__), 'face_training', f'{name}_images')

    def capture_new():
        name = name_var.get().strip()
        if not name:
            info_var.set('Enter a model name first')
            return
        frame = video.get_frame() if video is not None else None
        if frame is None:
            info_var.set('No frame to capture')
            return
        img_dir = _img_dir_for(name)
        try:
            os.makedirs(img_dir, exist_ok=True)
            fname = f'sample_{int(time.time())}_{random.randint(1000,9999)}.jpg'
            path = os.path.join(img_dir, fname)
            cv2.imwrite(path, frame)
            info_var.set(f'Captured to {path}')
        except Exception:
            info_var.set('Capture failed')

    def train_new():
        name = name_var.get().strip()
        if not name:
            info_var.set('Enter a model name first')
            return
        img_dir = _img_dir_for(name)
        def _train():
            import subprocess, sys as _sys
            try:
                os.makedirs(img_dir, exist_ok=True)
                subprocess.run([_sys.executable, os.path.join(os.path.dirname(__file__), 'face_training', 'train_face_model.py'), img_dir], check=True)
            except Exception:
                pass
            finally:
                def _after():
                    refresh_models_list()
                    # Select newly created model if present
                    try:
                        for i in range(model_listbox.size()):
                            if model_listbox.get(i) == name:
                                model_listbox.selection_clear(0, tk.END)
                                model_listbox.selection_set(i)
                                model_listbox.see(i)
                                detail_name_var.set(name)
                                break
                    except Exception:
                        pass
                root.after(0, _after)
        run_async(_train)

    ttk.Button(frm, text='Capture Photo', command=capture_new).grid(row=2, column=0, pady=6, sticky='w')
    ttk.Button(frm, text='Train Model', command=train_new).grid(row=2, column=1, pady=6, sticky='w')
    ttk.Button(frm, text='Close', command=win.destroy).grid(row=2, column=2, pady=6, sticky='e')

model_listbox.bind('<<ListboxSelect>>', on_model_select_event)

def _model_images_dir(name: str) -> str:
    # Choose an images dir: use the first samples dir if found, else default <name>_images
    try:
        candidates = find_samples_for_model(name)
        if candidates:
            return os.path.dirname(candidates[0])
    except Exception:
        pass
    return os.path.join(os.path.dirname(__file__), 'face_training', f'{name}_images')

 

def update_freefly_debug():
    try:
        if free_fly_controller is not None:
            info = free_fly_controller.get_debug_info()
            if info:
                debug_var.set(f"mode={info.get('mode')} dc={info.get('dc')} lapVar={info.get('lap_var')} looming={info.get('looming')} div={info.get('divergence')} tof={info.get('tof_cm')}")
            else:
                debug_var.set("")
    except Exception:
        pass
    finally:
        root.after(500, update_freefly_debug)

def ascend():
    try:
        tello.move_up(20)
        status_var.set("Up 20cm")
    except Exception:
        status_var.set("Up error")

def descend():
    try:
        tello.move_down(20)
        status_var.set("Down 20cm")
    except Exception:
        status_var.set("Down error")

## Removed duplicate Up/Down buttons (use the altitude pad instead)

# Recording controls
rec_controls = ttk.Frame(controls)
rec_controls.pack(pady=6)

def start_recording():
    global recording, video_writer
    try:
        if not is_connected or tello is None:
            status_var.set('Not connected')
            return
        os.makedirs(record_dir, exist_ok=True)
        # Determine frame size from latest frame
        if video is None:
            status_var.set('Video not ready')
            return
        frame = video.get_frame()
        if frame is None:
            status_var.set('No frame available')
            return
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fname = f"tello_{int(time.time())}.mp4"
        path = os.path.join(record_dir, fname)
        vw = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
        if not vw.isOpened():
            status_var.set('Recorder open failed')
            return
        video_writer = vw
        recording = True
        rec_var.set('REC')
        status_var.set(f'Recording: {path}')
        try:
            rec_btn.configure(text='Stop Recording')
        except Exception:
            pass
    except Exception:
        status_var.set('Recording start error')

def stop_recording():
    global recording, video_writer
    try:
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass
        video_writer = None
        recording = False
        rec_var.set('')
        try:
            rec_btn.configure(text='Start Recording')
        except Exception:
            pass
        status_var.set('Recording stopped')
    except Exception:
        status_var.set('Recording stop error')

def toggle_recording():
    if recording:
        stop_recording()
    else:
        start_recording()

rec_btn = BlueButton(rec_controls, text='Start Recording', command=lambda: run_async(toggle_recording))
rec_btn.pack()

def update_frame():
    if video is None:
        root.after(100, update_frame)
        return
    frame = video.get_frame()
    if frame is not None:
        # mark last time we saw a good video frame
        try:
            global last_video_ts
            last_video_ts = time.time()
        except Exception:
            pass
        # Resize to fit screen width and available height (leaving room for controls)
        try:
            root.update_idletasks()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            frame_h, frame_w = frame.shape[:2]

            # Reserve space for controls/status/battery; use sensible defaults if 0 early on
            controls_h = controls.winfo_height() or 140
            status_h = status_label.winfo_height() or 24
            battery_h = battery_label.winfo_height() or 20
            reserved_h = controls_h + status_h + battery_h + 24
            max_img_w = screen_w
            max_img_h = max(120, screen_h - reserved_h)

            scale = min(max_img_w / float(frame_w), max_img_h / float(frame_h))
            new_w = max(1, int(frame_w * scale))
            new_h = max(1, int(frame_h * scale))

            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Convert BGR (OpenCV) -> RGB (PIL/Tk). Tello via djitellopy delivers BGR frames.
            def to_rgb(img):
                try:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    return img

            # Optional overlay: face recognition bbox
            draw_bgr = resized.copy()
            if recognition_active and recognition_model_info is not None:
                try:
                    ok, bbox, name, metric = recognize_in_frame(recognition_model_info, frame)
                    if ok and bbox is not None:
                        x, y, w2, h2 = bbox
                        # scale bbox to resized
                        sx = new_w / float(frame_w)
                        sy = new_h / float(frame_h)
                        rx = int(x * sx)
                        ry = int(y * sy)
                        rw = int(w2 * sx)
                        rh = int(h2 * sy)
                        cv2.rectangle(draw_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                        label = recognition_name or name or 'face'
                        # Large, centered name across the box with outline for readability
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = max(0.8, min(3.0, rw / 160.0))
                        thickness = max(2, int(scale * 2))
                        (text_w, text_h), _ = cv2.getTextSize(label, font, scale, thickness)
                        tx = rx + max(0, (rw - text_w) // 2)
                        ty = ry + (rh // 2) + (text_h // 2)
                        # outline
                        cv2.putText(draw_bgr, label, (tx, ty), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                        # fill
                        cv2.putText(draw_bgr, label, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
                        # update last seen for follow
                        global recognition_last_bbox, recognition_last_seen_ts
                        recognition_last_bbox = (x + w2 // 2, y + h2 // 2, w2, h2)
                        recognition_last_seen_ts = time.time()
                        # start streak if not started
                        global recognition_streak_start_ts
                        if recognition_streak_start_ts == 0.0:
                            recognition_streak_start_ts = recognition_last_seen_ts
                    else:
                        # reset streak if lost
                        recognition_streak_start_ts = 0.0
                except Exception:
                    pass

            rgb = to_rgb(draw_bgr)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.configure(image=imgtk)
            video_label.image = imgtk
            # Write to recorder if enabled (use original frame size/colors)
            try:
                if recording and video_writer is not None:
                    video_writer.write(frame)
            except Exception:
                pass
        except Exception:
            # Fallback to raw frame on error
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.configure(image=imgtk)
            video_label.image = imgtk
    root.after(30, update_frame)

def on_close():
    # Clear UI and show closing overlay immediately
    try:
        _show_overlay('Closing...')
    except Exception:
        pass

    def _shutdown():
        try:
            try:
                if 'free_fly_controller' in globals() and free_fly_controller is not None:
                    free_fly_controller.stop()
            except Exception:
                pass
            # Stop recording if active
            try:
                if 'recording' in globals() and recording:
                    stop_recording()
            except Exception:
                pass
            # Best-effort land only if connected and flying
            if is_flying and is_connected and tello is not None:
                try:
                    status_var.set('Landing...')
                except Exception:
                    pass
                try:
                    tello.land()
                    time.sleep(1)
                except Exception:
                    pass
            # Stop video
            try:
                if video is not None:
                    video.stop()
            except Exception:
                pass
        except Exception:
            pass
        finally:
            try:
                if tello is not None:
                    tello.end()
            except Exception:
                pass
            try:
                root.destroy()
            except Exception:
                pass

    import threading
    threading.Thread(target=_shutdown, daemon=True).start()

def on_key_press(event):
    pressed.add(event.keysym)

def on_key_release(event):
    pressed.discard(event.keysym)

def update_rc():
    if free_fly_active or recognition_active:
        root.after(100, update_rc)
        return
    # Map arrows only: Left/Right -> yaw, Up/Down -> forward/back
    lr = 0
    fb = 0
    ud = 0
    yw = 0

    # Yaw on arrows
    if 'Left' in pressed:
        yw = -SPEED
    if 'Right' in pressed:
        yw = SPEED

    # Forward/back on Up/Down and W/S
    if 'Up' in pressed or 'w' in pressed or 'W' in pressed:
        fb = SPEED
    if 'Down' in pressed or 's' in pressed or 'S' in pressed:
        fb = -SPEED

    # Strafe on A/D
    if 'a' in pressed or 'A' in pressed:
        lr = -SPEED
    if 'd' in pressed or 'D' in pressed:
        lr = SPEED

    try:
        tello.send_rc_control(lr, fb, ud, yw)
    except Exception:
        pass
    root.after(100, update_rc)

def update_battery():
    try:
        if is_connected and tello is not None:
            batt = tello.get_battery()
            battery_var.set(f"Battery: {batt}%")
        else:
            battery_var.set("Battery: -")
    except Exception:
        pass
    finally:
        root.after(60000, update_battery)

def update_stats():
    try:
        if is_connected and tello is not None:
            # Height (prefer ToF if available)
            try:
                h = None
                if hasattr(tello, 'get_distance_tof'):
                    try:
                        h = int(tello.get_distance_tof())
                    except Exception:
                        h = None
                if h is None:
                    h = int(tello.get_height())
                height_var.set(f"H: {h} cm")
            except Exception:
                height_var.set("H: - cm")

            # Attitude (pitch/roll/yaw)
            try:
                p = int(tello.get_pitch())
                r = int(tello.get_roll())
                y = int(tello.get_yaw())
                tilt_var.set(f"Att: P {p}° R {r}° Y {y}°")
            except Exception:
                tilt_var.set("Att: P -° R -° Y -°")
        else:
            height_var.set("H: - cm")
            tilt_var.set("Att: P -° R -° Y -°")
    except Exception:
        pass
    finally:
        root.after(1000, update_stats)

def update_face_follow():
    try:
        if recognition_active and is_connected and tello is not None:
            now = time.time()
            # Check recognition streak (consider recent detection within 0.4s)
            recent = (now - recognition_last_seen_ts) < 0.4
            name = recognition_name or ''
            # Immediate/very short hold per requirements
            hold_s = 0.2
            action = model_settings.get(name, {}).get('action', 'none') if name else 'none'
            global recognition_action_done, recognition_streak_start_ts
            if recent:
                if recognition_streak_start_ts == 0.0:
                    recognition_streak_start_ts = now
                # Trigger action once when held long enough
                if (not recognition_action_done) and (now - recognition_streak_start_ts >= hold_s) and action != 'none':
                    def do_action():
                        try:
                            if action == 'takeoff' and not is_flying:
                                takeoff()
                            elif action == 'land' and is_flying:
                                land()
                            elif action == 'up':
                                tello.move_up(20)
                            elif action == 'down':
                                tello.move_down(20)
                        except Exception:
                            pass
                    run_async(do_action)
                    recognition_action_done = True
            else:
                recognition_streak_start_ts = 0.0
                recognition_action_done = False

            # Follow only if we have a bbox
            if recognition_last_bbox is not None:
                cx, cy, bw, bh = recognition_last_bbox
                # desired center is frame center; compute error
                frame = video.get_frame() if video is not None else None
                if frame is not None:
                    fh, fw = frame.shape[:2]
                    ex = (cx - fw / 2.0) / (fw / 2.0)
                    ey = (cy - fh / 2.0) / (fh / 2.0)
                    # simple P controller
                    yaw = int(np.clip(ex * 60, -50, 50)) if 'np' in globals() else int(ex * 60)
                    fb = int(np.clip((0.4 - (bw / float(fw))) * 80, -40, 40)) if 'np' in globals() else int((0.4 - (bw / float(fw))) * 80)
                    lr = 0
                    ud = int(np.clip(-ey * 40, -30, 30)) if 'np' in globals() else int(-ey * 40)
                    try:
                        tello.send_rc_control(lr, fb, ud, yaw)
                    except Exception:
                        pass
        else:
            # stop movement if not tracking
            try:
                if is_connected and tello is not None:
                    tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass
    except Exception:
        pass
    finally:
        root.after(150, update_face_follow)

root.protocol("WM_DELETE_WINDOW", on_close)
root.bind('<KeyPress>', on_key_press)
root.bind('<KeyRelease>', on_key_release)
root.bind('q', lambda e: on_close())
root.bind('Q', lambda e: on_close())
root.focus_set()
update_frame()
update_rc()
update_battery()
update_freefly_debug()
update_stats()
update_face_follow()
# Connectivity overlay and checks (CustomTkinter)
overlay = ctk.CTkFrame(root)
overlay_label = ctk.CTkLabel(overlay, text='Connecting...')
overlay_btn = ctk.CTkButton(overlay, text='Reconnect')
overlay_label.pack(pady=6)
overlay_btn.pack()
_overlay_visible = False

def _show_overlay(msg='Disconnected'):
    global _overlay_visible
    try:
        overlay_label.configure(text=msg)
        if not _overlay_visible:
            overlay.place(relx=0.5, rely=0.5, anchor='center')
            header.pack_forget()
            tabs.pack_forget()
            video_label.pack_forget()
            _overlay_visible = True
    except Exception:
        pass

def _hide_overlay():
    global _overlay_visible
    try:
        if _overlay_visible:
            overlay.place_forget()
            header.pack(fill='x', pady=6)
            tabs.pack(fill='x', pady=4)
            video_label.pack(fill='both', expand=True)
            _overlay_visible = False
    except Exception:
        pass

def attempt_connect():
    global tello, is_connected, video, video_started
    try:
        if tello is None:
            try:
                tello = Tello()
            except OSError as e:
                if getattr(e, 'errno', None) == 98:
                    ui_status('UDP ports busy (8889/8890/11111)')
                    is_connected = False
                    return
                raise
        # Faster retries: set a short response timeout if available
        try:
            setattr(tello, 'RESPONSE_TIMEOUT', 3)
        except Exception:
            pass
        connected = False
        ui_status('Connecting to Tello...')
        for _ in range(2):
            try:
                tello.connect()
                time.sleep(1)
                connected = True
                break
            except Exception:
                time.sleep(0.5)
        if not connected:
            is_connected = False
            ui_status('Connect failed (timeout)')
            return
        is_connected = True
        ui_status('Connected')
        if video is None:
            video = TelloVideo(tello)
        try:
            video.start()
            video_started = True
        except Exception:
            video_started = False
        _hide_overlay()
    except Exception:
        is_connected = False
        ui_status('Connect error')

overlay_btn.configure(command=lambda b=overlay_btn: run_async(attempt_connect, b))
_show_overlay('Connecting...')
run_async(attempt_connect)

# Startup grace window (~7s) to verify connection before showing disconnected
startup_grace_until = time.time() + 7.0

def _startup_finalize():
    try:
        if not is_connected:
            _show_overlay('Disconnected')
    except Exception:
        pass

root.after(7000, _startup_finalize)

def periodic_check():
    global is_connected, video_started
    # Debounce/failure counting to avoid overlay flicker
    if '_health_failures' not in globals():
        globals()['_health_failures'] = 0
    try:
        now = time.time()
        video_recent = False
        try:
            video_recent = (now - (last_video_ts or 0.0)) < 8.0
        except Exception:
            video_recent = False

        health_ok = False
        if tello is not None:
            try:
                _ = tello.get_battery()
                health_ok = True
            except Exception:
                health_ok = False

        # If video is recent, consider link alive even if battery poll failed
        if video_recent:
            health_ok = True

        if health_ok:
            is_connected = True
            globals()['_health_failures'] = 0
        else:
            globals()['_health_failures'] = globals().get('_health_failures', 0) + 1
            # Require more consecutive failures before dropping connection
            drop_threshold = 6
            if globals()['_health_failures'] >= drop_threshold:
                is_connected = False
            # else keep previous state (avoid flicker)

        if not is_connected:
            # Honor startup grace period: keep "Connecting..." for ~7 seconds
            try:
                grace_until = globals().get('startup_grace_until', 0.0)
            except Exception:
                grace_until = 0.0
            if time.time() < grace_until:
                _show_overlay('Connecting...')
                return
            try:
                if video is not None:
                    video.stop()
            except Exception:
                pass
            video_started = False
            _show_overlay('Disconnected')
        else:
            if not video_started and video is not None:
                try:
                    video.start()
                    video_started = True
                except Exception:
                    pass
            _hide_overlay()
    except Exception:
        pass
    finally:
        root.after(5000, periodic_check)

periodic_check()
root.mainloop()
