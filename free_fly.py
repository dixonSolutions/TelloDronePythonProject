import threading
import time
from typing import Optional

import cv2
import numpy as np


# Tunables (conservative defaults)
EDGE_LOW = 28.0       # edge density below which we consider the path relatively free
EDGE_HIGH = 42.0      # edge density above which we consider it cluttered
EDGE_BLOCK = 55.0     # strong block threshold for immediate stop/evade
FLOW_LOOMING = 2.2    # optical flow magnitude threshold for fast approach
FLOW_DIVERGENCE = 0.30  # radial flow divergence indicating approaching surface
MOVE_FB = 14          # forward/backward RC speed (reduced for safety)
MOVE_UD = 18          # vertical RC speed
MOVE_YW = 28          # yaw RC speed
NUDGE_T = 0.45        # default nudge duration (seconds)
SCAN_STEP_T = 0.35    # per-scan yaw step duration
VIDEO_STALL_MS = 600  # consider video stalled if no frame for this many ms
FORWARD_CLEAR_FRAMES = 4  # require N consecutive clear frames before moving forward
TEXTURE_MIN = 25.0    # variance of Laplacian below this -> treat center as risky/unknown


class FreeFlyController:
    def __init__(self, tello, video, status_var=None):
        self.tello = tello
        self.video = video
        self.status_var = status_var
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._debug_info = {}
        self._dbg_lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if self.status_var is not None:
            try:
                self.status_var.set("Free Fly: ON")
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=1.0)
        if self.status_var is not None:
            try:
                self.status_var.set("Free Fly: OFF")
            except Exception:
                pass

    def _loop(self) -> None:
        prev_gray = None
        last_altitude_check_ts = 0.0
        last_frame_ts = 0.0
        ema = None  # exponential moving average for sector densities
        ema_alpha = 0.5
        scan_dir = 1  # 1 clockwise, -1 counter-clockwise
        trap_counter = 0
        mode = "scan"  # modes: scan, move, evade
        forward_clear_count = 0
        last_debug_print_ts = 0.0
        while self._running:
            # Battery fail-safe
            try:
                batt = self.tello.get_battery()
                if isinstance(batt, int) and batt < 15:
                    if self.status_var is not None:
                        try:
                            self.status_var.set("Battery low. Stopping Free Fly.")
                        except Exception:
                            pass
                    self.stop()
                    break
            except Exception:
                pass

            # Maintain conservative altitude (60-120cm) using height or ToF if available
            now = time.time()
            if now - last_altitude_check_ts > 2.0:
                last_altitude_check_ts = now
                try:
                    h = None
                    if hasattr(self.tello, 'get_distance_tof'):
                        try:
                            # ToF returns cm from ground on some models
                            h = int(self.tello.get_distance_tof())
                        except Exception:
                            h = None
                    if h is None:
                        h = int(self.tello.get_height())
                    if h < 60:
                        self._nudge(ud=+MOVE_UD, duration=0.3)
                    elif h > 120:
                        self._nudge(ud=-MOVE_UD, duration=0.3)
                except Exception:
                    pass

            frame = self.video.get_frame()
            if frame is None:
                # Video stall detection / gentle yaw search
                if (time.time() - last_frame_ts) * 1000.0 > VIDEO_STALL_MS:
                    self._nudge(yw=scan_dir * MOVE_YW, duration=0.4)
                time.sleep(0.05)
                continue
            last_frame_ts = time.time()

            # Compute simple edge density in regions to infer open space
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(gray, 60, 120)
                h, w = edges.shape[:2]

                # Regions: five vertical sectors for finer steering
                sect = max(1, w // 5)
                s0 = edges[:, 0:sect]
                s1 = edges[:, sect:2 * sect]
                s2 = edges[:, 2 * sect:3 * sect]  # center
                s3 = edges[:, 3 * sect:4 * sect]
                s4 = edges[:, 4 * sect:w]

                # Up/Down regions for vertical decisions
                quarter = max(1, h // 4)
                up = edges[0:quarter, 2 * sect:3 * sect]
                down = edges[h - quarter:h, 2 * sect:3 * sect]

                dens = [s0.mean(), s1.mean(), s2.mean(), s3.mean(), s4.mean()]
                dens_up = up.mean()
                dens_down = down.mean()

                # Central texture measure (color/texture invariant)
                ch0 = h // 4
                ch1 = 3 * h // 4
                cw0 = w // 4
                cw1 = 3 * w // 4
                center_roi = gray[ch0:ch1, cw0:cw1]
                lap_var = cv2.Laplacian(center_roi, cv2.CV_64F).var()

                # Smooth densities to avoid twitchy behavior
                if ema is None:
                    ema = dens[:]
                else:
                    ema = [ema[i] * (1 - ema_alpha) + dens[i] * ema_alpha for i in range(5)]
                d0, d1, dc, d3, d4 = ema

                # Basic optical flow "looming" check to detect fast approach
                looming = False
                try:
                    if prev_gray is not None and prev_gray.shape == gray.shape:
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        # central area magnitude
                        central_mag = mag[ch0:ch1, cw0:cw1].mean()
                        looming = central_mag > FLOW_LOOMING

                        # Radial flow divergence: flow aligned with outward vectors indicates approach
                        cy = (ch0 + ch1) // 2
                        cx = (cw0 + cw1) // 2
                        yy, xx = np.mgrid[ch0:ch1, cw0:cw1]
                        rx = (xx - cx).astype(np.float32)
                        ry = (yy - cy).astype(np.float32)
                        norm = np.sqrt(rx * rx + ry * ry) + 1e-6
                        rx /= norm
                        ry /= norm
                        u = flow[ch0:ch1, cw0:cw1, 0]
                        v = flow[ch0:ch1, cw0:cw1, 1]
                        divergence = (u * rx + v * ry).mean()
                        if divergence > FLOW_DIVERGENCE:
                            looming = True
                except Exception:
                    looming = False
                prev_gray = gray

                # If forward is cluttered compared to up/down, try altitude change first
                vertical_choice = None
                if dc > EDGE_HIGH:
                    vertical_choice = "up" if dens_up < dens_down else "down"

                # Trap detection: high density everywhere and/or looming
                heavily_cluttered = dc > EDGE_HIGH and d1 > EDGE_HIGH and d3 > EDGE_HIGH
                # Hard front block: very high central density OR extremely low central texture
                hard_block = (dc > EDGE_BLOCK) or (lap_var < TEXTURE_MIN)
                if looming or heavily_cluttered or hard_block:
                    mode = "evade"

                if mode == "evade":
                    # Back off and yaw to search new path
                    self._nudge(fb=-MOVE_FB, duration=0.5)
                    self._nudge(yw=scan_dir * MOVE_YW, duration=0.6)
                    trap_counter += 1
                    if trap_counter % 3 == 0:
                        # every few evades, try small altitude change
                        self._nudge(ud=(+MOVE_UD if dens_up < dens_down else -MOVE_UD), duration=0.35)
                    mode = "scan"
                    continue

                if vertical_choice is not None:
                    self._nudge(ud=(+MOVE_UD if vertical_choice == "up" else -MOVE_UD), duration=0.35)
                    mode = "scan"
                    continue

                # Scan mode: rotate in the direction of the clearer side
                if mode == "scan":
                    left_cost = (d0 + d1) / 2.0
                    right_cost = (d3 + d4) / 2.0
                    if left_cost < right_cost:
                        scan_dir = -1
                    else:
                        scan_dir = 1
                    # If center is sufficiently clear for several frames, switch to move
                    if dc < EDGE_LOW:
                        forward_clear_count += 1
                        if forward_clear_count >= FORWARD_CLEAR_FRAMES:
                            mode = "move"
                    else:
                        forward_clear_count = 0
                        self._nudge(yw=scan_dir * MOVE_YW, duration=SCAN_STEP_T)
                        continue

                # Move mode: small forward steps with continuous monitoring
                if mode == "move":
                    if dc >= EDGE_LOW:
                        mode = "scan"
                        forward_clear_count = 0
                        continue
                    # bias slight yaw toward clearer side while moving
                    yaw_bias = 0
                    if d1 < d3:
                        yaw_bias = -int(MOVE_YW * 0.5)
                    elif d3 < d1:
                        yaw_bias = int(MOVE_YW * 0.5)
                    self._nudge(fb=+MOVE_FB, yw=yaw_bias, duration=NUDGE_T)
                    # After a nudge, reassess; remain in move if still clear
                    continue

                # Small idle to allow video update
                time.sleep(0.05)

            except Exception:
                # On vision error, perform a gentle yaw search
                self._nudge(yw=scan_dir * MOVE_YW, duration=0.4)
                time.sleep(0.05)

            # Publish debug info and occasionally print
            try:
                dbg = {
                    'mode': mode,
                    'dc': round(float(dc) if 'dc' in locals() else -1, 2),
                    'lap_var': round(float(lap_var) if 'lap_var' in locals() else -1, 2),
                    'looming': bool(looming) if 'looming' in locals() else False,
                    'divergence': round(float(divergence) if 'divergence' in locals() else 0.0, 3),
                }
                # ToF / height
                try:
                    tof = None
                    if hasattr(self.tello, 'get_distance_tof'):
                        tof = int(self.tello.get_distance_tof())
                    dbg['tof_cm'] = tof
                except Exception:
                    dbg['tof_cm'] = None
                with self._dbg_lock:
                    self._debug_info = dbg
                if time.time() - last_debug_print_ts > 1.0:
                    last_debug_print_ts = time.time()
                    print(f"[FreeFly] mode={dbg['mode']} dc={dbg['dc']} lapVar={dbg['lap_var']} looming={dbg['looming']} div={dbg['divergence']} tof={dbg['tof_cm']}")
            except Exception:
                pass

        # Ensure we stop movement when leaving
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass

    def _nudge(self, lr: int = 0, fb: int = 0, ud: int = 0, yw: int = 0, duration: float = 0.4):
        # Sends a small RC command for a duration while allowing quick stop
        t_end = time.time() + duration
        while self._running and time.time() < t_end:
            try:
                self.tello.send_rc_control(lr, fb, ud, yw)
            except Exception:
                pass
            time.sleep(0.05)
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass

    def get_debug_info(self):
        try:
            with self._dbg_lock:
                return dict(self._debug_info)
        except Exception:
            return {}


