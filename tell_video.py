import threading
import time
from typing import Optional

class TelloVideo:
    def __init__(self, tello):
        self.tello = tello
        self.frame_read = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        try:
            self.tello.streamon()
        except Exception:
            pass
        self.frame_read = self.tello.get_frame_read()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            frame = getattr(self.frame_read, 'frame', None)
            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
            time.sleep(1 / 30)

    def get_frame(self):
        with self._lock:
            return self._latest_frame

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.tello.streamoff()
        except Exception:
            pass
