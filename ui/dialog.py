import tkinter as tk
from typing import Callable, Sequence
from .theme import Theme
from .shapes import draw_rounded_rect, draw_shadow


class Dialog:
    def __init__(self, parent: tk.Tk | tk.Toplevel, title: str = 'Dialog', message: str = '', buttons: Sequence[str] = ('OK',), theme: Theme | None = None):
        self._parent = parent
        self._theme = theme or Theme()
        self._title = title
        self._message = message
        self._buttons = list(buttons)
        self._result: str | None = None
        self._on_close: Callable[[str | None], None] | None = None

        self._win = tk.Toplevel(parent)
        self._win.withdraw()
        self._win.title(title)
        self._win.configure(bg=self._theme.bg)
        self._win.overrideredirect(True)
        self._win.transient(parent)
        self._win.grab_set()

        self._canvas = tk.Canvas(self._win, bg=self._theme.bg, highlightthickness=0)
        self._canvas.pack(fill='both', expand=True)
        self._canvas.bind('<Configure>', lambda e: self._redraw())

        self._btn_frames: list[tk.Frame] = []

    def show(self, on_close: Callable[[str | None], None] | None = None) -> str | None:
        self._on_close = on_close
        self._position_center()
        self._win.deiconify()
        try:
            self._parent.wait_window(self._win)
        except Exception:
            pass
        return self._result

    def close(self, result: str | None = None):
        self._result = result
        try:
            self._win.grab_release()
        except Exception:
            pass
        self._win.destroy()
        if self._on_close:
            try: self._on_close(self._result)
            except Exception: pass

    def _position_center(self):
        self._win.update_idletasks()
        sw = self._parent.winfo_screenwidth()
        sh = self._parent.winfo_screenheight()
        w = min(520, int(sw * 0.5))
        h = min(260, int(sh * 0.3))
        x = (sw - w) // 2
        y = (sh - h) // 3
        self._win.geometry(f"{w}x{h}+{x}+{y}")

    def _redraw(self):
        c = self._canvas
        c.delete('all')
        w = c.winfo_width() or 480
        h = c.winfo_height() or 240

        # Shadow + card
        draw_shadow(c, 16 + 3, 16 + 4, w - 16 + 3, h - 16 + 4, 18, self._theme.shadow)
        draw_rounded_rect(c, 16, 16, w - 16, h - 16, 18, fill=self._theme.surface, outline=self._theme.border)

        # Title
        title_font = (self._theme.font_family, self._theme.font_size_lg + 2, 'bold')
        c.create_text(32, 42, text=self._title, fill=self._theme.text, font=title_font, anchor='w')

        # Message (wrap)
        msg_font = (self._theme.font_family, self._theme.font_size)
        c.create_text(32, 80, text=self._message, fill=self._theme.text_muted, font=msg_font, anchor='nw', width=w - 64)

        # Buttons row
        # Clear previous embedded windows
        for bf in self._btn_frames:
            try: bf.destroy()
            except Exception: pass
        self._btn_frames.clear()

        bw = 120
        bh = 42
        gap = 10
        total = len(self._buttons) * bw + (len(self._buttons) - 1) * gap
        start_x = (w - total) // 2
        y = h - 24 - bh
        for i, label in enumerate(self._buttons):
            frame = tk.Frame(self._win, bg=self._theme.bg)
            self._btn_frames.append(frame)
            c.create_window(start_x + i * (bw + gap), y, anchor='nw', window=frame, width=bw, height=bh)
            from .button import BlueButton
            btn = BlueButton(frame, text=label, width=bw, height=bh, command=lambda l=label: self.close(l))
            btn.pack(fill='both', expand=True)


