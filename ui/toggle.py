import tkinter as tk
from .theme import Theme
from .shapes import draw_rounded_rect


class ToggleSwitch(tk.Frame):
    def __init__(self, master, value: bool = False, command=None, width: int | None = None, height: int | None = None, theme: Theme | None = None, **kwargs):
        super().__init__(master, **kwargs)
        self._theme = theme or Theme()
        if height is None:
            height = 38
        if width is None:
            width = int(height * 1.9)
        self._width_px = int(width)
        self._height_px = int(height)
        self._command = command
        self._value = value
        self._animating = False
        self.configure(bg=self._theme.bg, highlightthickness=0)

        self._canvas = tk.Canvas(self, width=self._width_px, height=self._height_px, bg=self._theme.bg, highlightthickness=0, bd=0, cursor='hand2')
        self._canvas.pack(fill='both', expand=True)
        self._canvas.bind('<Button-1>', lambda e: self.toggle())
        self._canvas.bind('<Configure>', lambda e: self._redraw())

        self._redraw()

    def get(self) -> bool:
        return self._value

    def set(self, v: bool, fire: bool = False):
        self._value = bool(v)
        self._redraw()
        if fire and self._command is not None:
            try: self._command(self._value)
            except Exception: pass

    def toggle(self):
        self._value = not self._value
        self._animate_thumb()
        if self._command is not None:
            try: self._command(self._value)
            except Exception: pass

    def configure(self, **kwargs):  # type: ignore[override]
        if 'command' in kwargs:
            self._command = kwargs.pop('command')
        if 'width' in kwargs:
            try:
                self._width_px = int(kwargs.pop('width'))
            except Exception:
                kwargs.pop('width', None)
        if 'height' in kwargs:
            try:
                self._height_px = int(kwargs.pop('height'))
            except Exception:
                kwargs.pop('height', None)
        super().configure(**kwargs)
        if hasattr(self, '_canvas'):
            self._redraw()

    config = configure

    def _redraw(self):
        c = self._canvas
        c.delete('all')
        # Use current canvas size if available; otherwise fallback to preferred
        cw = c.winfo_width()
        ch = c.winfo_height()
        if cw <= 2 or ch <= 2:
            w = max(60, int(self._width_px))
            h = max(32, int(self._height_px))
            c.configure(width=w, height=h)
        else:
            w, h = cw, ch

        track_r = h // 2
        on_col = self._theme.accent
        off_col = '#cbd5e1'
        track_col = on_col if self._value else off_col
        draw_rounded_rect(c, 2, 2, w - 2, h - 2, track_r, fill=track_col, outline='')

        # Thumb
        pad = 4
        thumb_d = h - pad * 2
        if not hasattr(self, '_thumb_x'):
            self._thumb_x = pad if not self._value else (w - pad - thumb_d)
        y0 = pad
        y1 = pad + thumb_d
        x0 = int(self._thumb_x)
        x1 = int(self._thumb_x + thumb_d)
        draw_rounded_rect(c, x0, y0, x1, y1, thumb_d // 2, fill='white', outline='')

        # Labels
        font = (self._theme.font_family, self._theme.font_size, 'bold')
        if self._value:
            c.create_text(w - h, h // 2, text='ON', fill='white', font=font)
        else:
            c.create_text(h, h // 2, text='OFF', fill='#475569', font=font)

    def _animate_thumb(self):
        if self._animating:
            return
        self._animating = True
        c = self._canvas
        w = int(c['width'])
        h = int(c['height'])
        pad = 4
        thumb_d = h - pad * 2
        start = self._thumb_x if hasattr(self, '_thumb_x') else (pad if not self._value else (w - pad - thumb_d))
        end = (w - pad - thumb_d) if self._value else pad
        steps = 8
        dx = (end - start) / float(steps)

        def step(i=0, x=start):
            self._thumb_x = x
            self._redraw()
            if i < steps:
                self.after(12, lambda: step(i + 1, x + dx))
            else:
                self._thumb_x = end
                self._redraw()
                self._animating = False

        step()


