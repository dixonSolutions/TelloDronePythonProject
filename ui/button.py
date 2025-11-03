import tkinter as tk
import tkinter.font as tkfont
from .theme import Theme
from .shapes import draw_rounded_rect, draw_shadow


class BlueButton(tk.Frame):
    def __init__(self, master, text: str = 'Button', command=None, width: int | None = None, height: int | None = None, theme: Theme | None = None, **kwargs):
        super().__init__(master, **kwargs)
        self._theme = theme or Theme()
        self._text = text
        self._command = command
        self._width_px = width if isinstance(width, int) else None
        self._height_px = height if isinstance(height, int) else None
        self._state_disabled = False
        self.configure(bg=self._theme.bg, highlightthickness=0)

        self._canvas = tk.Canvas(self, bg=self._theme.bg, highlightthickness=0, bd=0)
        self._canvas.pack(fill='both', expand=True)

        self._hover = False
        self._pressed = False

        self._canvas.bind('<Enter>', lambda e: self._set_hover(True))
        self._canvas.bind('<Leave>', lambda e: self._set_hover(False))
        self._canvas.bind('<ButtonPress-1>', self._on_press)
        self._canvas.bind('<ButtonRelease-1>', self._on_release)
        self._canvas.bind('<Configure>', lambda e: self._redraw())

        self._redraw()

    def configure(self, **kwargs):  # type: ignore[override]
        if 'text' in kwargs:
            self._text = kwargs.pop('text')
        if 'state' in kwargs:
            st = kwargs.pop('state')
            self._state_disabled = (st == 'disabled')
        if 'command' in kwargs:
            self._command = kwargs.pop('command')
        # Allow external width/height updates to control preferred size
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

    def _set_hover(self, hv: bool):
        self._hover = hv
        self._redraw()

    def _on_press(self, _):
        if self._state_disabled:
            return
        self._pressed = True
        self._redraw()

    def _on_release(self, _):
        was_pressed = self._pressed
        self._pressed = False
        self._redraw()
        if was_pressed and not self._state_disabled and self._command is not None:
            try:
                self._command()
            except Exception:
                pass

    def _redraw(self):
        c = self._canvas
        c.delete('all')
        # Determine target size
        cw = c.winfo_width()
        ch = c.winfo_height()

        def _measure():
            font = tkfont.Font(family=self._theme.font_family, size=self._theme.font_size_lg, weight='bold')
            tw = font.measure(self._text)
            th = font.metrics('linespace')
            pad_x = 24
            pad_y = 12
            return max(40, tw + pad_x * 2), max(28, th + pad_y * 2)

        if (cw <= 2 or ch <= 2):
            # initial sizing or collapsed; set a reasonable request size
            if self._width_px is not None and self._height_px is not None:
                w, h = int(self._width_px), int(self._height_px)
            else:
                w, h = _measure()
            c.configure(width=w, height=h, bg=self._theme.bg)
        else:
            w, h = cw, ch

        # Shadow
        sx = 3
        sy = 4
        draw_shadow(c, 6 + sx, 6 + sy, w - 6 + sx, h - 6 + sy, self._theme.radius_lg, self._theme.shadow)

        # Background
        if self._state_disabled:
            bg = '#96b4e0'
        elif self._pressed:
            bg = self._theme.primary_active
        elif self._hover:
            bg = self._theme.primary_hover
        else:
            bg = self._theme.primary
        draw_rounded_rect(c, 6, 6, w - 6, h - 6, self._theme.radius_lg, fill=bg, outline='')

        # Text
        font = (self._theme.font_family, self._theme.font_size_lg, 'bold')
        c.create_text(w // 2, h // 2, text=self._text, fill='white', font=font)


