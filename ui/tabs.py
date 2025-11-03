import tkinter as tk
from .theme import Theme
from .shapes import draw_rounded_rect


class Tabs(tk.Frame):
    def __init__(self, master, theme: Theme | None = None, **kwargs):
        super().__init__(master, **kwargs)
        self._theme = theme or Theme()
        self.configure(bg=self._theme.bg)
        self._tabs: dict[str, tk.Frame] = {}
        self._order: list[str] = []
        self._active: str | None = None

        self._bar = tk.Canvas(self, height=52, bg=self._theme.bg, highlightthickness=0)
        self._bar.pack(fill='x', side='top')
        self._bar.bind('<Configure>', lambda e: self._redraw_bar())
        self._bar.bind('<Button-1>', self._on_bar_click)

        self._body = tk.Frame(self, bg=self._theme.bg)
        self._body.pack(fill='both', expand=True)

    def add(self, name: str) -> tk.Frame:
        if name in self._tabs:
            return self._tabs[name]
        holder = tk.Frame(self._body, bg=self._theme.bg)
        self._tabs[name] = holder
        self._order.append(name)
        if self._active is None:
            self._active = name
            holder.pack(fill='both', expand=True)
        self._redraw_bar()
        return holder

    def set_active(self, name: str):
        if name == self._active or name not in self._tabs:
            return
        if self._active:
            try:
                self._tabs[self._active].pack_forget()
            except Exception:
                pass
        self._active = name
        self._tabs[name].pack(fill='both', expand=True)
        self._redraw_bar()

    def _on_bar_click(self, evt):
        x = evt.x
        for item in getattr(self, '_tab_hitboxes', []):
            name, x0, x1 = item
            if x0 <= x <= x1:
                self.set_active(name)
                break

    def _redraw_bar(self):
        c = self._bar
        c.delete('all')
        w = int(c.winfo_width() or 400)
        h = int(c['height'])

        # Background surface
        draw_rounded_rect(c, 6, 8, w - 6, h - 6, self._theme.radius_md, fill=self._theme.surface, outline='')

        # Layout tabs as pills
        x = 18
        y0 = 12
        y1 = h - 10
        gap = 8
        self._tab_hitboxes = []
        font = (self._theme.font_family, self._theme.font_size_lg, 'bold')
        for name in self._order:
            text_id = c.create_text(0, 0, text=name, font=font)
            bbox = c.bbox(text_id)
            c.delete(text_id)
            tw = (bbox[2] - bbox[0]) if bbox else 60
            pad = 18
            x0 = x
            x1 = x + tw + pad * 2
            active = (name == self._active)
            fill = self._theme.primary if active else '#e6eef9'
            text_col = 'white' if active else self._theme.text_muted
            draw_rounded_rect(c, x0, y0, x1, y1, self._theme.radius_md, fill=fill, outline='')
            c.create_text((x0 + x1) // 2, (y0 + y1) // 2, text=name, fill=text_col, font=font)
            self._tab_hitboxes.append((name, x0, x1))
            x = x1 + gap


