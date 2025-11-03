import tkinter as tk


def draw_rounded_rect(canvas: tk.Canvas, x0: int, y0: int, x1: int, y1: int, r: int, fill: str, outline: str = '', width: int = 1):
    r = max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    if r == 0:
        return canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=outline, width=width)
    items = []
    items.append(canvas.create_arc(x0, y0, x0 + 2 * r, y0 + 2 * r, start=90, extent=90, style='pieslice', outline=outline, width=width, fill=fill))
    items.append(canvas.create_arc(x1 - 2 * r, y0, x1, y0 + 2 * r, start=0, extent=90, style='pieslice', outline=outline, width=width, fill=fill))
    items.append(canvas.create_arc(x1 - 2 * r, y1 - 2 * r, x1, y1, start=270, extent=90, style='pieslice', outline=outline, width=width, fill=fill))
    items.append(canvas.create_arc(x0, y1 - 2 * r, x0 + 2 * r, y1, start=180, extent=90, style='pieslice', outline=outline, width=width, fill=fill))
    items.append(canvas.create_rectangle(x0 + r, y0, x1 - r, y1, outline=outline, width=0, fill=fill))
    items.append(canvas.create_rectangle(x0, y0 + r, x1, y1 - r, outline=outline, width=0, fill=fill))
    return items


def draw_shadow(canvas: tk.Canvas, x0: int, y0: int, x1: int, y1: int, r: int, color: str):
    # Simple single-layer shadow; callers can layer multiple for a nicer effect
    return draw_rounded_rect(canvas, x0, y0, x1, y1, r, fill=color, outline='')


