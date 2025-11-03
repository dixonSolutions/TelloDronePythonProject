from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    # Fedora-inspired blue with PrimeNG accent balance
    primary: str = '#3c6eb4'  # Fedora blue
    primary_hover: str = '#2f5d99'
    primary_active: str = '#264c7d'
    accent: str = '#3B82F6'   # PrimeNG blue
    bg: str = '#f5f7fa'
    surface: str = '#ffffff'
    text: str = '#0f1b2a'
    text_muted: str = '#3b4a60'
    border: str = '#d7dee9'
    # Tkinter Canvas does not support alpha in hex; use an opaque light shadow color
    shadow: str = '#c7d2e6'

    radius_lg: int = 14
    radius_md: int = 10
    radius_sm: int = 8

    pad_lg: int = 16
    pad_md: int = 12
    pad_sm: int = 8

    font_family: str = 'Cantarell'  # Fedora default; falls back to Sans
    font_size: int = 12
    font_size_lg: int = 14

    @staticmethod
    def as_font(family: str | None = None, size: int | None = None, weight: str | None = None):
        fam = family or Theme.font_family
        sz = size or Theme.font_size
        if weight:
            return (fam, sz, weight)
        return (fam, sz)


