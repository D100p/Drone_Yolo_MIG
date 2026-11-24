"""Plafond de RAM pour le processus.

- Linux (Pi) : vraie limite noyau via RLIMIT_AS -> une allocation au-delà échoue (MemoryError).
- macOS      : le noyau ignore RLIMIT_AS / RLIMIT_DATA, donc on surveille la RAM réelle
               (RSS) à chaque frame et on s'arrête proprement si elle dépasse le plafond.
Le chien de garde tourne sur les deux systèmes, la limite noyau en plus sur Linux.
"""

import os
import psutil
import sys


def _rss_bytes() -> int:
    """RAM physique actuellement utilisée par le processus."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        # Repli sans psutil : pic de RAM (octets sur macOS, kilo-octets sur Linux)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024


class MemoryGuard:
    def __init__(self, limit_gb: float):
        self.limit_bytes = int(limit_gb * 1024**3)
        self.peak_bytes = 0
        self.kernel_limit = False
        if sys.platform.startswith("linux"):
            try:
                # RLIMIT_AS limite l'espace virtuel (>= RAM réelle), donc c'est un plafond
                # strict : ce programme réserve bien moins de 8 Go de virtuel.
                resource.setrlimit(resource.RLIMIT_AS, (self.limit_bytes, self.limit_bytes))
                self.kernel_limit = True
            except (ValueError, OSError):
                pass

    def check(self) -> float:
        """Renvoie la RAM utilisée en Mo ; lève MemoryError si le plafond est dépassé."""
        rss = _rss_bytes()
        self.peak_bytes = max(self.peak_bytes, rss)
        if rss > self.limit_bytes:
            raise MemoryError(
                f"Plafond RAM dépassé : {rss / 1024**3:.2f} Go > {self.limit_bytes / 1024**3:.2f} Go"
            )
        return rss / 1024**2

    def describe(self) -> str:
        mode = "limite noyau + surveillance" if self.kernel_limit else "surveillance RSS"
        return f"plafond RAM {self.limit_bytes / 1024**3:.1f} Go ({mode})"
