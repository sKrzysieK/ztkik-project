"""Pobranie SMS Spam Collection z UCI do `data/raw/`.

Skrypt jest idempotentny: jeśli plik już jest i ma poprawną sumę SHA-256, nic nie pobiera.
Z archiwum wypakowujemy tylko `SMSSpamCollection` (5574 wiersze `etykieta<TAB>treść`).

Uruchomienie: `python data/download.py`
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

# Uruchamiany jako skrypt (`python data/download.py`), więc katalog główny repo
# trzeba dodać ręcznie, żeby wziąć ścieżki z `src.config`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_RAW  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
FILENAME = "SMSSpamCollection"
# Suma wypakowanego pliku, nie archiwum: chroni przed uszkodzonym albo zmienionym zbiorem.
SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"


def _sha256(path: Path) -> str:
    """Zwraca sumę SHA-256 pliku jako tekst szesnastkowy."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(dest_dir: Path = DATA_RAW, url: str = URL, sha256: str = SHA256) -> Path:
    """Zwraca ścieżkę do `SMSSpamCollection`; pobiera go tylko, jeśli go brakuje lub jest uszkodzony."""
    path = dest_dir / FILENAME
    if path.exists() and _sha256(path) == sha256:
        return path

    with urllib.request.urlopen(url) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    content = archive.read(FILENAME)
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"suma SHA-256 pliku {FILENAME} z {url} nie zgadza się z oczekiwaną")

    # Zapis przez plik tymczasowy, żeby przerwany zapis nie zostawił połowy pliku.
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)
    return path


def main() -> None:
    path = DATA_RAW / FILENAME
    existed = path.exists() and _sha256(path) == SHA256
    download()
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    print(f"{'już jest' if existed else 'pobrano'}: {path} ({n_lines} wierszy)")


if __name__ == "__main__":
    main()
