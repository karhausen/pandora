from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .genome import PandoraGenome


class GenomeLoader:
    """Loads and writes the static Pandora Genome configuration.

    This loader is intentionally small. Runtime systems should read from it,
    but not write to it without an explicit controlled maintenance action.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2]
        self.path = self.root / "config" / "system" / "pandora_genome.json"

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> PandoraGenome:
        if not self.path.exists():
            genome = PandoraGenome.default()
            self.write(genome)
            return genome
        data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        return PandoraGenome.from_dict(data)

    def write(self, genome: PandoraGenome) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(genome.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
