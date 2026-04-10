from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HoloFooter:
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default=None) -> Any:
        return self.data.get(key, default)

    def to_json(self) -> str:
        return json.dumps(self.data)

    @staticmethod
    def from_json(s: str) -> "HoloFooter":
        return HoloFooter(data=json.loads(s))

    @staticmethod
    def empty() -> "HoloFooter":
        return HoloFooter(data={})
