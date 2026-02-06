"""
Utility for saving and loading CLI commands.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


def _store_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    store_dir = root / ".transcribator"
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir / "commands.json"


def _settings_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    store_dir = root / ".transcribator"
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir / "settings.json"


def load_commands() -> Dict[str, Dict]:
    path = _store_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        return {}
    return {}


def list_commands() -> List[Dict]:
    commands = load_commands()
    items = []
    for name, payload in commands.items():
        items.append({
            "name": name,
            "command": payload.get("command", ""),
            "description": payload.get("description"),
            "updated_at": payload.get("updated_at"),
        })
    return sorted(items, key=lambda x: x.get("name", ""))


def save_command(name: str, command: str, description: Optional[str] = None) -> None:
    name = name.strip()
    if not name:
        raise ValueError("Название команды не может быть пустым")

    data = load_commands()
    now = datetime.utcnow().isoformat() + "Z"
    entry = data.get(name, {})
    if "created_at" not in entry:
        entry["created_at"] = now
    entry["updated_at"] = now
    entry["command"] = command
    if description:
        entry["description"] = description
    data[name] = entry

    path = _store_path()
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_settings() -> Dict[str, Any]:
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        return {}
    return {}


def save_settings(settings: Dict[str, Any]) -> None:
    path = _settings_path()
    with path.open("w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
