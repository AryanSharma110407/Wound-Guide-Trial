import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'


class KnowledgeRetriever:
    """Loads data/knowledge_base.json plus every data/extra/*.json file and merges them.
    To grow the knowledge base, drop another JSON file (same format) into data/extra/."""

    def __init__(self):
        self.data = {'injuries': {}, 'materials': {}, 'sources': []}
        files = [DATA_DIR / 'knowledge_base.json'] + sorted((DATA_DIR / 'extra').glob('*.json'))
        for path in files:
            with path.open('r', encoding='utf-8') as f:
                part = json.load(f)
            self.data['injuries'].update(part.get('injuries', {}))
            self.data['materials'].update(part.get('materials', {}))
            self.data['sources'] += [s for s in part.get('sources', []) if s not in self.data['sources']]

    @staticmethod
    def _has_phrase(text: str, phrase: str) -> bool:
        return re.search(rf'\b{re.escape(phrase)}\b', text) is not None

    def retrieve(self, injury: str, items: list[str]) -> dict:
        materials, unmatched = [], []
        for raw in items:
            name = raw.strip().lower()
            if not name:
                continue
            found = False
            for key, value in self.data['materials'].items():
                names = [key] + value.get('aliases', [])
                if any(self._has_phrase(name, n) for n in names):
                    if not any(m['item'] == key for m in materials):
                        materials.append({'item': key, **value})
                    found = True
            if not found:
                unmatched.append(raw.strip())
        return {
            'injury': injury,
            'injury_guidance': self.data['injuries'].get(injury, {}),
            'matched_materials': materials,
            'unmatched_items': unmatched,
            'sources': self.data['sources'],
        }
