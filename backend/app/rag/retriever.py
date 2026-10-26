import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / 'data' / 'knowledge_base.json'

class KnowledgeRetriever:
    def __init__(self):
        with DATA_PATH.open('r', encoding='utf-8') as f:
            self.data = json.load(f)

    def retrieve(self, injury: str, items: list[str]) -> dict:
        injury_data = self.data['injuries'].get(injury, {})
        materials = []
        for raw in items:
            name = raw.strip().lower()
            if not name:
                continue
            for key, value in self.data['materials'].items():
                if key in name or name in key:
                    materials.append({'item': key, **value})
        return {
            'injury': injury,
            'injury_guidance': injury_data,
            'matched_materials': materials,
            'sources': self.data.get('sources', []),
        }
