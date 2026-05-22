import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.web.app import app


def test_upload_route_is_preserved_for_api_compatibility():
    routes = {str(rule) for rule in app.url_map.iter_rules()}

    assert "/api/upload" in routes
    assert "/api/reindex" in routes
