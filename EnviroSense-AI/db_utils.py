from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

_ENGINE = None

def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(
            "postgresql://postgres:rachna@69.62.83.135:5432/envirosense"
        )
    return _ENGINE