from __future__ import annotations

from typing import Optional, Tuple

from ..db import check_db


def check_db_health() -> Tuple[bool, Optional[str]]:
    return check_db()
