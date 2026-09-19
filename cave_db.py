"""
cave_db.py: SQLite persistence for rescue missions.

Each verified mission (safety PASS or FAIL) is one row in missions.db, next to this file.
A fresh connection is opened per call, so one MissionDatabase can be shared across
Streamlit's script threads (sqlite3 connections can't be shared between threads).
"""
import json
import os
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'missions.db')

COLUMNS = ('id', 'timestamp', 'command_text', 'action', 'target_chamber', 'payload', 'priority', 'path_json',
           'path_length', 'safety_verdict', 'safety_reason', 'battery_before', 'battery_after', 'bot_x', 'bot_y',
           'algorithm_used', 'bot_id')


class MissionDatabase:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        missing = not os.path.exists(self.db_path)  # e.g. deleted to reset while the app is running
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        if missing:
            with conn:
                self._create_schema(conn)
        return conn

    @contextmanager
    def _session(self):
        """Connection that commits (or rolls back) and is always closed; `with conn:` alone never closes it"""
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._session() as conn:
            self._create_schema(conn)

    @staticmethod
    def _create_schema(conn: sqlite3.Connection):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS missions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT    NOT NULL,
                command_text   TEXT    NOT NULL,
                action         TEXT,
                target_chamber INTEGER,
                payload        TEXT,
                priority       TEXT,
                path_json      TEXT    NOT NULL,
                path_length    INTEGER NOT NULL,
                safety_verdict TEXT    NOT NULL CHECK (safety_verdict IN ('PASS', 'FAIL')),
                safety_reason  TEXT,
                battery_before REAL,
                battery_after  REAL,
                bot_x          INTEGER,
                bot_y          INTEGER,
                algorithm_used TEXT,
                bot_id         TEXT    NOT NULL DEFAULT 'UNIT-01'
            )""")
        # Migration for databases created before the second bot: every earlier mission was UNIT-01's
        existing = {row['name'] for row in conn.execute("PRAGMA table_info(missions)")}
        if 'bot_id' not in existing:
            conn.execute("ALTER TABLE missions ADD COLUMN bot_id TEXT NOT NULL DEFAULT 'UNIT-01'")

    def log_mission(self, command: str, parsed_intent: Dict[str, Any], path: Sequence[Tuple[int, int]],
                    safety_verdict, battery_before: float, battery_after: float,
                    bot_position: Tuple[int, int], algorithm_used: str = "A*", bot_id: str = "UNIT-01") -> int:
        """Inserts one mission and returns its id.

        safety_verdict is 'PASS'/'FAIL', a bool, or SafetyVerifier's (is_safe, violations) tuple.
        path_length is the number of moves (cells in the path minus the start cell).
        """
        verdict, reason = self._normalise_verdict(safety_verdict)
        cells = [[int(x), int(y)] for x, y in path]
        row = (
            datetime.now().isoformat(timespec='seconds'), command,
            parsed_intent.get('action'), parsed_intent.get('target_chamber'), parsed_intent.get('payload'),
            parsed_intent.get('priority'), json.dumps(cells), max(len(cells) - 1, 0), verdict, reason,
            battery_before, battery_after, int(bot_position[0]), int(bot_position[1]), algorithm_used, bot_id,
        )
        with self._session() as conn:
            cursor = conn.execute(
                f"INSERT INTO missions ({', '.join(COLUMNS[1:])}) VALUES ({', '.join('?' * (len(COLUMNS) - 1))})",
                row)
            return int(cursor.lastrowid or 0)

    def query_missions(self, chamber: Optional[int] = None, action: Optional[str] = None,
                       safety_outcome: Optional[str] = None, limit: int = 50,
                       bot_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Newest-first missions matching every given filter; each dict also carries the decoded 'path'"""
        clauses, params = [], []
        for column, value in (('target_chamber', chamber), ('action', action), ('safety_verdict', safety_outcome),
                              ('bot_id', bot_id)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ''
        with self._session() as conn:
            rows = conn.execute(f"SELECT * FROM missions {where} ORDER BY id DESC LIMIT ?",
                                (*params, int(limit))).fetchall()
        missions = [dict(row) for row in rows]
        for mission in missions:
            mission['path'] = [tuple(cell) for cell in json.loads(mission['path_json'])]
        return missions

    def get_stats(self) -> Dict[str, Any]:
        """total: missions logged; success_rate: PASS fraction (None if empty);
        avg_path_length: mean moves of executed (PASS) missions; most_visited_chamber: most common PASS target"""
        with self._session() as conn:
            total, passed, avg_path = conn.execute(
                "SELECT COUNT(*), SUM(safety_verdict = 'PASS'), "
                "AVG(CASE WHEN safety_verdict = 'PASS' THEN path_length END) FROM missions").fetchone()
            visits = conn.execute(
                "SELECT target_chamber FROM missions WHERE safety_verdict = 'PASS' AND target_chamber IS NOT NULL"
            ).fetchall()
        counts = Counter(row[0] for row in visits)
        return {
            'total': total,
            'success_rate': (passed or 0) / total if total else None,
            'avg_path_length': avg_path,
            'most_visited_chamber': counts.most_common(1)[0][0] if counts else None,
        }

    @staticmethod
    def _normalise_verdict(safety_verdict) -> Tuple[str, str]:
        if isinstance(safety_verdict, tuple):  # (is_safe, violations) straight from SafetyVerifier
            is_safe, violations = safety_verdict
            return ('PASS' if is_safe else 'FAIL'), '; '.join(violations)
        if isinstance(safety_verdict, bool):
            return ('PASS' if safety_verdict else 'FAIL'), ''
        verdict = str(safety_verdict).upper()
        if verdict not in ('PASS', 'FAIL'):
            raise ValueError(f"safety_verdict must be PASS/FAIL, a bool, or (is_safe, violations); got {safety_verdict!r}")
        return verdict, ''
