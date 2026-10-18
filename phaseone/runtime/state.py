"""Durable conversation, tool-action, and execution records using SQLite WAL."""

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .errors import RuntimeFault
from .package import canonical


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False, timeout=15)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, package TEXT NOT NULL, model TEXT NOT NULL,
                created REAL NOT NULL, messages TEXT NOT NULL, active_turn TEXT
            );
            CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY, session TEXT NOT NULL REFERENCES sessions(id),
                fingerprint TEXT NOT NULL, status TEXT NOT NULL,
                created REAL NOT NULL, result TEXT
            );
            CREATE TABLE IF NOT EXISTS actions (
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, status TEXT NOT NULL,
                created REAL NOT NULL, result TEXT
            );
            CREATE TABLE IF NOT EXISTS receipts (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL, previous TEXT NOT NULL, hash TEXT NOT NULL
            );
        """)

    @contextmanager
    def transaction(self):
        with self.lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                yield self.connection
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def close(self):
        with self.lock:
            self.connection.close()

    def receipt(self, kind: str, payload: dict) -> dict:
        with self.transaction() as db:
            row = db.execute("SELECT hash FROM receipts ORDER BY seq DESC LIMIT 1").fetchone()
            previous = row["hash"] if row else "0" * 64
            created = time.time()
            record = {"created": created, "kind": kind, "payload": payload, "previous": previous}
            digest = hashlib.sha256(canonical(record)).hexdigest()
            cursor = db.execute(
                "INSERT INTO receipts(created,kind,payload,previous,hash) VALUES(?,?,?,?,?)",
                (created, kind, canonical(payload).decode(), previous, digest),
            )
            return {"seq": cursor.lastrowid, **record, "hash": digest}

    def receipts(self, after: int = 0, limit: int = 100) -> list[dict]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM receipts WHERE seq>? ORDER BY seq LIMIT ?", (after, min(limit, 1000)),
            ).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def verify_receipts(self) -> dict:
        previous, count = "0" * 64, 0
        with self.lock:
            for row in self.connection.execute("SELECT * FROM receipts ORDER BY seq"):
                body = {"created": row["created"], "kind": row["kind"],
                        "payload": json.loads(row["payload"]), "previous": row["previous"]}
                if row["previous"] != previous or hashlib.sha256(canonical(body)).hexdigest() != row["hash"]:
                    return {"valid": False, "first_invalid_seq": row["seq"], "records": count}
                previous = row["hash"]
                count += 1
        return {"valid": True, "records": count, "head": previous,
                "assurance": "local_hash_chain; not hardware attestation"}

    def create_session(self, package: str, model: str, messages: list[dict] | None = None) -> dict:
        session_id = str(uuid.uuid4())
        with self.transaction() as db:
            db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,NULL)",
                       (session_id, package, model, time.time(), canonical(messages or []).decode()))
        return self.session(session_id)

    def session(self, session_id: str) -> dict:
        with self.lock:
            row = self.connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise RuntimeFault("session_not_found", "Unknown session")
        return {**dict(row), "messages": json.loads(row["messages"])}

    def begin_turn(self, session_id: str, turn_id: str, request: dict) -> dict:
        fingerprint = hashlib.sha256(canonical(request)).hexdigest()
        with self.transaction() as db:
            session = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if session is None:
                raise RuntimeFault("session_not_found", "Unknown session")
            prior = db.execute("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchone()
            if prior:
                if prior["fingerprint"] != fingerprint or prior["session"] != session_id:
                    raise RuntimeFault("idempotency_conflict", "Turn ID was used for a different request")
                if prior["status"] == "complete":
                    return {"cached": True, "result": json.loads(prior["result"])}
                raise RuntimeFault("turn_incomplete", "Turn is pending or failed; inspect it before retrying",
                                   details={"turn_id": turn_id, "status": prior["status"]})
            if session["active_turn"]:
                raise RuntimeFault("session_busy", "A session can have only one active turn",
                                   details={"turn_id": session["active_turn"]})
            db.execute("INSERT INTO turns VALUES(?,?,?,?,?,NULL)",
                       (turn_id, session_id, fingerprint, "pending", time.time()))
            db.execute("UPDATE sessions SET active_turn=? WHERE id=?", (turn_id, session_id))
            return {"cached": False, "messages": json.loads(session["messages"])}

    def finish_turn(self, session_id: str, turn_id: str, messages: list[dict], result: dict,
                    *, status: str = "complete"):
        if status not in ("complete", "failed", "cancelled"):
            raise ValueError("Invalid terminal turn status")
        with self.transaction() as db:
            session = db.execute("SELECT active_turn FROM sessions WHERE id=?", (session_id,)).fetchone()
            if session is None or session["active_turn"] != turn_id:
                raise RuntimeFault("turn_conflict", "This turn no longer owns the session")
            db.execute("UPDATE turns SET status=?,result=? WHERE id=? AND session=?",
                       (status, canonical(result).decode(), turn_id, session_id))
            db.execute("UPDATE sessions SET messages=?,active_turn=NULL WHERE id=?",
                       (canonical(messages).decode(), session_id))

    def recover_session(self, session_id: str) -> dict:
        """Explicit recovery releases a crashed turn without re-executing any tool."""
        with self.transaction() as db:
            row = db.execute("SELECT active_turn FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise RuntimeFault("session_not_found", "Unknown session")
            if row["active_turn"]:
                db.execute("UPDATE turns SET status='failed',result=? WHERE id=?",
                           (canonical({"error": "recovered_after_interruption"}).decode(), row["active_turn"]))
                db.execute("UPDATE sessions SET active_turn=NULL WHERE id=?", (session_id,))
        return self.session(session_id)

    def begin_action(self, action_id: str, request: dict) -> dict:
        fingerprint = hashlib.sha256(canonical(request)).hexdigest()
        with self.transaction() as db:
            row = db.execute("SELECT * FROM actions WHERE id=?", (action_id,)).fetchone()
            if row:
                if row["fingerprint"] != fingerprint:
                    raise RuntimeFault("idempotency_conflict", "Action ID was used for different arguments")
                if row["status"] in ("complete", "denied", "failed"):
                    return {"cached": True, "status": row["status"], "result": json.loads(row["result"])}
                raise RuntimeFault("action_outcome_unknown",
                                   "Action may have executed; automatic replay is blocked",
                                   details={"action_id": action_id})
            db.execute("INSERT INTO actions VALUES(?,?,?,?,NULL)",
                       (action_id, fingerprint, "pending", time.time()))
        return {"cached": False}

    def finish_action(self, action_id: str, result: dict, status: str = "complete"):
        if status not in ("complete", "denied", "failed", "uncertain"):
            raise ValueError("Invalid terminal action status")
        with self.transaction() as db:
            changed = db.execute("UPDATE actions SET status=?,result=? WHERE id=? AND status='pending'",
                                 (status, canonical(result).decode(), action_id)).rowcount
            if changed != 1:
                raise RuntimeFault("action_conflict", "Action is not pending")

    def action(self, action_id: str) -> dict:
        with self.lock:
            row = self.connection.execute("SELECT * FROM actions WHERE id=?", (action_id,)).fetchone()
        if row is None:
            raise RuntimeFault("action_not_found", "Unknown action")
        return {**dict(row), "result": json.loads(row["result"]) if row["result"] else None}
