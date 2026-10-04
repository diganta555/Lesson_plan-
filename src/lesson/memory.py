import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "lesson_memory.sqlite3"
FAILURE_THRESHOLD = 2

GUIDANCE = {
    "accuracy": (
        "Check every technical claim against the reference facts. "
        "Describe embeddings and vector stores as one common RAG approach; "
        "do not imply that every RAG system requires them."
    ),
    "beginner_language": (
        "Explain each new idea in a short, plain-language sentence before "
        "adding technical detail. Assume no previous AI knowledge."
    ),
    "worked_example": (
        "Include an explicit question, retrieved passage, context supplied "
        "to the model, and answer. Show how the answer follows from the passage."
    ),
    "jargon": (
        "Define each acronym and technical term at its first appearance, "
        "including terms used inside the worked example."
    ),
    "coverage": (
        "Check for what RAG is, why it matters, document preparation, "
        "question-time retrieval and generation, and at least two limitations."
    ),
    "teaching_flow": (
        "Order the lesson as definition, motivation, mechanism, worked "
        "example, limitations, and recap. Introduce ideas before using them."
    ),
}


def _context(topic: str, reference_facts: str, rubric: dict) -> str:
    payload = {
        "topic": " ".join(topic.casefold().split()),
        "reference_facts": reference_facts,
        "rubric": rubric,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _connect(db_path: Path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            saved_at TEXT NOT NULL,
            context TEXT NOT NULL,
            topic TEXT NOT NULL,
            status TEXT NOT NULL,
            report TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS failures (
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            checkpoint TEXT NOT NULL,
            PRIMARY KEY (run_id, checkpoint)
        );
        CREATE TABLE IF NOT EXISTS prompt_versions (
            context TEXT NOT NULL,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            guidance TEXT NOT NULL,
            PRIMARY KEY (context, version)
        );
        CREATE TABLE IF NOT EXISTS human_feedback (
            feedback_id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            checkpoint TEXT NOT NULL,
            note TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (run_id, checkpoint, note)
        );
    """)
    return db


def _snapshot(db, context: str) -> dict:
    row = db.execute(
        "SELECT version, guidance FROM prompt_versions "
        "WHERE context = ? ORDER BY version DESC LIMIT 1",
        (context,),
    ).fetchone()
    counts = db.execute(
        "SELECT f.checkpoint, COUNT(*) AS count "
        "FROM failures AS f JOIN runs AS r ON r.run_id = f.run_id "
        "WHERE r.context = ? GROUP BY f.checkpoint",
        (context,),
    ).fetchall()
    human_counts = db.execute(
        "SELECT h.checkpoint, COUNT(DISTINCT h.run_id) AS count "
        "FROM human_feedback AS h JOIN runs AS r ON r.run_id = h.run_id "
        "WHERE r.context = ? GROUP BY h.checkpoint",
        (context,),
    ).fetchall()
    return {
        "context": context,
        "version": row["version"] if row else 0,
        "guidance": json.loads(row["guidance"]) if row else {},
        "failure_counts": {item["checkpoint"]: item["count"] for item in counts},
        "human_failure_counts": {
            item["checkpoint"]: item["count"] for item in human_counts
        },
    }


def _update_guidance(db, context: str) -> dict:
    snapshot = _snapshot(db, context)
    guidance = {
        checkpoint: instruction
        for checkpoint, instruction in GUIDANCE.items()
        if (
            snapshot["failure_counts"].get(checkpoint, 0) >= FAILURE_THRESHOLD
            or snapshot["human_failure_counts"].get(checkpoint, 0) > 0
        )
    }
    if guidance != snapshot["guidance"]:
        db.execute(
            "INSERT INTO prompt_versions VALUES (?, ?, ?, ?)",
            (
                context,
                snapshot["version"] + 1,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(guidance, ensure_ascii=False),
            ),
        )
    return _snapshot(db, context)


def load_memory(
    topic: str,
    reference_facts: str,
    rubric: dict,
    db_path: Path = DB_PATH,
) -> dict:
    context = _context(topic, reference_facts, rubric)
    with closing(_connect(db_path)) as db:
        return _snapshot(db, context)


def remember_run(result: dict, db_path: Path = DB_PATH) -> dict:
    context = _context(
        result["topic"], result["reference_facts"], result["rubric"]
    )
    failed = {
        check["checkpoint"]
        for attempt in result["attempts"]
        if attempt["passed"] is False
        for check in attempt["failed_checks"]
    }
    if not failed <= (set(result["rubric"]) & set(GUIDANCE)):
        raise ValueError("Run contains an unknown failed checkpoint.")

    timestamp = datetime.now(timezone.utc).isoformat()
    with closing(_connect(db_path)) as db:
        with db:
            inserted = db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO NOTHING",
                (
                    result["run_id"], timestamp, context, result["topic"],
                    result["status"], json.dumps(result, ensure_ascii=False),
                ),
            ).rowcount
            if not inserted:
                saved = db.execute(
                    "SELECT context FROM runs WHERE run_id = ?",
                    (result["run_id"],),
                ).fetchone()
                if saved["context"] != context:
                    raise ValueError("This run_id already belongs to another context.")
                return _snapshot(db, context)

            db.executemany(
                "INSERT INTO failures VALUES (?, ?)",
                [(result["run_id"], checkpoint) for checkpoint in sorted(failed)],
            )
            after = _update_guidance(db, context)
            report = {**result, "memory_after": after}
            db.execute(
                "UPDATE runs SET report = ? WHERE run_id = ?",
                (json.dumps(report, ensure_ascii=False), result["run_id"]),
            )
            return after


def record_feedback(
    run_id: str,
    checkpoint: str,
    note: str,
    db_path: Path = DB_PATH,
) -> dict:
    note = note.strip()
    if not note:
        raise ValueError("Explain the failure in the feedback note.")

    with closing(_connect(db_path)) as db:
        with db:
            run = db.execute(
                "SELECT context, status, report FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise ValueError(f"Run not found in memory: {run_id}")

            report = json.loads(run["report"])
            if checkpoint not in GUIDANCE or checkpoint not in report["rubric"]:
                raise ValueError("Checkpoint is not in this run's rubric.")

            db.execute(
                "INSERT INTO human_feedback "
                "(run_id, checkpoint, note, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(run_id, checkpoint, note) DO NOTHING",
                (
                    run_id, checkpoint, note,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            after = _update_guidance(db, run["context"])
            return {
                "run_id": run_id,
                "checkpoint": checkpoint,
                "note": note,
                "evaluator_status": run["status"],
                "human_review_status": "NEEDS_REVIEW",
                "memory_after": after,
            }