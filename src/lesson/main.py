import json
from difflib import unified_diff
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv, main
from lesson.generator import generate_lesson
from lesson.evaluator import evaluate_lesson, get_evidence
from lesson.references import REFERENCE_FACTS
from lesson.rubric import RUBRIC, all_checks_pass
from lesson.memory import load_memory, remember_run
import argparse

load_dotenv()
def run_pipeline(
    topic: str,
    max_retries: int = 2,
    *,
    initial_lesson: str | None = None,
) -> dict:
    if type(max_retries) is not int or not 0 <= max_retries <= 2:
        raise ValueError("max_retries must be 0, 1, or 2.")

    if initial_lesson is not None:
        if not isinstance(initial_lesson, str) or not initial_lesson.strip():
            raise ValueError("initial_lesson must be a nonempty string.")
        initial_lesson = initial_lesson.strip()

    memory_state = load_memory(topic, REFERENCE_FACTS, RUBRIC)
    print(f"Memory version used: {memory_state['version']}")

    result = {
        "run_id": uuid4().hex[:12],
        "memory_before": memory_state,
        "topic": topic,
        "status": "NEEDS_REVIEW",
        "lesson": None,
        "max_retries": max_retries,
        "reference_facts": REFERENCE_FACTS,
        "rubric": RUBRIC,
        "attempts": [],
    }
    previous_lesson = ""
    failed_checks = []

    for number in range(1, max_retries + 2):
        print(f"\nAttempt {number}/{max_retries + 1}")
        record = {
            "attempt": number,
            "lesson": None,
            "passed": None,
            "evaluation": {},
            "failed_checks": [],
            "changes_from_previous": "",
        }
        result["attempts"].append(record)

        try:
            if number == 1 and initial_lesson is not None:
                print("Using supplied draft...")
                lesson = initial_lesson
                record["draft_source"] = "supplied"
            else:
                print("Generating..." if not previous_lesson else "Revising...")
                lesson = generate_lesson(
                    topic,
                    previous_lesson,
                    failed_checks,
                    learned_guidance=memory_state["guidance"],
                )
                record["draft_source"] = (
                    "revised" if previous_lesson else "generated"
                )

            record["lesson"] = lesson
            
            
            if previous_lesson:
                record["changes_from_previous"] = "".join(unified_diff(
                    (previous_lesson + "\n").splitlines(keepends=True),
                    (lesson + "\n").splitlines(keepends=True),
                    fromfile=f"attempt_{number - 1}",
                    tofile=f"attempt_{number}",
                ))

            print("Evaluating...")
            evaluation = evaluate_lesson(topic, lesson)
            flags = {}
            failed_checks = []

            for checkpoint in RUBRIC:
                check = getattr(evaluation, checkpoint)
                details = {
                    **check.model_dump(),
                    "evidence": get_evidence(check, lesson),
                }
                record["evaluation"][checkpoint] = details
                flags[checkpoint] = check.passed

                print(f"{checkpoint}: {'PASS' if check.passed else 'FAIL'}")
                if not check.passed:
                    failed_checks.append({"checkpoint": checkpoint, **details})
                    print(f"  Reason: {check.reason}")
                    print(f"  Fix: {check.fix}")

            record["passed"] = all_checks_pass(flags)
            record["failed_checks"] = failed_checks

            if record["passed"]:
                result["status"] = "ACCEPTED"
                result["lesson"] = lesson
                break

            previous_lesson = lesson
            print("Lesson rejected.")

        except Exception as error:
            result["status"] = "ERROR"
            result["error"] = f"{type(error).__name__}: {error}"
            break

    return result

def save_run(result: dict) -> Path:
    folder = Path("outputs") / result["run_id"]
    folder.mkdir(parents=True, exist_ok=False)

    (folder / "run_log.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rejections = []
    for index, attempt in enumerate(result["attempts"]):
        if attempt["lesson"] is not None:
            (folder / f"attempt_{attempt['attempt']}.txt").write_text(
                attempt["lesson"], encoding="utf-8"
            )

        if attempt["attempt"] > 1 and attempt["lesson"] is not None:
            (folder / f"revision_{attempt['attempt']}.patch").write_text(
                attempt["changes_from_previous"], encoding="utf-8"
            )

        if attempt["passed"] is False:
            next_attempt = (
                result["attempts"][index + 1]
                if index + 1 < len(result["attempts"]) else None
            )
            rejections.append({
                "attempt": attempt["attempt"],
                "failed_checks": attempt["failed_checks"],
                "changes_on_retry": (
                    next_attempt["changes_from_previous"]
                    if next_attempt and next_attempt["lesson"] is not None
                    else None
                ),
                "retry_passed": next_attempt["passed"] if next_attempt else None,
            })

    (folder / "rejection_log.json").write_text(
        json.dumps(rejections, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if result["status"] == "ACCEPTED":
        (folder / "final_lesson.txt").write_text(
            result["lesson"], encoding="utf-8"
        )

    return folder.resolve()

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate a beginner RAG lesson."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Lesson revisions after the first evaluation (default: 2).",
    )
    parser.add_argument(
        "--initial-lesson",
        type=Path,
        help="Evaluate this saved draft first, then revise it if needed.",
    )
    args = parser.parse_args(argv)

    initial_lesson = None
    if args.initial_lesson is not None:
        try:
            initial_lesson = args.initial_lesson.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            parser.error(f"Cannot read --initial-lesson: {error}")

        if not initial_lesson.strip():
            parser.error("--initial-lesson must contain a nonempty lesson.")

    result = run_pipeline(
        "Introduction to RAG",
        max_retries=args.max_retries,
        initial_lesson=initial_lesson,
    )

    if args.initial_lesson is not None:
        result["initial_lesson_source"] = str(args.initial_lesson.resolve())

    result["memory_after"] = remember_run(result)
    folder = save_run(result)

    rejected = sum(
        attempt["passed"] is False
        for attempt in result["attempts"]
    )

    print(f"Memory version saved: {result['memory_after']['version']}")
    print(f"\nStatus: {result['status']}")
    print(f"Attempts recorded: {len(result['attempts'])}")
    print(f"Rejected attempts: {rejected}")
    print(f"Saved output: {folder}")
    print(f"Rejection log: {folder / 'rejection_log.json'}")

    if result["status"] == "ACCEPTED":
        print("\nFINAL LESSON\n")
        print(result["lesson"])
    elif result["status"] == "ERROR":
        print(result["error"])
    else:
        print("Retry limit reached. Human review is required.")

    return {"ACCEPTED": 0, "NEEDS_REVIEW": 2, "ERROR": 1}[result["status"]]

if __name__ == "__main__":
    raise SystemExit(main())