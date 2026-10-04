import argparse
from pathlib import Path

from dotenv import load_dotenv
from lesson.evaluator import evaluate_lesson, get_evidence

QUALIFICATION = (
    "The following explanation shows one common RAG implementation using "
    "chunks, embeddings, and a vector store. Other RAG systems can retrieve "
    "information by matching words in documents or looking up database "
    "records, without using this vector-store setup."
)


def build_cases(lesson: str) -> list[tuple[str, str, bool]]:
    heading = "## How It Works"
    if heading not in lesson:
        raise ValueError(f"Expected heading not found: {heading}")

    corrected = lesson.replace(
        heading, f"{heading}\n\n{QUALIFICATION}", 1
    )
    corrected = corrected.replace(
        "RAG happens in two main stages: **document preparation** "
        "and **question answering**.",
        "This document-based implementation has two stages: "
        "**document preparation** and **question answering**.",
    )
    corrected = corrected.replace(
        "RAG works only on configured document sources and does not "
        "automatically scan the entire internet.",
        "RAG uses configured information sources and does not "
        "automatically scan the entire internet.",
    )

    contradiction = (
        corrected
        + "\n\nEvery RAG system must use embeddings and a vector store.\n"
    )
    return [
        ("missing_qualification", lesson, False),
        ("qualified_implementation", corrected, True),
        ("contradictory_claim", contradiction, False),
    ]


def run_checks(lesson: str) -> bool:
    successful = 0
    cases = build_cases(lesson)

    for name, candidate, expected in cases:
        print(f"\nCase: {name}")
        expected_label = "PASS" if expected else "FAIL"
        try:
            evaluation = evaluate_lesson("Introduction to RAG", candidate)
        except Exception as error:
            print(f"CHECK ERROR: {type(error).__name__}: {error}")
            continue

        check = evaluation.accuracy
        actual_label = "PASS" if check.passed else "FAIL"
        print(f"Expected accuracy: {expected_label}")
        print(f"Received accuracy: {actual_label}")
        print(f"Reason: {check.reason}")

        if check.passed is expected:
            successful += 1
            print("CHECK PASSED")
        else:
            print("CHECK FAILED")
            print(get_evidence(check, candidate))

    print(f"\nCalibration: {successful}/{len(cases)} checks passed")
    return successful == len(cases)


def main():
    parser = argparse.ArgumentParser(
        description="Test the accuracy judge on a known unqualified lesson."
    )
    parser.add_argument("--lesson", type=Path, required=True)
    args = parser.parse_args()

    load_dotenv()
    try:
        lesson = args.lesson.read_text(encoding="utf-8")
        passed = run_checks(lesson)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())