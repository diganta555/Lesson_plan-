import json

from lesson.memory import load_memory
from lesson.references import REFERENCE_FACTS
from lesson.rubric import RUBRIC


def main() -> int:
    memory = load_memory(
        "Introduction to RAG",
        REFERENCE_FACTS,
        RUBRIC,
    )

    print(json.dumps(memory, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())