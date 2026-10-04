import argparse
import json
from lesson.memory import GUIDANCE, record_feedback


def main():
    parser = argparse.ArgumentParser(
        description="Record a human-identified checkpoint failure."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", required=True, choices=sorted(GUIDANCE))
    parser.add_argument("--note", required=True)
    args = parser.parse_args()

    try:
        result = record_feedback(args.run_id, args.checkpoint, args.note)
    except ValueError as error:
        parser.error(str(error))

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())