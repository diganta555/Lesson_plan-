import io
import json
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from lesson.evaluator import CheckResult, EvaluationResult
from lesson.main import run_pipeline, save_run
from lesson.rubric import RUBRIC

TOPIC = "Introduction to RAG"
FIRST_DRAFT = "Draft 1\nRAG always requires a vector store."
PASSING_DRAFT = (
    "Draft 2\nOne common RAG approach uses a vector store.\n"
    "Other retrieval methods can use database lookups."
)
REASON = "The draft incorrectly requires a vector store for every RAG system."
FIX = "Explain that vector-store retrieval is one common implementation."
MEMORY = {
    "context": "test-context",
    "version": 1,
    "guidance": {"accuracy": FIX},
    "failure_counts": {"accuracy": 2},
    "human_failure_counts": {},
}


def evaluation(failed_checkpoint=None):
    return EvaluationResult(**{
        checkpoint: CheckResult(
            passed=checkpoint != failed_checkpoint,
            reason=REASON if checkpoint == failed_checkpoint else "Requirement met.",
            evidence_lines=[2],
            fix=FIX if checkpoint == failed_checkpoint else "",
        )
        for checkpoint in RUBRIC
    })


def execute(drafts, verdicts, max_retries=2):
    with (
        patch("lesson.main.load_memory", return_value=MEMORY) as memory,
        patch("lesson.main.generate_lesson", side_effect=drafts) as generator,
        patch("lesson.main.evaluate_lesson", side_effect=verdicts) as evaluator,
        redirect_stdout(io.StringIO()),
    ):
        result = run_pipeline(TOPIC, max_retries=max_retries)
    return result, generator, evaluator, memory


@contextmanager
def saved_run(result):
    with TemporaryDirectory() as temporary:
        root = Path(temporary) / "outputs"
        with patch("lesson.main.Path", return_value=root):
            folder = save_run(result)
        yield folder


class PipelineTests(unittest.TestCase):
    def test_first_pass_stops_immediately(self):
        result, generator, evaluator, memory = execute(
            [PASSING_DRAFT], [evaluation()]
        )
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertEqual(result["lesson"], PASSING_DRAFT)
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(evaluator.call_count, 1)
        memory.assert_called_once()
        generator.assert_called_once_with(
            TOPIC, "", [], learned_guidance=MEMORY["guidance"]
        )

    def test_failure_is_sent_to_revision(self):
        result, generator, evaluator, memory = execute(
            [FIRST_DRAFT, PASSING_DRAFT],
            [evaluation("accuracy"), evaluation()],
        )
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertEqual(result["lesson"], PASSING_DRAFT)
        self.assertEqual(generator.call_count, 2)
        self.assertEqual(evaluator.call_count, 2)
        memory.assert_called_once()
        self.assertIs(result["attempts"][0]["passed"], False)
        self.assertIs(result["attempts"][1]["passed"], True)

        retry = generator.call_args_list[1]
        self.assertEqual(retry.args[1], FIRST_DRAFT)
        self.assertEqual(
            retry.args[2], result["attempts"][0]["failed_checks"]
        )
        feedback = retry.args[2][0]
        self.assertEqual(feedback["checkpoint"], "accuracy")
        self.assertEqual(feedback["reason"], REASON)
        self.assertEqual(feedback["fix"], FIX)

        for call in generator.call_args_list:
            self.assertEqual(
                call.kwargs["learned_guidance"], MEMORY["guidance"]
            )

        changes = result["attempts"][1]["changes_from_previous"]
        self.assertIn("-RAG always requires a vector store.", changes)
        self.assertIn("+Other retrieval methods can use database lookups.", changes)

    def test_repeated_failures_stop_at_the_configured_limit(self):
        for retries in (0, 1, 2):
            with self.subTest(max_retries=retries):
                attempts = retries + 1
                result, generator, evaluator, memory = execute(
                    [FIRST_DRAFT] * attempts,
                    [evaluation("accuracy") for _ in range(attempts)],
                    max_retries=retries,
                )
                self.assertEqual(result["status"], "NEEDS_REVIEW")
                self.assertIsNone(result["lesson"])
                self.assertEqual(len(result["attempts"]), attempts)
                self.assertEqual(generator.call_count, attempts)
                self.assertEqual(evaluator.call_count, attempts)
                memory.assert_called_once()

                with saved_run(result) as folder:
                    self.assertFalse((folder / "final_lesson.txt").exists())
                    log = json.loads(
                        (folder / "rejection_log.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(len(log), attempts)
                    self.assertIsNone(log[-1]["changes_on_retry"])

    def test_evaluator_error_never_accepts_a_lesson(self):
        result, generator, evaluator, _ = execute(
            [FIRST_DRAFT],
            [RuntimeError("Simulated invalid evaluation")],
        )
        self.assertEqual(result["status"], "ERROR")
        self.assertIsNone(result["lesson"])
        self.assertIn("Simulated invalid evaluation", result["error"])
        self.assertEqual(generator.call_count, 1)
        self.assertEqual(evaluator.call_count, 1)

        with saved_run(result) as folder:
            self.assertFalse((folder / "final_lesson.txt").exists())
            report = json.loads(
                (folder / "run_log.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "ERROR")

    def test_rejection_log_records_reasons_and_actual_changes(self):
        result, _, _, _ = execute(
            [FIRST_DRAFT, PASSING_DRAFT],
            [evaluation("accuracy"), evaluation()],
        )
        with saved_run(result) as folder:
            self.assertEqual(folder.name, result["run_id"])
            self.assertEqual(
                (folder / "final_lesson.txt").read_text(encoding="utf-8"),
                PASSING_DRAFT,
            )
            log = json.loads(
                (folder / "rejection_log.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(log), 1)
            self.assertEqual(log[0]["failed_checks"][0]["reason"], REASON)
            self.assertEqual(log[0]["failed_checks"][0]["fix"], FIX)
            self.assertIs(log[0]["retry_passed"], True)
            self.assertEqual(
                log[0]["changes_on_retry"],
                (folder / "revision_2.patch").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "+Other retrieval methods can use database lookups.",
                log[0]["changes_on_retry"],
            )

    def test_invalid_retry_limits_are_rejected_before_generation(self):
        with (
            patch("lesson.main.load_memory") as memory,
            patch("lesson.main.generate_lesson") as generator,
            patch("lesson.main.evaluate_lesson") as evaluator,
        ):
            for invalid in (-1, 3, True, False, "2", None):
                with self.subTest(value=invalid):
                    with self.assertRaises(ValueError):
                        run_pipeline(TOPIC, max_retries=invalid)
            memory.assert_not_called()
            generator.assert_not_called()
            evaluator.assert_not_called()


if __name__ == "__main__":
    unittest.main()