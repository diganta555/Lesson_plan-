# Self-Evaluating RAG Lesson Generator

Generates a standalone beginner lesson about Retrieval-Augmented Generation
(RAG), evaluates it against six Boolean checkpoints, and revises rejected
lessons up to two times.

The current reference facts and topic-specific checks support
Introduction to RAG.

## Setup

Requirements: Python 3.12, uv, and an OpenAI API key.

```bash
uv sync
cp .env.example .env
```

Set OPENAI_API_KEY in .env. Optional GENERATOR_MODEL and EVALUATOR_MODEL
settings default to gpt-4.1-mini.

## Run

Command-line interface:

```bash
uv run python -m lesson.main
```

Streamlit interface:

```bash
uv run streamlit run frontend/app.py
```

Open http://localhost:8501. Enter RAG or Introduction to RAG, choose the
revision limit, and click Generate lesson.

## Architecture

- generator.py creates or revises the complete lesson using reference
  facts, the rubric, failed-check feedback, and learned guidance.
- evaluator.py independently evaluates six checkpoints using structured
  Boolean results with reasons, evidence lines, and requested fixes.
- main.py controls acceptance and termination. Every checkpoint must pass.
- memory.py persists run history, failures, feedback, and prompt guidance
  in SQLite.
- save_run() writes drafts, evaluations, rejection reasons, and actual
  revision diffs.
- frontend/app.py presents the workflow and displays accepted lessons.

Python functions keep this bounded workflow easy to inspect and test.
Separate generation and evaluation calls make the feedback explicit.
Typed outputs validate the evaluator's format; reference facts and
calibration help assess its judgments.

## Rubric

Every checkpoint passes or fails; there are no scores or partial credit.

1. Accuracy: technical claims agree with the reference facts.
2. Beginner language: explanations assume no previous AI knowledge.
3. Worked example: question, retrieved passage, context, and supported answer.
4. Jargon: technical terms are explained when first introduced.
5. Coverage: what, why, how, preparation, answering, and limitations.
6. Teaching flow: concept, motivation, mechanism, example, limitations, recap.

Accuracy checks explicitly require embeddings and vector stores to be
described as one common implementation, with alternative retrieval methods.

## Retry and error handling

The default limit is three lesson attempts: one initial attempt and two
revisions. Each revision receives the previous draft and failed-check details.

- ACCEPTED: every checkpoint passed; final_lesson.txt is written.
- NEEDS_REVIEW: the lesson exhausted its revision allowance.
- ERROR: generation or evaluation could not produce a usable result.

An invalid evaluator result gets one correction attempt. If it remains
invalid, the run ends with ERROR.

## Persistent memory and evolution

Failures are counted once per checkpoint per run, even if several revisions
fail that checkpoint. Two separate failing runs activate a curated guidance
instruction. Human feedback can activate guidance immediately.

This adapts generator instructions while preserving the hard acceptance
requirements. The rubric was also tightened during development after an
accuracy weakness was identified through feedback and calibration.

Memory is scoped to the topic, reference facts, and rubric. It is stored
locally under data/ and is created for each installation.

```bash
uv run python -m lesson.memory_status
```

## Deliberate-error demonstration

The demo draft contains this false claim:

"Every RAG system must use embeddings and a vector store."

Run the actual evaluator and revision loop:

```bash
uv run python -m lesson.main \
  --initial-lesson demo/incorrect_rag_lesson.txt \
  --max-retries 2
```

The expected accuracy verdict for the supplied draft is FAIL. Subsequent
drafts are generated and evaluated normally; verdicts are never forced.

## Validation

Controller unit tests use mocked model calls:

```bash
uv run python -m unittest discover -s tests -v
```

Accuracy calibration calls the real evaluator:

```bash
uv run python -m lesson.calibrate \
  --lesson examples/rejected_lesson.txt
```

During development, six controller tests and three accuracy calibration
cases passed. Calibration covers specific accuracy cases; model judgments
can still be inconsistent.

## Output evidence

examples/ contains the accepted lesson, accepted-run report, rejected draft,
revision-run report, rejection log, and actual revision patch.

New runs are saved under outputs/<run_id>/.

The generated lesson is available only after acceptance. Evaluations and
rejection logs remain available for inspecting unsuccessful runs.