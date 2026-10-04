import json
import os
from openai import OpenAI
from lesson.references import REFERENCE_FACTS
from lesson.rubric import RUBRIC

GENERATOR_INSTRUCTIONS = """
Teach the topic to a complete beginner in a standalone 500–800 word lesson.
Use these sections: What it is, Why it matters, How it works,
Worked example, Limitations, Recap.
Define technical terms and acronyms when first introduced.
Ground technical claims in the supplied reference facts.
Clearly label fictional examples.
For RAG, show the question, retrieved passage, context supplied, and answer.
Describe chunking, embeddings, and vector stores as one common approach.
RAG uses configured information sources; do not restrict all RAG to
prepared documents or claim it automatically searches the whole internet.
Explain at least two limitations. Never guarantee correct answers.
Meet every supplied rubric checkpoint.
On revision, improve the full previous lesson using every failed check.
Return only the complete lesson.
"""

def generate_lesson(
    topic: str,
    previous_lesson: str = "",
    failed_checks: list[dict] | None = None,
    learned_guidance: dict[str, str] | None = None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_actual_api_key":
        raise ValueError("Set your actual OPENAI_API_KEY in .env.")

    client = OpenAI(api_key=api_key, timeout=60.0, max_retries=1)
    response = client.responses.create(
        model=os.getenv("GENERATOR_MODEL", "gpt-4.1-mini"),
        instructions=GENERATOR_INSTRUCTIONS,
        input=json.dumps({
            "task": "revise" if previous_lesson else "create",
            "topic": topic,
            "reference_facts": REFERENCE_FACTS,
            "rubric": RUBRIC,
            "previous_lesson": previous_lesson,
            "failed_checks": failed_checks or [],
        }),
        max_output_tokens=3000,
        store=False,
    )

    if response.status != "completed" or not response.output_text.strip():
        raise RuntimeError("Generator did not return a complete lesson.")

    return response.output_text.strip()