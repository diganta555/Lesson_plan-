RUBRIC = {
    "accuracy": (
    "All technical claims agree with the supplied reference facts. "
    "RAG is described as retrieving external information and supplying "
    "it to a language model for answering. "
    "If the lesson explains chunking, embeddings, or vector stores, "
    "it must explicitly state BOTH that this is one common implementation, "
    "rather than a requirement for all RAG, AND that other retrieval "
    "methods are possible without that vector-store setup, such as "
    "matching words or looking up database records. "
    "Naming a database without explaining it as an alternative retrieval "
    "method is insufficient. "
    "There must be no contradictory claim elsewhere that every RAG "
    "system requires embeddings or a vector store. "
    "No claim that RAG guarantees correct answers, always searches "
    "the internet, or requires retraining whenever documents change."
),
    "beginner_language": (
        "The lesson assumes no previous AI knowledge. "
        "It introduces ideas using plain language before technical detail. "
        "A beginner can understand the explanations without outside reading."
    ),
    "worked_example": (
        "Includes a complete example showing a question, the actual "
        "retrieved passage, the information supplied to the model, "
        "and an answer that follows from that passage."
    ),
    "jargon": (
        "Every technical term and acronym is explained in plain language "
        "when first introduced. This includes RAG, LLM, embedding, chunk, "
        "vector store, and hallucination if those terms are used."
    ),
    "coverage": (
        "Explains what RAG is, why it is useful, and how it works. "
        "Distinguishes preparing documents from answering a question. "
        "Explains retrieval, supplying context, and generating an answer. "
        "Includes at least two meaningful limitations."
    ),
    "teaching_flow": (
        "Introduces the concept before its mechanism. "
        "Explains the mechanism before the worked example. "
        "Then explains limitations and finishes with a recap."
    ),
}

def all_checks_pass(results: dict[str, bool]) -> bool:
    if set(results) != set(RUBRIC):
        raise ValueError("Results must contain exactly the six rubric checks.")

    if any(type(value) is not bool for value in results.values()):
        raise TypeError("Every result must be True or False.")

    return all(results.values())

if __name__ == "__main__":
    print("Our rubric checkpoints:")
    for number, checkpoint in enumerate(RUBRIC, start=1):
        print(f"{number}. {checkpoint}")

    example_results = {checkpoint: True for checkpoint in RUBRIC}
    print("\nExample with all checks passing:",
          all_checks_pass(example_results))

    example_results["jargon"] = False
    print("Example with jargon failing:",
          all_checks_pass(example_results))