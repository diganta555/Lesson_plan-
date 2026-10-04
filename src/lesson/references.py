REFERENCE_FACTS = """
RAG means Retrieval-Augmented Generation.
It combines finding relevant external information with generating an answer.

A large language model, or LLM, processes and generates language.
Retrieval means finding relevant information.
Generation means producing the answer.
Context is information supplied to the model while it answers.

A common document-preparation process collects documents, splits them into
smaller passages called chunks, creates embeddings, and indexes them.
An embedding is a numerical representation used to compare text by meaning.
A vector store keeps these representations and associated passages for search.
Other retrieval approaches can also support RAG.

At question time, the application retrieves relevant passages, supplies those
passages and the question to the model, and asks it to produce an answer.
Document preparation and answering a question are separate stages.
Ordinary document updates do not require retraining the language model.
RAG does not inherently search the entire internet.

Conservative quality requirements:
Retrieved information may be irrelevant, incomplete, or outdated.
A model may still produce unsupported statements, called hallucinations.
RAG does not guarantee correct answers.
Invented examples must be clearly labeled as fictional.

Sources:
https://arxiv.org/abs/2005.11401
https://developers.openai.com/api/docs/guides/retrieval
"""