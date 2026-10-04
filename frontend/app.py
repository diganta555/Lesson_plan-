import json
import streamlit as st
from lesson.main import run_pipeline, save_run
from lesson.memory import remember_run

TOPIC = "Introduction to RAG"
TOPIC_NAMES = {
    "rag",
    "introduction to rag",
    "retrieval-augmented generation",
    "rag (retrieval-augmented generation)",
    "teach me rag",
    "explain rag",
    "explain rag simply",
}

st.set_page_config(
    page_title="RAG Lesson Studio",
    page_icon="📚",
    layout="wide",
)
st.title("📚 RAG Lesson Studio")
st.write("Generate a beginner lesson that must pass six quality checks.")
st.caption("Try RAG, Introduction to RAG, or Teach me RAG.")
st.session_state.setdefault("run_view", None)

with st.form("lesson_request"):
    requested_topic = st.text_input("Lesson request", value=TOPIC)
    max_retries = st.selectbox(
        "Maximum revisions",
        options=[0, 1, 2],
        index=2,
    )
    submitted = st.form_submit_button(
        "Generate lesson",
        type="primary",
    )

if submitted:
    st.session_state["run_view"] = None
    normalized_topic = " ".join(requested_topic.casefold().split())

    if normalized_topic not in TOPIC_NAMES:
        st.error(
            "This version generates beginner RAG lessons. "
            "Try 'Teach me RAG'."
        )
        st.stop()

    try:
        with st.spinner("Generating, evaluating, and revising if needed..."):
            result = run_pipeline(TOPIC, max_retries=max_retries)
            result["learner_request"] = requested_topic.strip()
            result["memory_after"] = remember_run(result)
            folder = save_run(result)

            rejections = json.loads(
                (folder / "rejection_log.json").read_text(
                    encoding="utf-8"
                )
            )
            st.session_state["run_view"] = {
                "result": result,
                "rejections": rejections,
            }
    except Exception as error:
        st.error(
            f"Could not complete the request: "
            f"{type(error).__name__}: {error}"
        )

view = st.session_state["run_view"]
if view is None:
    st.stop()

result = view["result"]
rejections = view["rejections"]

st.caption(f"Run: {result['run_id']} · Status: {result['status']}")

first, second, third = st.columns(3)
first.metric("Attempts", len(result["attempts"]))
second.metric("Rejected drafts", len(rejections))
third.metric("Memory version used", result["memory_before"]["version"])

lesson_tab, evaluation_tab, revision_tab = st.tabs(
    ["Lesson", "Evaluation", "Revision history"]
)

with lesson_tab:
    if result["status"] == "ACCEPTED":
        st.success("All six quality checks passed.")
        st.markdown(result["lesson"])
        st.download_button(
            "Download lesson",
            data=result["lesson"],
            file_name=f"lesson_{result['run_id']}.txt",
            mime="text/plain",
            on_click="ignore",
        )
    elif result["status"] == "ERROR":
        st.error(result.get("error", "The run could not finish."))
    else:
        st.warning(
            "The lesson did not pass within the revision limit. "
            "Human review is required."
        )

with evaluation_tab:
    rows = [
        {
            "Attempt": attempt["attempt"],
            "Checkpoint": checkpoint,
            "Result": "PASS" if check["passed"] else "FAIL",
            "Reason": check["reason"],
        }
        for attempt in result["attempts"]
        for checkpoint, check in attempt["evaluation"].items()
    ]
    if rows:
        st.table(rows)
    else:
        st.info("No complete checkpoint results were recorded.")

with revision_tab:
    if not rejections:
        st.info("No lesson rejections were recorded for this run.")

    for rejection in rejections:
        with st.expander(f"Rejected attempt {rejection['attempt']}"):
            for check in rejection["failed_checks"]:
                st.write(f"**{check['checkpoint']}**: {check['reason']}")
                st.write(f"**Requested fix:** {check['fix']}")

            if rejection["changes_on_retry"]:
                st.code(rejection["changes_on_retry"], language="diff")
            else:
                st.info("No completed revision followed this rejection.")

    st.download_button(
        "Download rejection log",
        data=json.dumps(rejections, indent=2, ensure_ascii=False),
        file_name=f"rejections_{result['run_id']}.json",
        mime="application/json",
        on_click="ignore",
    )

st.download_button(
    "Download full run report",
    data=json.dumps(result, indent=2, ensure_ascii=False),
    file_name=f"run_{result['run_id']}.json",
    mime="application/json",
    on_click="ignore",
)