from cat.mad_hatter.decorators import tool

@tool(return_direct=False)
def declarative_search(question: str, cat):

    """Guidelines:
    - Deterministic: the same question always produces the same answer.
    - Ask clear, specific, and well-focused questions.
    - Iterations:
        * Provide a maximum of 2 follow-up questions per iteration.
        * Each question must deepen ONE specific topic (not multiple distinct topics).
        * Questions must not be generic and must be no longer than 30 words each.
    - Simulate a "reading chain": provide a structured, sequential synthesis of the sections/checks performed (e.g., "steps verified" or "check-list"), explaining what was verified and why, **without revealing private thoughts, hidden reasoning, or intermediate calculations**.
    - Do not skip intermediate or related sections of the document. If a section applies, include it explicitly in your analysis before moving forward.
    - The output must include:
        * Useful information derived from the context (facts, concise conclusions, references if available).
        * Remaining doubts or open points that still need clarification.
        * Up to 2 follow-up questions meeting the constraints above (≤2, ≤30 words, single topic).
    """
    # """
    # Use this tool when you have a question that could be answered by a document stored in your memory.
    # Input: the informations you need to know.
    # Guidelines:
    # - If you ask the same question, you will always receive the same answer
    # - Ask clear, specific, and well-focused questions.
    # - Iterations:
    #     * Provide a maximum of 2 follow-up questions per iteration.
    #     * Each question must deepen ONE specific topic (not multiple distinct topics).
    #     * Questions must not be generic and must be no longer than 30 words each.
    # - Simulate a "reading chain": provide a structured, sequential synthesis of the sections/checks performed (e.g., "steps verified" or "check-list"), explaining what was verified and why, **without revealing private thoughts, hidden reasoning, or intermediate calculations**.
    # - The output must include:
    #     * Useful information derived from the context (facts, concise conclusions, references if available).
    #     * Remaining doubts or open points that still need clarification.
    #     * Up to 2 follow-up questions meeting the constraints above (≤2, ≤30 words, single topic).
    # """
    # """
    # Use this tool when you have a question that could be answered by a document stored in your memory.
    # Input: the informations you need to know.    
    # Guidelines:
    # - If you ask the same question, you will always receive the same answer (deterministic behavior).
    # - Formulate clear and specific questions to improve the quality of the answer.
    # - With each iteration, follow-up questions should be progressively more detailed and precise to reduce ambiguity and narrow open points.
    # - The output must include:
    #     * Useful information derived from the context.
    #     * Any remaining doubts or open points that still need clarification.
    # """
    raw = cat.memory.vectors.declarative.recall_memories_from_embedding(
        embedding=cat.embedder.embed_query(question),
        k=4,
        threshold=0.7,
    )
    cat.send_ws_message(question)
    if not raw:
        return ""  # sempre stringa

    blocks = []
    seen = set()
    idx = 1

    for doc, _score, _emb, _id in raw:
        text = (getattr(doc, "page_content", "") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)

        meta = getattr(doc, "metadata", {}) or {}
        source = str(meta.get("source", "n/d"))

        blocks.append(f"From document: {source}\nContext_{idx}:\n{text}")
        idx += 1

    
    deepsearch_return= "\n\n".join(blocks) if blocks else ""
    # cat.send_ws_message(deepsearch_return)
    # Se dopo la dedup non resta nulla, restituisci stringa vuota
    return deepsearch_return
