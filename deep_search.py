from cat.mad_hatter.decorators import tool

@tool(return_direct=False)
def declarative_search(question: str, cat):
    """
    To be used when you are looking for any answer.
    Input: the question you need to answer.
    """
    raw = cat.memory.vectors.declarative.recall_memories_from_embedding(
        embedding=cat.embedder.embed_query(question),
        k=2,
        threshold=0.7,
    )
    cat.send_ws_message(question,"chat")
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
