
from cat.mad_hatter.decorators import tool
from cat.log import log
from pathlib import Path
import json
from typing import Dict

# === Costanti ===
USER_STATUS_PATH = Path("cat/static/user_status.json")


def _load_json_safe(path: Path, default: dict) -> dict:
    """Carica un file JSON in modo sicuro."""
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"[deep_search] JSON load error on {path}: {e}")
        return default


def _get_metadata_filter(user: str) -> Dict[str, bool]:
    """
    Costruisce i metadati di filtro per l'utente.
    Restituisce un dict con i tag attivi (True) + {user: True}.
    """
    user_status = _load_json_safe(USER_STATUS_PATH, {})
    tags_for_user = user_status.get(user, {})

    metadata = {}
    for tag, tag_obj in tags_for_user.items():
        if isinstance(tag_obj, dict) and tag_obj.get("status", False):
            metadata[tag] = True

    metadata[user] = True  # flag utente
    
    log.info(f'[deep_search] metadata filter for user {user}: {metadata}')
    return metadata

@tool(return_direct=False)
def declarative_search(question: str, cat):
    """
    Use this tool when you have a question that could be answered by a document stored in your memory.
    - The input: the informations you need to know. (Maximum 2 questions)
    - The output must include:
        * Useful information derived from the context (facts, concise conclusions, references if available).
        * Remaining doubts or open points that still need clarification.
        * Up to 2 follow-up questions meeting the constraints above (≤2, ≤30 words, single topic).
    """        
    
    # """
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

    # Ottieni i metadati di filtro per l'utente corrente
    try:
        user = cat.user_id
        metadata_filter = _get_metadata_filter(user)
    except Exception as e:
        log.error(f"[deep_search] Error getting user metadata: {e}")
        metadata_filter = None

    # Esegui la ricerca con i metadati di filtro
    raw = cat.memory.vectors.declarative.recall_memories_from_embedding(
        embedding=cat.embedder.embed_query(question),
        k=4,
        threshold=0.7,
        metadata=metadata_filter  # <-- AGGIUNTO IL FILTRO METADATI
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
   
    deepsearch_return = "\n\n".join(blocks) if blocks else ""
    # cat.send_ws_message(deepsearch_return)
    
    # Se dopo la dedup non resta nulla, restituisci stringa vuota
    return deepsearch_return


    # raw = cat.memory.vectors.declarative.recall_memories_from_embedding(
    #     embedding=cat.embedder.embed_query(question),
    #     k=4,
    #     threshold=0.7,
    # )
    # cat.send_ws_message(question)
    # if not raw:
    #     return ""  # sempre stringa

    # blocks = []
    # seen = set()
    # idx = 1

    # for doc, _score, _emb, _id in raw:
    #     text = (getattr(doc, "page_content", "") or "").strip()
    #     if not text or text in seen:
    #         continue
    #     seen.add(text)

    #     meta = getattr(doc, "metadata", {}) or {}
    #     source = str(meta.get("source", "n/d"))

    #     blocks.append(f"From document: {source}\nContext_{idx}:\n{text}")
    #     idx += 1

    
    # deepsearch_return= "\n\n".join(blocks) if blocks else ""
    # # cat.send_ws_message(deepsearch_return)
    # # Se dopo la dedup non resta nulla, restituisci stringa vuota
    # return deepsearch_return
