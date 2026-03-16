from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embedding_model


def semantic_similarity(text_a: str, text_b: str) -> float:
    model = get_embedding_model()
    emb = model.encode([str(text_a), str(text_b)], convert_to_numpy=True)
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return float(sim)


def contains_target_word(generated_cue: str, target_word: str) -> bool:
    return str(target_word) in str(generated_cue)


def brevity_score(text: str, ideal_min: int = 4, ideal_max: int = 20) -> float:
    length = len(str(text))
    if length == 0:
        return 0.0
    if ideal_min <= length <= ideal_max:
        return 1.0
    if length < ideal_min:
        return max(0.0, length / ideal_min)
    return max(0.0, ideal_max / length)


def compute_proxy_score(generated_cue: str, target_word: str, reference_cue: str) -> dict:
    if not generated_cue or not str(generated_cue).strip():
        return {
            "semantic_similarity": 0.0,
            "brevity_score": 0.0,
            "target_leak_penalty": 1.0,
            "proxy_score": 0.0,
        }

    sim = semantic_similarity(generated_cue, reference_cue)
    brev = brevity_score(generated_cue)
    leak_penalty = 1.0 if contains_target_word(generated_cue, target_word) else 0.0

    proxy_score = (0.70 * sim) + (0.30 * brev) - (0.50 * leak_penalty)
    proxy_score = max(0.0, min(1.0, proxy_score))

    return {
        "semantic_similarity": sim,
        "brevity_score": brev,
        "target_leak_penalty": leak_penalty,
        "proxy_score": proxy_score,
    }