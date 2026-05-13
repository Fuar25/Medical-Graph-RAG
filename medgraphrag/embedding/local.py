from sentence_transformers import SentenceTransformer

_MODEL_PATH = "/home/sbh/.cache/huggingface/hub/Qwen3-Embedding-8B"
_EMBED_DIM = 1024
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            _MODEL_PATH,
            model_kwargs={"device_map": "cuda:0"},
            processor_kwargs={"padding_side": "left"},
            truncate_dim=_EMBED_DIM,
        )
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()
