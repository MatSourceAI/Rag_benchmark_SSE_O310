"""SSE RAG components; online dependencies are loaded only when requested."""
from importlib import import_module
__all__ = ["GoogleEmbedder", "MilvusRetriever", "MilvusHybridRetriever", "MilvusConfig", "CrossEncoderReranker", "OpenAICompatGenerator", "GoogleGenerator", "RAGPipeline", "HyDE", "MultiQuery", "reciprocal_rank_fusion"]
def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = ".pipeline" if name == "RAGPipeline" else ".components"
    if name in ("HyDE", "MultiQuery", "reciprocal_rank_fusion"):
        module = ".query_transforms"
    return getattr(import_module(module, __name__), name)
