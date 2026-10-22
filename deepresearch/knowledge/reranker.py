"""Optional bounded local cross-encoder; failures are handled by RRF fallback."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import math


def document_for_rerank(hit):
    text = "\n".join([hit.get("title", ""), " > ".join(hit.get("heading", [])), hit["content"]])
    # Keep the target passage first; truncating the optional tail cannot remove it.
    neighbors = hit.get("_rerank_neighbors")
    if neighbors:
        text += "\n\n前文（仅供定位）：\n"+neighbors['before']+"\n后文（仅供定位）：\n"+neighbors['after']
    return text


def load_cross_encoder(name, **kwargs):
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(name, **kwargs)
    if model.device.type == "cpu":
        # Materialize owned contiguous weights. The local Apple CPU runtime can
        # SIGBUS on a file-backed classifier parameter; Python fallback cannot
        # catch a process signal. Copying preserves values/dtype and avoids it.
        for parameter in model.model.parameters():
            parameter.data = parameter.detach().clone().contiguous()
    return model


class CrossEncoderReranker:
    def __init__(self, model: str, timeout: float = 5):
        self.name, self.timeout = model, timeout
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-rerank")
        self.lock = Lock()
        self.pending = None
        self.model = None

    def __call__(self, query, hits):
        def rank():
            if self.model is None:
                self.model = load_cross_encoder(self.name)
            scores = self.model.predict([(query, document_for_rerank(h)) for h in hits])
            if len(scores) != len(hits) or any(not math.isfinite(float(score)) for score in scores):
                raise ValueError("invalid reranker scores")
            return [h for _, h in sorted(zip(scores, hits), key=lambda pair: -float(pair[0]))]
        with self.lock:
            if self.pending is not None and not self.pending.done():
                raise TimeoutError("reranker busy; use RRF")
            self.pending = self.executor.submit(rank)
            future = self.pending
        return future.result(timeout=self.timeout)
