"""The RAG pipeline from the source page, offline phase then online phase, end to end.

    OFFLINE   load documents -> chunk -> embed -> store
    ONLINE    embed the percept -> top-k by cosine -> inject -> generate -> validate

Every step except the generation is deterministic infrastructure. That is the shape the
whole series argues for: the LLM is one component in the middle of a pipeline that a
person can single-step through, not the pipeline itself.

Two embedding backends behind one interface:

    api    a trained model, used when EMBEDDING_API_KEY is set
    hash   deterministic hashed bag of tokens and character 4-grams, used otherwise

The hash backend is NOT a small embedding model. It has no semantic content whatsoever
beyond the lexical overlap that was designed into it -- "refund" and "money back" are
strangers to it. It exists so that this file runs on a fresh machine with no key, no
install and no network, and so that the control flow you read is the control flow that
runs. Read the README before drawing any conclusion from a hash-mode similarity score.

shared/llm.py deliberately exposes no embedding function, so the API path lives here.
It is guarded end to end: a missing key, a bad URL or a failed request all fall back to
the hash backend and the run continues offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.llm import llm_call  # noqa: E402

try:  # optional: scores the whole store in one matrix product when present
    import numpy as np
except ImportError:  # pragma: no cover - exercised on a bare Python install
    np = None


KB_DIR = Path(__file__).resolve().parent / "kb"

# Inside the 256-4096 range the source page gives for dense vectors, but chosen for a
# duller reason: at 512 dims two hashed features collided hard enough to rank the
# "final sale" passage above the laptop return window on the first test query. Hash
# collisions are a real failure mode of the hashing trick and more buckets is the real
# fix. A trained model has no equivalent knob -- its width is fixed at training time.
HASH_DIMS = 2048
MAX_CHUNK_CHARS = 400
TOP_K = 3

# Dropped before hashing. Function words are in every document, so they push every hash
# vector toward the same corner of the space and flatten the ranking.
STOPWORDS = frozenset(
    """a an and are as at be been but by can cannot did do does for from had has have how i
    if in into is it its me my not of on or our out so that the their there they this to was
    we were what when where which who why will with you your""".split()
)

QUERIES: list[tuple[str, str]] = [
    ("How long do I have to return a laptop?", "nlp_rag_returns"),
    ("My package says delivered but I never got it.", "nlp_rag_shipping"),
    ("How do I reset my password?", "nlp_rag_account"),
    ("What is the warranty on refurbished items?", "nlp_rag_warranty"),
]


# -- vector math ---------------------------------------------------------------------

def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2_normalize(vector: list[float]) -> list[float]:
    length = dot(vector, vector) ** 0.5
    if length == 0.0:
        return vector
    return [value / length for value in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Kept explicit for readability. Every vector here is normalized on the way in, so
    the store scores with a bare dot product and gets the same number for less work."""
    denominator = (dot(a, a) ** 0.5) * (dot(b, b) ** 0.5)
    if denominator == 0.0:
        return 0.0
    return dot(a, b) / denominator


# -- the offline embedder ------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


def hash_embed(text: str, dims: int = HASH_DIMS) -> list[float]:
    """Deterministic pseudo-embedding: the hashing trick over tokens and 4-grams.

    Each feature is hashed to a bucket and to a sign. Signed buckets let collisions
    cancel on average instead of always inflating similarity, which is the reason the
    trick is used in production feature pipelines rather than plain modulo counting.

    Character 4-grams carry half the weight of whole tokens so that "refund" and
    "refunds", or "cancel" and "cancelled", are near neighbours rather than unrelated
    features. That is morphology, not meaning: nothing in here knows that "money back"
    and "refund" are the same request.
    """
    vector = [0.0] * dims
    for token in tokenize(text):
        _add_feature(vector, token, 1.0, dims)
        padded = f"^{token}$"
        for start in range(len(padded) - 3):
            _add_feature(vector, padded[start:start + 4], 0.5, dims)
    return l2_normalize(vector)


def _add_feature(vector: list[float], feature: str, weight: float, dims: int) -> None:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    bucket = value % dims                      # low bits choose the bucket
    sign = 1.0 if (value >> 63) & 1 else -1.0  # top bit chooses the sign
    vector[bucket] += sign * weight


class Embedder:
    """One embedding function used for both the corpus and the query.

    Using the same function on both sides is not a style preference. Cosine similarity
    between vectors from two different models is meaningless -- the axes do not line up.
    This class exists so there is exactly one place that decision is made.
    """

    def __init__(self) -> None:
        self.api_key = os.environ.get("EMBEDDING_API_KEY")
        self.base_url = os.environ.get("EMBEDDING_API_BASE_URL", "https://api.openai.com/v1")
        # The default matches the embedding_model named in the source page's sensor config.
        self.model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.mode = "api" if self.api_key else "hash"
        self.dims: int | None = None if self.mode == "api" else HASH_DIMS

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.mode == "api":
            vectors = self._embed_via_api(texts)
            if vectors is not None:
                self.dims = len(vectors[0])
                return [l2_normalize(vector) for vector in vectors]
            # One failure is enough. Retrying per batch would turn a dead endpoint into
            # a very slow run, and the offline path is always available.
            self.mode = "hash"
            self.dims = HASH_DIMS
            print("[embeddings] API call failed; falling back to hash embeddings for this run.")
        return [hash_embed(text) for text in texts]

    def _embed_via_api(self, texts: list[str]) -> list[list[float]] | None:
        """POST to an OpenAI-compatible /embeddings endpoint using only urllib.

        Returns None on any failure at all. Nothing about this path is allowed to break
        the offline run, so the except clause is deliberately as wide as it looks.
        """
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            return [item["embedding"] for item in body["data"]]
        except Exception as error:  # network, auth, schema -- all handled the same way
            print(f"[embeddings] {type(error).__name__}: {error}")
            return None

    def banner(self) -> str:
        if self.mode == "api":
            # Intent, not yet fact -- nothing has been embedded when this prints. Both
            # scripts confirm the mode again after the first real batch.
            return (
                f"[embeddings] mode=api  model={self.model}  endpoint={self.base_url}\n"
                f"             EMBEDDING_API_KEY found; vectors will come from a trained model"
            )
        return (
            f"[embeddings] mode=hash  dims={HASH_DIMS}  no EMBEDDING_API_KEY found\n"
            f"             deterministic hash of tokens and character 4-grams -- lexical\n"
            f"             overlap only, no learned meaning"
        )


# -- chunking and storage ------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    title: str
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#{self.index}"


def chunk_document(path: Path, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split one document into retrievable passages.

    Paragraph boundaries first, because the author already grouped related sentences.
    Oversized paragraphs are then split on sentence boundaries so no single chunk can
    swallow a whole page of the context window.
    """
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else path.stem
    body = "\n".join(lines[1:]) if lines and lines[0].startswith("#") else raw

    passages: list[str] = []
    for paragraph in body.split("\n\n"):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            passages.append(paragraph)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.?!])\s+", paragraph):
            if current and len(current) + len(sentence) + 1 > max_chars:
                passages.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            passages.append(current)

    return [
        Chunk(doc_id=path.stem, index=index, title=title, text=text)
        for index, text in enumerate(passages, start=1)
    ]


class VectorStore:
    """The smallest thing that deserves the name. A real one adds an index; the query
    semantics below are what Pinecone, pgvector and the rest are approximating."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self._matrix = None

    def add(self, chunk: Chunk, vector: list[float]) -> None:
        self.chunks.append(chunk)
        self.vectors.append(vector)

    def finalize(self) -> str:
        """Freeze the store and report which scoring path queries will take."""
        if np is not None and self.vectors:
            self._matrix = np.array(self.vectors, dtype=float)
            return "numpy (one matrix-vector product per query)"
        return "pure python (one dot product per stored chunk)"

    def query(self, vector: list[float], top_k: int = TOP_K) -> list[tuple[float, Chunk]]:
        if self._matrix is not None:
            scores = (self._matrix @ np.array(vector, dtype=float)).tolist()
        else:
            scores = [dot(stored, vector) for stored in self.vectors]
        # Ties break on insertion order so the same corpus always ranks the same way.
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
        return [(scores[i], self.chunks[i]) for i in order]


# -- the online phase ----------------------------------------------------------------

def build_prompt(question: str, hits: list[tuple[float, Chunk]]) -> str:
    passages = "\n\n".join(
        f"[S{position}] ({chunk.chunk_id}) {chunk.text}"
        for position, (_score, chunk) in enumerate(hits, start=1)
    )
    return (
        "You are a support agent. Answer the question using ONLY the passages below.\n"
        "Cite every passage you used with its marker, like [S1].\n"
        "If the passages do not contain the answer, reply exactly: not in the knowledge base\n"
        "\n"
        f"{passages}\n"
        "\n"
        f"Question: {question}\n"
        "Answer:"
    )


def cited_passages(answer: str, hit_count: int) -> list[int]:
    """Which retrieved passages did the answer actually cite?

    Deterministic, and the only thing standing between a fluent answer and a fluent
    answer about a product that does not exist. Markers outside the retrieved range are
    discarded rather than trusted -- an invented [S9] is evidence against the answer.
    """
    found = {int(marker) for marker in re.findall(r"\[S(\d+)\]", answer)}
    return sorted(index for index in found if 1 <= index <= hit_count)


def answer_query(
    question: str, mock_key: str, store: VectorStore, embedder: Embedder
) -> tuple[bool, str]:
    """Run one online cycle. Returns (answer survived validation, the prompt that was sent)."""
    print("-" * 78)
    print(f"QUERY  {question}")

    query_vector = embedder.embed([question])[0]
    hits = store.query(query_vector, TOP_K)

    print(f"  retrieved top {len(hits)} of {len(store.chunks)} chunks by cosine similarity")
    for position, (score, chunk) in enumerate(hits, start=1):
        preview = chunk.text if len(chunk.text) <= 88 else chunk.text[:85] + "..."
        print(f"    [S{position}] {score:+.3f}  {chunk.chunk_id:<20} {preview}")

    prompt = build_prompt(question, hits)

    # tier=mid: retrieval has already done the hard part, so this call is bounded --
    # restate what three supplied passages say and tag each claim with its marker. It is
    # not a one-of-N label, so "small" is the wrong shape: the citation format has to
    # hold across several sentences, and under the check below a dropped [S1] does not
    # produce a slightly worse answer, it produces a rejected one. It is not open-ended
    # reasoning either, so "frontier" would buy latency and cost for a job that ends at
    # the edge of the retrieved text.
    answer = llm_call(prompt, mock_key=mock_key, tier="mid")

    print(f"  ANSWER {answer.strip()}")

    citations = cited_passages(answer, len(hits))
    if citations:
        print(f"  grounding check PASSED -- cites {', '.join(f'[S{i}]' for i in citations)}")
        return True, prompt

    # The fallback that makes the pipeline safe to deploy: an answer that cannot point
    # at retrieved text does not get to be the answer.
    print("  grounding check FAILED -- the answer cites none of the retrieved passages")
    print("  falling back to the retrieved passages verbatim, unsummarised:")
    for _score, chunk in hits:
        print(f"    ({chunk.chunk_id}) {chunk.text}")
    print("  and flagging the query for human review")
    return False, prompt


def main() -> None:
    print("=" * 78)
    print("RAG pipeline: offline indexing, then online retrieval and generation")
    print("=" * 78)
    print()

    embedder = Embedder()
    print(embedder.banner())
    print()

    print("OFFLINE PHASE")
    documents = sorted(KB_DIR.glob("*.md"))
    store = VectorStore()
    for path in documents:
        chunks = chunk_document(path)
        vectors = embedder.embed([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            store.add(chunk, vector)
        print(f"  {path.name:<22} {len(chunks)} chunks embedded and stored")
    print(f"  store holds {len(store.chunks)} vectors of {embedder.dims} dimensions")
    print(f"  embedder mode after indexing: {embedder.mode}")
    print(f"  scoring backend: {store.finalize()}")
    print()

    print("ONLINE PHASE")
    grounded = 0
    for position, (question, mock_key) in enumerate(QUERIES):
        survived, prompt = answer_query(question, mock_key, store, embedder)
        grounded += int(survived)
        if position == 0:
            print()
            print("  the prompt that produced the answer above, in full:")
            for line in prompt.splitlines():
                print(f"  | {line}")
            print("  (shown once; every query builds the same shape)")
    print("-" * 78)
    print()

    rejected = len(QUERIES) - grounded
    print(f"{grounded} of {len(QUERIES)} answers cited a retrieved passage and were kept.")
    print(f"{rejected} {'was' if rejected == 1 else 'were'} rejected by the deterministic "
          "grounding check.")
    if rejected:
        print("The rejected answer offers a plan that appears nowhere in kb/, and the check")
        print("caught it without being told what the right answer was.")
    else:
        # Said plainly rather than narrated around. An earlier recording had one answer
        # improvise past the corpus and this run does not, so the check has nothing to
        # show -- which is a fact about the recording, not a property of the check.
        print("Nothing was rejected on this run. The check is in the code either way, and")
        print("it is worth having on the days a model improvises rather than only on the")
        print("days it does not.")
    print()
    print(f"Embedding mode for this run: {embedder.mode}.")
    print("Retrieval quality above is a property of that backend, not of RAG. In hash mode")
    print("a query only finds a passage it already shares vocabulary with. Set")
    print("EMBEDDING_API_KEY to score the same corpus with a trained model and see how far")
    print("the paraphrases move.")


if __name__ == "__main__":
    main()
