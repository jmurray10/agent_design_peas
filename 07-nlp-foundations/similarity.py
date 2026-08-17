"""Dot product and cosine similarity, written out by hand.

Two customer messages that share no words at all -- "I want my money back" and "please
process a refund" -- sit close together in a dense space and exactly at zero in a sparse
one. That gap is the entire reason embeddings exist, and it is why an LLM can serve as
the agent function f: P -> a when the percepts are human sentences.

The vectors in EMBEDDINGS are HAND-AUTHORED. Nothing here was trained. Six named
dimensions a person can read, filled in by a person, so the arithmetic is inspectable.
A trained model gives you 256 to 4096 dimensions nobody labelled and nobody can read --
but the arithmetic it runs on them is character for character the arithmetic below.
That is the honest claim this file supports: the math is small and checkable; the
learned coordinates are the part you cannot write by hand.

Standard library only. If numpy happens to be installed it is used at the end to check
the hand-written functions against a battle-tested implementation, and skipped silently
if it is not.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

try:  # numpy is optional here -- it verifies the math, it does not perform it
    import numpy as np
except ImportError:  # pragma: no cover - exercised on a bare Python install
    np = None


# The six axes of the hand-built space. A trained embedding model has no such list:
# its dimensions are whatever gradient descent found useful and are not individually
# meaningful. Naming them here is the concession that makes the demo readable.
DIMENSIONS: tuple[str, ...] = (
    "refund_intent",
    "delivery_issue",
    "device_fault",
    "account_access",
    "politeness",
    "urgency",
)

# Authored, not trained. Every number below was typed by a human who decided what the
# sentence means. See the module docstring and the README before quoting any of it.
EMBEDDINGS: dict[str, list[float]] = {
    "I want my money back":                 [0.95, 0.02, 0.05, 0.00, 0.05, 0.45],
    "please process a refund":              [0.93, 0.02, 0.02, 0.00, 0.55, 0.15],
    "reimburse me for order 4471":          [0.90, 0.05, 0.00, 0.00, 0.35, 0.25],
    "where is my parcel":                   [0.05, 0.95, 0.00, 0.00, 0.20, 0.45],
    "the screen flickers after the update": [0.02, 0.00, 0.95, 0.05, 0.10, 0.30],
    "I cannot sign in to my account":       [0.00, 0.02, 0.20, 0.95, 0.10, 0.50],
}

HEADLINE_PAIR = ("I want my money back", "please process a refund")


# -- the math, from scratch ----------------------------------------------------------

def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Sum of elementwise products: how aligned two vectors are, unnormalized."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def magnitude(a: Sequence[float]) -> float:
    """Euclidean length. sqrt(a . a), which is why dot() is the only primitive needed."""
    return math.sqrt(dot(a, a))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product divided by both magnitudes: -1 (opposite) to +1 (same direction).

    Normalizing is what makes the measure about *direction* rather than *loudness*. A
    long angry message and a short polite one about the same problem point the same way.
    """
    denominator = magnitude(a) * magnitude(b)
    # A zero vector has no direction, so the angle is undefined rather than zero.
    # Returning 0.0 keeps every caller from having to special-case an empty message.
    if denominator == 0.0:
        return 0.0
    return dot(a, b) / denominator


# -- the sparse baseline we are arguing against --------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocabulary(texts: Sequence[str]) -> list[str]:
    words: set[str] = set()
    for text in texts:
        words.update(tokenize(text))
    return sorted(words)


def bag_of_words(text: str, vocabulary: Sequence[str]) -> list[float]:
    """One dimension per vocabulary word, counting occurrences. Mostly zeros."""
    counts = {word: 0 for word in vocabulary}
    for token in tokenize(text):
        if token in counts:
            counts[token] += 1
    return [float(counts[word]) for word in vocabulary]


# -- output --------------------------------------------------------------------------

def print_space() -> None:
    print("Hand-authored embedding space")
    print("  dimensions:", ", ".join(DIMENSIONS))
    print()
    header = "  " + " ".join(f"{name[:9]:>10}" for name in DIMENSIONS)
    print(f"  {'phrase':<40}{header}")
    for index, (phrase, vector) in enumerate(EMBEDDINGS.items(), start=1):
        row = " ".join(f"{value:>10.2f}" for value in vector)
        print(f"  P{index} {phrase:<37}  {row}")
    print()


def print_headline_pair(vocabulary: Sequence[str]) -> None:
    left, right = HEADLINE_PAIR
    shared = set(tokenize(left)) & set(tokenize(right))

    print("The pair the whole section is about")
    print(f"  A: {left!r}")
    print(f"  B: {right!r}")
    print(f"  words in common: {len(shared)} {sorted(shared)}")
    print()

    sparse_a = bag_of_words(left, vocabulary)
    sparse_b = bag_of_words(right, vocabulary)
    print(f"  sparse (bag-of-words, {len(vocabulary)} dims)")
    print(f"    dot(A, B)      = {dot(sparse_a, sparse_b):.4f}")
    print(f"    cosine(A, B)   = {cosine_similarity(sparse_a, sparse_b):.4f}")
    print("    a sparse vector can only report word overlap, and there is none")
    print()

    dense_a = EMBEDDINGS[left]
    dense_b = EMBEDDINGS[right]
    print(f"  dense (hand-authored, {len(DIMENSIONS)} dims)")
    print(f"    dot(A, B)      = {dot(dense_a, dense_b):.4f}")
    print(f"    |A|            = {magnitude(dense_a):.4f}")
    print(f"    |B|            = {magnitude(dense_b):.4f}")
    print(f"    cosine(A, B)   = {cosine_similarity(dense_a, dense_b):.4f}")
    print("    the refund_intent axis dominates; politeness and urgency differ and cost")
    print("    the pair some similarity, which is what makes it 0.85 and not 1.00")
    print()


def print_matrix(title: str, vectors: Sequence[Sequence[float]], phrases: Sequence[str]) -> None:
    print(title)
    labels = [f"P{i}" for i in range(1, len(phrases) + 1)]
    print("  " + " " * 40 + "".join(f"{label:>7}" for label in labels))
    for row_index, phrase in enumerate(phrases):
        scores = "".join(
            f"{cosine_similarity(vectors[row_index], vectors[col_index]):>7.2f}"
            for col_index in range(len(phrases))
        )
        print(f"  {labels[row_index]} {phrase:<37}{scores}")
    print()


def check_against_numpy() -> None:
    if np is None:
        print("Cross-check against numpy")
        print("  numpy not installed -- skipped. Nothing above needed it.")
        print()
        return

    worst = 0.0
    for a in EMBEDDINGS.values():
        for b in EMBEDDINGS.values():
            mine = cosine_similarity(a, b)
            va, vb = np.array(a), np.array(b)
            theirs = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
            worst = max(worst, abs(mine - theirs))

    print("Cross-check against numpy")
    print(f"  numpy {np.__version__} agrees with the from-scratch functions")
    print(f"  largest disagreement over all {len(EMBEDDINGS) ** 2} pairs: {worst:.2e}")
    print()


def main() -> None:
    phrases = list(EMBEDDINGS)
    vocabulary = build_vocabulary(phrases)

    print("=" * 78)
    print("Dot product and cosine similarity, from scratch")
    print("=" * 78)
    print()
    print_space()
    print_headline_pair(vocabulary)
    print_matrix(
        f"Cosine similarity in the hand-authored dense space ({len(DIMENSIONS)} dims)",
        [EMBEDDINGS[phrase] for phrase in phrases],
        phrases,
    )
    print_matrix(
        f"Cosine similarity in bag-of-words space ({len(vocabulary)} dims)",
        [bag_of_words(phrase, vocabulary) for phrase in phrases],
        phrases,
    )
    check_against_numpy()
    print("Read the two matrices side by side. The dense one groups P1, P2 and P3 -- three")
    print("ways of asking for money back -- above 0.85 while pushing the delivery, device")
    print("and account messages below 0.30. The sparse one is near-empty off the diagonal")
    print("because those phrases were written to share almost no vocabulary.")
    print()
    print("The coordinates above were authored by hand. A trained model would supply them")
    print("from data instead. The dot products would be computed exactly as they are here.")


if __name__ == "__main__":
    main()
