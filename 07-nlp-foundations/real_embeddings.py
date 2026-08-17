"""The same arithmetic, on coordinates nobody wrote by hand.

`similarity.py` makes a claim it cannot check. Its vectors are hand-authored -- six named
dimensions a person filled in -- and it says of a trained model:

    A trained model gives you 256 to 4096 dimensions nobody labelled and nobody can read
    -- but the arithmetic it runs on them is character for character the arithmetic below.

This file checks it. It imports `cosine_similarity`, `bag_of_words` and `build_vocabulary`
from `similarity.py` -- the same functions, not reimplementations -- and runs them over
real embeddings fetched from a trained model instead of the authored ones.

If the claim is true, the only thing that changes is where the numbers came from, and
that is what happens: the sparse column puts "I want my money back" and "please process a
refund" at exactly zero because they share no words, while the dense column puts them
close because a model that read the internet knows they are the same request.

The control row is the one worth running for. Sparse ranks "I want my money back" as
closer to "I want to buy more" than to "please process a refund", because those two share
"I" and "want". Dense ranks it last of the four. The arithmetic is identical in both
columns; only the coordinates changed, and the ordering inverted.

Run it:

    python 07-nlp-foundations/real_embeddings.py

With no Hugging Face token configured it replays recorded vectors. With one, it fetches
them live. Nothing here invents a vector -- an input with no recording raises.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from shared.embeddings import embed  # noqa: E402
from similarity import (  # noqa: E402
    bag_of_words,
    build_vocabulary,
    cosine_similarity,
)

# The pair that motivates the whole file: no shared words, same meaning.
PAIRS = [
    ("I want my money back", "please process a refund"),
    ("my order never arrived", "where is my package"),
    ("how do I reset my password", "I am locked out of my account"),
    # A control. These two share a word and mean different things, so a dense space
    # should not be fooled by the overlap.
    ("I want my money back", "I want to buy more"),
]

SOURCE = "07_nlp_foundations__real_embeddings"


def main() -> None:
    phrases = sorted({p for pair in PAIRS for p in pair})
    vocabulary = build_vocabulary(phrases)

    print("The same cosine function, over two kinds of coordinates")
    print()
    print("  sparse: bag-of-words over a vocabulary built from these phrases alone.")
    print("  dense : a trained sentence embedding, fetched or replayed.")
    print("  Both columns are computed by cosine_similarity() from similarity.py.")
    print()

    vectors = {phrase: embed(phrase, tier="small", source=SOURCE) for phrase in phrases}
    width = len(next(iter(vectors.values())))

    print(f"  embedding width: {width} dimensions, none of them named")
    print(f"  vocabulary size: {len(vocabulary)} words, every one of them readable")
    print()
    print(f"  {'sparse':>8}  {'dense':>8}   phrases")

    for left, right in PAIRS:
        sparse = cosine_similarity(bag_of_words(left, vocabulary),
                                   bag_of_words(right, vocabulary))
        dense = cosine_similarity(vectors[left], vectors[right])
        print(f"  {sparse:>8.3f}  {dense:>8.3f}   {left!r} / {right!r}")

    print()
    print("Read the control row against the first one. Sparse scores the control 0.400")
    print("and the refund pair 0.000, so bag-of-words says 'I want my money back' is more")
    print("like 'I want to buy more' than it is like 'please process a refund'. It is not")
    print("merely uninformative there, it is ranked backwards, and it is confident about")
    print("it: the two phrases really do share 'I' and 'want'.")
    print()
    print("Dense reverses that ordering. The control is the lowest of the four at 0.357")
    print("and the refund pair the second highest. Same four pairs, same function, and")
    print("the rank order inverts on the one comparison that matters.")
    print()
    print("What did not change: cosine_similarity is the same function in both columns,")
    print("imported from similarity.py rather than rewritten here. The dot products and")
    print("magnitudes are the ones written out by hand in that file. A trained model")
    print("supplied better coordinates; it did not supply better arithmetic.")
    print()
    print("That is the whole of what an embedding buys, stated as narrowly as it can be:")
    print("the geometry was always available, and the coordinates were the hard part.")


if __name__ == "__main__":
    main()
