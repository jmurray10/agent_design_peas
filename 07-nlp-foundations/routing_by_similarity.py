"""Classification without a classifier, run head to head against an LLM classifier.

The pattern from the source page, in three lines of arithmetic:

    1. embed a handful of example inputs for each category
    2. average them into a centroid, one vector per category
    3. embed the new input and route it to the nearest centroid

No model call, no training, no labels beyond the examples someone already wrote. It is
deterministic, it costs nothing per request, and it cannot invent a category.

Then the same inputs go to an LLM classifier and both decisions are printed side by
side. The disagreements are the point of the file. Neither router is the winner: they
fail in different directions, and knowing which direction is how you pick one.

The embedder is imported from rag_pipeline rather than copied, so both files share one
embedding function. Offline that function is the deterministic hash embedder, which
scores lexical overlap and nothing else. Every paraphrase that shares no vocabulary with
its category examples is expected to be routed badly here -- that failure is real and it
is on the screen, not hidden in a footnote.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root, for shared/
sys.path.insert(0, str(HERE))         # this directory, for rag_pipeline

from shared.llm import llm_call  # noqa: E402

# Sharing the embedder is deliberate rather than lazy: two routers that embed
# differently are not comparable, and neither are a corpus and a query. Python already
# puts a script's own directory on sys.path, but not when this file is imported or run
# through a wrapper, so the line above says it out loud.
from rag_pipeline import Embedder, cosine_similarity, l2_normalize, tokenize  # noqa: E402


# Four categories, each defined only by example inputs a support lead could have written
# in ten minutes. This is the entire "training set".
CATEGORIES: dict[str, list[str]] = {
    "billing": [
        "I need a refund for order 4471",
        "you charged me twice for the same subscription",
        "cancel my subscription and refund the last invoice",
        "the payment did not go through at checkout",
    ],
    "shipping": [
        "where is my parcel",
        "the tracking number has not updated in four days",
        "my package was marked delivered but never arrived",
        "can I change the delivery address on order 2210",
    ],
    "technical_support": [
        "the app crashes when I open settings",
        "the screen flickers after the latest update",
        "export to PDF fails with an error",
        "the device will not turn on after charging overnight",
    ],
    "account_security": [
        "I cannot sign in to my account",
        "someone logged in from a country I have never visited",
        "how do I reset my password",
        "enable two factor authentication on my account",
    ],
}

# Below this, the nearest centroid is not near enough to be evidence of anything. The
# threshold is the deterministic half of the router: without it, nearest-centroid always
# answers, and always answering is how a router silently misroutes novel input.
SIMILARITY_THRESHOLD = 0.15

ALLOWED_LABELS = tuple(CATEGORIES) + ("other",)

TEST_INPUTS: list[tuple[str, str]] = [
    ("I want my money back", "nlp_route_money_back"),
    ("please process a refund for order 9912", "nlp_route_refund"),
    ("the app crashes when I open the settings page", "nlp_route_app_crash"),
    ("my parcel has been sitting in the same city for six days", "nlp_route_stuck_parcel"),
    ("someone logged into my account from another country", "nlp_route_foreign_login"),
    ("do you offer bulk pricing for a 200 seat purchase", "nlp_route_bulk_pricing"),
    ("it stopped working after the update and I want compensation", "nlp_route_compensation"),
]


# -- router one: nearest centroid ----------------------------------------------------

def build_centroids(embedder: Embedder) -> dict[str, list[float]]:
    """Average each category's example vectors, then renormalize.

    Renormalizing matters: the mean of unit vectors is shorter than a unit vector, and
    how much shorter depends on how spread out the examples are. Without the second
    normalization a tightly worded category would score higher than a broad one purely
    because its examples agree with each other.
    """
    centroids: dict[str, list[float]] = {}
    for label, examples in CATEGORIES.items():
        vectors = embedder.embed(examples)
        dims = len(vectors[0])
        mean = [sum(vector[i] for vector in vectors) / len(vectors) for i in range(dims)]
        centroids[label] = l2_normalize(mean)
    return centroids


def route_by_similarity(
    text: str, centroids: dict[str, list[float]], embedder: Embedder
) -> tuple[str, float, list[tuple[str, float]]]:
    """Return (label, best score, every score). Label is "unrouted" below the threshold."""
    vector = embedder.embed([text])[0]
    scores = sorted(
        ((label, cosine_similarity(vector, centroid)) for label, centroid in centroids.items()),
        key=lambda pair: -pair[1],
    )
    best_label, best_score = scores[0]
    if best_score < SIMILARITY_THRESHOLD:
        return "unrouted", best_score, scores
    return best_label, best_score, scores


# -- router two: the LLM -------------------------------------------------------------

def build_routing_prompt(text: str) -> str:
    lines = ["Route this customer message to exactly one queue.", "", "Queues:"]
    for label, examples in CATEGORIES.items():
        lines.append(f"- {label}: e.g. {examples[0]!r}, {examples[1]!r}")
    lines.append("- other: none of the above")
    lines += [
        "",
        f"Message: {text}",
        "",
        "Reply with the queue name and nothing else.",
    ]
    return "\n".join(lines)


def parse_label(raw: str) -> str | None:
    """Normalize a model reply into one of ALLOWED_LABELS, or None if it is not one.

    Deterministic and deliberately strict. Tolerating "billing_and_technical_support"
    because it contains the substring "billing" is how a routing bug becomes a support
    queue nobody is watching.
    """
    candidate = raw.strip().splitlines()[0] if raw.strip() else ""
    candidate = candidate.split(":")[-1]
    candidate = candidate.strip().strip(".\"'` ").lower().replace(" ", "_").replace("-", "_")
    return candidate if candidate in ALLOWED_LABELS else None


def route_by_llm(text: str, mock_key: str) -> tuple[str | None, str]:
    """Return (validated label or None, the raw reply)."""
    # tier=small: pick one label from a five-item list. This is the small tier in its
    # purest form -- the answer is a single short string drawn from a set the prompt
    # already contains, so there is nothing to synthesise and nothing for the extra
    # capability of a larger model to act on, while its cost and latency would be paid
    # on every routed request. The engineering that matters here is not the model
    # choice, it is the parse-and-validate step below that refuses anything off the list.
    raw = llm_call(build_routing_prompt(text), mock_key=mock_key, tier="small")
    return parse_label(raw), raw


# -- the comparison ------------------------------------------------------------------

def elide(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def shared_token_count(text: str, label: str) -> int:
    """How many content words the input shares with a category's examples.

    Printed for disagreements so the lexical router's decision is explainable rather
    than mysterious. Zero shared words is a complete explanation of a zero-ish score.
    """
    if label not in CATEGORIES:
        return 0
    example_tokens = set()
    for example in CATEGORIES[label]:
        example_tokens.update(tokenize(example))
    return len(set(tokenize(text)) & example_tokens)


def main() -> None:
    print("=" * 100)
    print("Routing by embedding similarity, next to routing by LLM")
    print("=" * 100)
    print()

    embedder = Embedder()
    print(embedder.banner())
    print()

    centroids = build_centroids(embedder)
    example_count = sum(len(examples) for examples in CATEGORIES.values())
    print(f"Built {len(centroids)} centroids from {example_count} example inputs.")
    print(f"Embedder mode after building them: {embedder.mode} ({embedder.dims} dimensions).")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD:.2f}. Below it the router declines.")
    print("Model calls used to build the centroids: 0.")
    print()

    # Both routers run over every input before anything is printed, so the table below
    # stays a table instead of being interrupted by the shim's one-time mode banner.
    rows = []
    llm_calls = 0
    for text, mock_key in TEST_INPUTS:
        sim_label, sim_score, _all_scores = route_by_similarity(text, centroids, embedder)
        llm_label, raw = route_by_llm(text, mock_key)
        llm_calls += 1

        if llm_label is None:
            # Deterministic fallback: an unparseable label is not a routing decision.
            # The free, deterministic router is already sitting right there. This row is
            # neither an agreement nor a disagreement, so it is scored separately.
            agree = "n/a"
        else:
            agree = "yes" if sim_label == llm_label else "NO"

        rows.append((text, sim_label, sim_score, llm_label, raw, agree))

    print()
    header = (
        f"{'input':<58} {'similarity route':<18} {'score':>6}   {'LLM route':<22} agree"
    )
    print(header)
    print("-" * len(header))
    for text, sim_label, sim_score, llm_label, _raw, agree in rows:
        llm_shown = llm_label if llm_label is not None else "REJECTED -> similarity"
        print(
            f"{elide(text, 57):<58} {sim_label:<18} {sim_score:>6.3f}   {llm_shown:<22} {agree}"
        )

    print()

    agreements = sum(1 for row in rows if row[5] == "yes")
    comparable = sum(1 for row in rows if row[5] != "n/a")
    rejected = len(rows) - comparable
    print(f"Agreed on {agreements} of the {comparable} inputs where the LLM returned a usable label.")
    print(f"{rejected} reply was rejected by the validator and fell back to the similarity route.")
    print(f"Model calls: similarity router 0, LLM router {llm_calls}.")
    print()

    print("Every row the two routers did not simply agree on")
    print("-" * 100)
    for text, sim_label, sim_score, llm_label, raw, agree in rows:
        if agree == "yes":
            continue
        print(f"  input           {text!r}")
        print(f"  similarity      {sim_label} at {sim_score:.3f} against its best centroid")
        if llm_label is None:
            print(f"  LLM             returned {raw.strip()!r}")
            print(f"  validator       not one of {', '.join(ALLOWED_LABELS)}")
            print("                  so the reply was rejected rather than pattern-matched")
            print("  resolution      the similarity decision stands, and no retry call fired")
        elif llm_label == "other":
            print("  LLM             other")
            print("  reason          centroid space has no 'other'. Nobody can write example")
            print("                  inputs for a category defined by not being the other four,")
            print("                  so the threshold is the only thing stopping the similarity")
            print("                  router from picking a queue anyway.")
        else:
            overlap = shared_token_count(text, llm_label)
            print(f"  LLM             {llm_label}")
            print(f"  reason          the input shares {overlap} content words with the")
            print(f"                  {llm_label!r} examples, so the hash embedder had nothing")
            print("                  to score on")
        print()

    print("The tradeoff, stated plainly")
    print("-" * 100)
    print("  Similarity routing is deterministic, costs no model call, and returns the same")
    print("  queue for the same input forever. It cannot route a category nobody wrote")
    print("  examples for: centroid space has no 'other', only 'nearest, and here is how")
    print("  near'. The threshold above is what turns that into a refusal rather than a")
    print("  confident wrong answer.")
    print()
    print("  LLM routing handles paraphrase and novelty, and will answer 'other' when the")
    print("  message deserves it. It costs a call per request, it is nondeterministic at")
    print("  temperature above zero, and it can return a label that does not exist -- which")
    print("  is why parse_label() validates against a fixed list and falls back rather than")
    print("  trusting the reply.")
    print()
    print("  Offline, the embedder is the hash backend, which compares vocabulary and not")
    print("  meaning. Paraphrases that share no words with their category examples are")
    print("  routed on almost no evidence, and the rows above show it happening. Set")
    print("  EMBEDDING_API_KEY and rerun to see what a trained model does with the same")
    print("  seven inputs and the same sixteen examples.")


if __name__ == "__main__":
    main()
