# Embeddings, retrieval and routing

**Source:** reference/07-support-nlp-foundations.md

## The claim

Cosine similarity over dense vectors is the whole mechanism behind semantic search, RAG
retrieval and embedding-based routing, and it is small enough to write out by hand and
check line by line. These three scripts do exactly that: `similarity.py` shows two
messages with no words in common landing at 0.85 in a dense space and 0.00 in a sparse
one, `rag_pipeline.py` runs the offline-then-online pipeline end to end against the
documents in `kb/`, and `routing_by_similarity.py` replaces an LLM classifier with four
averaged vectors and prints every input where the two disagree. If the claim is wrong,
the disagreement is on the screen.

**The vectors in `similarity.py` are illustrative and were authored, not trained.** The
six named dimensions -- `refund_intent`, `politeness`, and so on -- were typed in by a
person who decided what each sentence means. No model produced them, and they exist so
that a reader can check the arithmetic against numbers they can read. That is a statement
about hand-built vectors, and it is unaffected by anything else in this repository.

**The offline embeddings are deterministic hashes.** With no `EMBEDDING_API_KEY` set,
`rag_pipeline.py` and `routing_by_similarity.py` embed text by hashing its tokens and
character 4-grams into 2048 buckets. The same string always yields the same vector, which
is the only property they are meant to have. They carry no semantic content beyond the
lexical overlap that was designed into them: in hash space, "refund" and "money back" are
strangers, and the routing table shows that failure rather than hiding it. Nothing here
was trained on anything, and no similarity score printed by these scripts is evidence
about how a trained embedding model behaves.

The generation and classification calls are a separate matter. Those go through
`shared/llm.py`, and with no backend configured they replay what a real model returned to
these exact prompts: `claude-sonnet-5` for the four RAG answers, `claude-haiku-4-5` for
the seven routing labels, both recorded 2026-08-04 and stored in `shared/transcripts/`.
The answers below are a model's, not an author's. They are also one run on one date, so
they are evidence about what that model did then, not a prediction about what it does now.

Set `EMBEDDING_API_KEY` (optionally with `EMBEDDING_API_BASE_URL` and `EMBEDDING_MODEL`,
which default to the OpenAI endpoint and `text-embedding-3-small`) and both scripts call a
real embeddings API instead, and say so in their first line of output. `shared/llm.py`
exposes no embedding function on purpose, so that path lives in `rag_pipeline.py`, guarded
so that a missing key, a bad URL or a failed request all fall back to hashes and the run
finishes offline.

## Run it

    python 07-nlp-foundations/similarity.py
    python 07-nlp-foundations/rag_pipeline.py
    python 07-nlp-foundations/routing_by_similarity.py
    python 07-nlp-foundations/real_embeddings.py

`similarity.py`, which calls no model at all:

    The pair the whole section is about
      A: 'I want my money back'
      B: 'please process a refund'
      words in common: 0 []

      sparse (bag-of-words, 27 dims)
        dot(A, B)      = 0.0000
        cosine(A, B)   = 0.0000
      dense (hand-authored, 6 dims)
        dot(A, B)      = 0.9799
        cosine(A, B)   = 0.8522

    Cross-check against numpy
      numpy 2.4.2 agrees with the from-scratch functions
      largest disagreement over all 36 pairs: 2.22e-16

`rag_pipeline.py`:

    [embeddings] mode=hash  dims=2048  no EMBEDDING_API_KEY found

    OFFLINE PHASE
      store holds 14 vectors of 2048 dimensions
      scoring backend: numpy (one matrix-vector product per query)

    QUERY  How long do I have to return a laptop?
      retrieved top 3 of 14 chunks by cosine similarity
        [S1] +0.176  returns-policy#2     Laptops, tablets and other opened electronics...
      ANSWER You have 14 days from the delivery date to return a laptop, since it's an opened electronic item with a shorter return window [S1].
      grounding check PASSED -- cites [S1]
    ...
    4 of 4 answers cited a retrieved passage and were kept.
    0 was rejected by the deterministic grounding check.

`routing_by_similarity.py`:

    input                                                      similarity route    score   LLM route              agree
    I want my money back                                       unrouted            0.000   billing                NO
    please process a refund for order 9912                     billing             0.271   billing                yes
    the app crashes when I open the settings page              technical_support   0.418   technical_support      yes
    my parcel has been sitting in the same city for six days   shipping            0.305   shipping               yes
    someone logged into my account from another country        account_security    0.489   account_security       yes
    do you offer bulk pricing for a 200 seat purchase          unrouted            0.043   other                  NO
    it stopped working after the update and I want compens...  technical_support   0.207   technical_support      yes

    Agreed on 5 of the 7 inputs where the LLM returned a usable label.
    0 reply was rejected by the validator and fell back to the similarity route.
    Model calls: similarity router 0, LLM router 7.

All three run with no key, no install and no network. numpy is used where it is present --
to cross-check the hand-written math, and to score the vector store in one matrix product --
and every script produces the same numbers without it.

## What the recorded run does not show you

Two deterministic guards are in the code and did not fire on this recording, and the
difference between "did not fire" and "is not there" is worth being explicit about.

The grounding check in `rag_pipeline.py` rejected nothing: `claude-sonnet-5` cited a
retrieved passage in all four answers, so all four were kept. Under the hand-written mock
responses this repository used to ship, one answer was written to cite nothing so that the
rejection path ran on every offline run. That was a person demonstrating a code path, not
a model failing to cite. The check still runs on every answer, and it still rejects one
that cites nothing -- there is now no offline run in which it does.

The label validator in `routing_by_similarity.py` rejected nothing either: all seven
`claude-haiku-4-5` replies were labels from the fixed list. It is still the reason a reply
of "probably billing?" cannot become a routing decision.

Neither absence is evidence that a model always cites or always answers in-vocabulary.
It is one run, one date, one model per script. Point a key at either script and you are
measuring today's model instead of reading yesterday's.

### The hand-authored vectors, checked against real ones

`similarity.py` uses six named dimensions a person filled in, and says so at the top. It
also makes a claim it cannot check from inside itself: that a trained model gives you
hundreds of dimensions nobody labelled, and that the arithmetic run over them is character
for character the arithmetic in that file.

`real_embeddings.py` checks it. It imports `cosine_similarity`, `bag_of_words` and
`build_vocabulary` from `similarity.py` rather than reimplementing them, and runs the same
functions over 384-dimension vectors from `sentence-transformers/all-MiniLM-L6-v2`,
reached through `shared/embeddings.py` on the same record-or-replay terms as every model
call in this repository.

The row worth running it for is the control:

      sparse     dense   phrases
       0.000     0.597   'I want my money back' / 'please process a refund'
       0.400     0.357   'I want my money back' / 'I want to buy more'

Bag-of-words does not merely fail to see that the first pair means one thing. It ranks the
second pair as the more similar of the two, because those phrases share "I" and "want",
and it is not hedging when it does so. The dense column puts the control last of the four
pairs it scores.

Same function, same dot products, same magnitudes written out by hand. Only the
coordinates changed, and the ordering inverted. That is what an embedding buys, stated as
narrowly as it can be: the geometry was always available and the coordinates were the hard
part.

## What changed

In `rag_pipeline.py` the LLM replaced exactly one component: the step that turns three
retrieved passages into a sentence. Chunking, embedding, storage, cosine ranking, top-k
selection, prompt assembly and the citation check that can reject the generated answer are
all deterministic code around it.

`routing_by_similarity.py` runs the swap in the other direction. The classifier that page
05 implements with an LLM call is replaced by four centroids and a dot product, with a
similarity threshold that declines rather than guesses. The LLM router still runs on the
same inputs so both decisions print side by side, and its reply is validated against a
fixed label list before it is allowed to be a routing decision.

The two rows marked NO are the argument. "I want my money back" shares no content word
with the billing examples, so the hash embedder scores it 0.000 and the threshold turns
that into a refusal; the model routed it to billing. "Do you offer bulk pricing" belongs
to a category nobody can write examples for, because it is defined by not being the other
four; centroid space has no "other", and the model answered "other".

## What it costs

The grounding check proves citation, not truth: an invented claim carrying a plausible
`[S1]` would pass it. Similarity routing gives up novel categories entirely -- centroid
space has no "other" -- and needs example inputs per category up front. LLM routing gives
up determinism and a fixed cost per request, and can return a label that does not exist,
which is why the validator exists. Offline, the hash embedder gives up meaning altogether:
it matches vocabulary, so paraphrases route on almost no evidence.
