# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
rag.py
------
Step 3: the answering brain.

Takes a question -> finds relevant chunks -> asks Claude to answer USING ONLY
those chunks, with citations. Crucially, we instruct Claude to say
"I don't have a documented answer" when the chunks don't contain it.

That honest-abstention behaviour is one of our headline differentiators:
in a safety-critical plant, a confident wrong answer is worse than "I don't know."
"""

from __future__ import annotations
from dataclasses import dataclass

from .embed_store import VectorStore
from .ingest import Chunk
from .llm import chat

_MIN_RELEVANCE = 0.25        # cosine gate, used when no reranker is present
# Cross-encoder gate. Off-topic queries score ~0.00; even casual/typo'd real
# questions clear this comfortably, so 0.12 keeps honest abstention while not
# refusing a judge who types informally.
_MIN_RERANK = 0.12

_CAPABILITIES = (
    "I'm the **Plant Operations Brain** — I read your plant's documents "
    "(manuals, SOPs, maintenance logs, incidents, drawings, emails) and help you "
    "with:\n\n"
    "- 💬 **Answering questions** with cited sources (and I say *I don't know* "
    "rather than guess)\n"
    "- 🛡️ **Safety checks** before a job — STOP / CAUTION / SAFE\n"
    "- 🔧 **Root-cause analysis** & predictive maintenance\n"
    "- 📋 **Compliance** gap detection and audit evidence\n"
    "- ⚠️ **Conflict detection** between documents (and I can draft the fix)\n"
    "- 📚 **Lessons learned** from incident history\n"
    "- 🧠 **Capturing tribal knowledge** from retiring experts\n\n"
    "Ask me something like *\"What is the lockout procedure for Pump 7?\"* or "
    "*\"Why does the FD-2 fan keep tripping?\"*"
)


def _conversational_reply(q: str) -> str | None:
    """
    Handle everyday conversation (greetings, thanks, 'what can you do')
    WITHOUT an API call. Returns a reply string, or None to fall through to
    normal document-grounded RAG. Kept deliberately narrow so real questions
    are never intercepted.
    """
    import re
    s = q.strip().lower()

    # NEVER intercept a message that references specific plant content — an
    # equipment tag (P-7, FD-2), any identifier with a digit, or a domain noun.
    # Otherwise "what is this valve" / "hi P-7 status" get swallowed.
    if re.search(r"\b[a-z]{1,4}-?\d", s) or any(ch.isdigit() for ch in s):
        return None
    _DOMAIN = ("valve", "pump", "fan", "boiler", "motor", "compressor", "bearing",
               "seal", "temperature", "pressure", "vibration", "lockout", "loto",
               "procedure", "sop", "permit", "inspection", "maintenance",
               "lubric", "trip", "alarm", "limit", "incident", "equipment",
               "certificate", "regulation", "audit", "compliance", "risk")
    if any(w in s for w in _DOMAIN):
        return None

    words = re.findall(r"[a-z']+", s)
    n = len(words)
    greet = {"hi", "hello", "hey", "hii", "helo", "hola", "namaste",
             "yo", "greetings", "gm", "morning", "hiya"}
    thanks = {"thanks", "thank", "thx", "ty", "cool", "great", "nice", "awesome",
              "ok", "okay", "bye", "goodbye", "welcome", "good"}
    filler = {"there", "a", "the", "you", "u", "day", "morning", "team", "bro"}

    # Greeting ONLY if the whole message is greeting/filler words.
    if words and all(w in greet | filler for w in words):
        return "👋 Hello! " + _CAPABILITIES
    if words and all(w in thanks | greet | filler for w in words):
        return "You're welcome! Ask me anything about your plant's documents. 🙂"

    # Capability questions — match the WHOLE normalised query (not substring),
    # so "what is this valve" is never caught by "what is this".
    s_norm = re.sub(r"[^a-z ]", "", s).strip()
    meta_exact = {
        "what can you do", "what do you do", "who are you", "what are you",
        "how can you help", "how can you help me", "what can i ask",
        "what can i ask you", "capabilities", "how do you work",
        "what can u do", "what is this", "what do you know", "help",
        "what are your features", "what can you help with", "who r u",
    }
    if s_norm in meta_exact or s in {"?", "help"}:
        return _CAPABILITIES
    return None


@dataclass
class Answer:
    text: str
    sources: list[Chunk]
    confident: bool
    score: float = 0.0   # top retrieval-relevance score (0..1) — shown in the UI


_SYSTEM_PROMPT = """You are the Plant Operations Brain, an assistant for industrial \
plant engineers and technicians. You answer questions about equipment, maintenance, \
safety procedures, and regulatory compliance using ONLY the provided document excerpts.

Rules you must follow strictly:
1. Answer ONLY from the provided excerpts. Never use outside knowledge or guess.
2. Cite your sources inline like [Source 1], [Source 2] after each claim.
3. If the excerpts do not contain the answer, reply exactly:
   "I don't have a documented answer for this. No source covers it."
   Do NOT invent a plausible-sounding answer -- in a plant, a wrong answer is dangerous.
4. If excerpts seem to CONTRADICT each other, point out the conflict explicitly.
5. Be concise and practical -- a technician may be reading this on a phone on the floor.
"""


def _format_excerpts(hits: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for i, (chunk, score) in enumerate(hits, start=1):
        loc = f"{chunk.source_file}" + (f", p.{chunk.page}" if chunk.page else "")
        blocks.append(f"[Source {i}] (from {loc})\n{chunk.text}")
    return "\n\n".join(blocks)


class PlantBrain:
    def __init__(self, store: VectorStore, graph=None, reranker=None):
        self.store = store
        self.graph = graph  # optional KnowledgeGraph for GraphRAG hybrid retrieval
        self.reranker = reranker   # optional cross-encoder second stage
        # Map chunk_id -> Chunk so the graph can pull in connected chunks by id.
        self._by_id = {c.chunk_id: c for c in store.chunks}

    def retrieve(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        hits, _, _ = self._retrieve_scored(query, k)
        return hits

    _STOP = {"what", "the", "and", "for", "with", "this", "that", "how", "why",
             "who", "where", "when", "is", "are", "was", "of", "in", "on", "a",
             "an", "to", "does", "do", "can", "should", "tell", "me", "about",
             "give", "show", "explain"}

    _ID_RE = None   # compiled lazily

    def _corpus_ids(self) -> set[str]:
        """
        All identifier-like tokens (P-7, WO-228, 24BDS085, FB-2025-09…) that
        actually exist in the corpus, normalized (lowercase, dashes/spaces
        stripped). Built once; used to refuse questions about entities that
        exist nowhere — the classic entity-substitution hallucination trap.
        """
        if not hasattr(self, "_ids_cache"):
            import re
            rx = re.compile(r"\b[A-Za-z]{1,6}[- ]?\d[\w-]*\b")
            ids = set()
            for c in self.store.chunks:
                for m in rx.findall(c.text):
                    ids.add(re.sub(r"[-\s]", "", m).lower())
                for m in rx.findall(c.source_file):
                    ids.add(re.sub(r"[-\s]", "", m).lower())
            self._ids_cache = ids
            self._id_rx = rx
        return self._ids_cache

    def _classify_ids(self, query: str):
        """Return (known_ids, unknown_ids) for the ID-like tokens in a query."""
        import re
        ids = self._corpus_ids()
        known, unknown = [], []
        for m in self._id_rx.findall(query) if hasattr(self, "_id_rx") else []:
            # The spaced form can greedily grab a preceding word ("is 24bds085")
            # — ignore matches whose letter prefix is an ordinary English word.
            if " " in m and m.split()[0].lower() in self._STOP:
                continue
            tok = re.sub(r"[-\s]", "", m).lower()
            # Known iff the token IS a corpus id, or a long-enough fragment OF
            # one ("bds085" ⊂ "24bds085"). Reverse containment is banned:
            # corpus "P-9" must NOT legitimise a query about "P-99".
            # Membership is checked at ANY length (P-7 → "p7" is a valid tag);
            # only UNKNOWN tokens must be >=3 chars to avoid flagging noise.
            if any(tok == cid or (len(tok) >= 4 and tok in cid) for cid in ids):
                known.append(m)
            elif len(tok) >= 3:
                unknown.append(m)
        return known, unknown

    def _unknown_ids(self, query: str) -> list[str]:
        """
        IDs in the query that exist nowhere — but ONLY report them when there is
        no KNOWN id to anchor the answer. "Compare P-7 and P-99" still answers
        (about P-7; the model can't fabricate P-99 data); "what is P-99" refuses.
        """
        known, unknown = self._classify_ids(query)
        return [] if known else unknown

    def _lexical_scored(self, query: str, limit: int = 8):
        """Like _lexical_candidates but returns (score, distinct_terms, chunk)."""
        import re
        terms = [t.lower() for t in re.findall(r"[A-Za-z0-9][\w-]{2,}", query)
                 if t.lower() not in self._STOP]
        if not terms:
            return []
        scored = []
        for c in self.store.chunks:
            text = c.text.lower()
            fname = c.source_file.lower()
            s, distinct = 0.0, 0
        # (loop below mirrors _lexical_candidates scoring)
            for t in terms:
                digity = any(ch.isdigit() for ch in t)
                w = 3.0 if digity else 1.0
                tn = t.replace("-", "")
                fn = fname.replace("-", "").replace("_", "")
                tx = text  # plain check first (cheap)
                hit = False
                if t in fname or (digity and tn in fn):
                    s += w * 2
                    hit = True
                if t in tx or (digity and tn in tx.replace("-", "").replace(" ", "")):
                    s += w
                    hit = True
                if hit:
                    distinct += 1
            if s > 0:
                scored.append((s, distinct, c))
        scored.sort(key=lambda x: -x[0])
        return scored[:limit]

    def _lexical_candidates(self, query: str, limit: int = 8) -> list[Chunk]:
        """
        Exact-match keyword scan over chunk text AND file names. Dense embeddings
        are weak at alphanumeric identifiers ("24BDS085", "WO-228") — a literal
        substring hit is the strongest possible signal for those, so this layer
        guarantees such chunks reach the reranker. Digit-bearing terms score
        highest; file-name hits count double (IDs often live in the file name).
        """
        return [c for _, _, c in self._lexical_scored(query, limit)]

    def _retrieve_scored(self, query: str, k: int = 5):
        """
        Two-stage hybrid retrieval used everywhere. Returns (hits, top_score,
        passed_gate) so `ask()` can abstain on the SAME signal it retrieves on.
          1. vector search over-fetches candidates (fast, recall-oriented),
          2. the GPU cross-encoder reranks them for precision (if available) —
             its score cleanly separates relevant from irrelevant, so it also
             drives honest abstention,
          3. GraphRAG adds knowledge-graph-connected chunks the query mentions.
        """
        if self.reranker is not None:
            candidates = self.store.search(query, k=max(k * 4, 12))
            seen0 = {c.chunk_id for c, _ in candidates}
            # Hybrid: exact keyword/ID hits join the dense candidates so the
            # reranker gets to judge them ("24BDS085" never embeds well).
            lex = self._lexical_scored(query)
            for _, _, c in lex:
                if c.chunk_id not in seen0:
                    candidates.append((c, 0.0))
                    seen0.add(c.chunk_id)
            hits = self.reranker.rerank(query, candidates, top_k=k)
            top = hits[0][1] if hits else 0.0
            passed = top >= _MIN_RERANK
            # Keyword override — this is what saves CASUAL, messy questions the
            # cross-encoder scores low ("p7 keeps breaking why", "tell me about
            # the pump that keeps failing"). If 2+ DISTINCT query words (or an
            # exact ID) appear verbatim in one document, that's real evidence:
            # promote it and pass. Off-topic queries ("who won the world cup")
            # share 0-1 words with the corpus, so they still abstain honestly.
            if not passed and lex:
                lscore, ldistinct, lchunk = lex[0]
                has_id = any(ch.isdigit() for ch in query)
                if ldistinct >= 2 or (has_id and lscore >= 3.0):
                    hits = ([(lchunk, 0.5)]
                            + [(c, s) for c, s in hits if c.chunk_id != lchunk.chunk_id])
                    top, passed = 0.5, True

        else:
            hits = self.store.search(query, k=k)
            seen0 = {c.chunk_id for c, _ in hits}
            for c in self._lexical_candidates(query, limit=3):
                if c.chunk_id not in seen0:
                    hits.append((c, _MIN_RELEVANCE))   # lexical hit = relevant
                    seen0.add(c.chunk_id)
            top = hits[0][1] if hits else 0.0
            passed = top >= _MIN_RELEVANCE


        # Unknown-identifier guard: if the query names an ID that exists NOWHERE
        # in the corpus (P-99, FD-9, WO-999…), similar chunks WILL score well —
        # and the LLM would be tempted to answer with the wrong equipment's
        # data. Refuse deterministically instead. Costs nothing, kills the
        # entity-substitution hallucination class outright.
        self._last_unknown = self._unknown_ids(query)
        if passed and self._last_unknown:
            passed, top = False, 0.0

        if self.graph is not None:
            seen = {c.chunk_id for c, _ in hits}
            for cid in self.graph.chunks_near(query):
                if cid not in seen and cid in self._by_id:
                    hits.append((self._by_id[cid], 0.0))  # graph-sourced context
                    seen.add(cid)
        return hits, top, passed

    def ask(self, question: str, k: int = 5, language: str = "English",
            history: list[tuple[str, str]] | None = None) -> Answer:
        # Conversational layer: greetings and "what can you do?" get a friendly
        # reply instead of a document refusal, so the assistant feels human --
        # while genuine out-of-scope facts ("capital of France") still abstain.
        conv = _conversational_reply(question)
        if conv is not None:
            return Answer(text=conv, sources=[], confident=True, score=1.0)

        # For follow-up questions ("what about its temperature?"), include the
        # previous user turn in the retrieval query so pronouns still resolve.
        retrieval_query = question
        if history:
            last_user = next((c for r, c in reversed(history) if r == "user"), "")
            retrieval_query = f"{last_user} {question}".strip()

        # Cross-lingual bridge: documents are English, but the question may be
        # typed/spoken in Kannada, Hindi, Tamil… If the query contains a
        # non-Latin script, translate it to English FOR RETRIEVAL ONLY (the
        # answer still comes back in the user's language).
        if any(ord(ch) > 0x0900 for ch in retrieval_query):
            try:
                en = chat("Translate this question to English. Keep equipment "
                          "tags/IDs unchanged. Output ONLY the translation.",
                          retrieval_query, max_tokens=120).strip()
                if en:
                    retrieval_query = en
            except Exception:
                pass

        # Retrieve once, and abstain on the SAME relevance signal (the reranker
        # score when available) — its clean separation makes "I don't know"
        # reliable even against a large, noisy corpus.
        hits, best_score, passed = self._retrieve_scored(retrieval_query, k=k)

        # Unknown identifier -> specific refusal, and no point spell-fixing.
        if not passed and getattr(self, "_last_unknown", None):
            bad = ", ".join(f"'{t}'" for t in self._last_unknown[:3])
            return Answer(
                text=f"I can't answer this: the identifier {bad} does not "
                     f"appear in any document in the knowledge base. "
                     f"(I won't guess using data from similar-looking equipment.)",
                sources=[], confident=False, score=0.0)

        # Typo rescue: users type "bonifide ceritificate" for "bonafide
        # certificate". Before abstaining, spend ONE cheap LLM call to spell-fix
        # the query and retry — honesty preserved (we still abstain if even the
        # corrected query finds nothing).
        if not passed:
            try:
                fixed = chat(
                    "Correct the spelling and grammar of this search query. "
                    "Keep every ID/code/tag exactly as typed. "
                    "Output ONLY the corrected query, nothing else.",
                    retrieval_query, max_tokens=60).strip().strip('"')
                if fixed and fixed.lower() != retrieval_query.lower():
                    hits2, score2, passed2 = self._retrieve_scored(fixed, k=k)
                    if passed2:
                        hits, best_score, passed = hits2, score2, passed2
            except Exception:
                pass

        if not passed:
            return Answer(
                text="I don't have a documented answer for this. "
                     "No source in the knowledge base covers it.",
                sources=[],
                confident=False,
                score=best_score,
            )

        excerpts = _format_excerpts(hits)

        # Language handling. Put the instruction in BOTH the system prompt and
        # the top of the user message so it can't be lost after a long excerpt
        # block, and give Indic scripts a bigger token budget (Kannada/Tamil/
        # Devanagari cost ~2-3x the tokens of English for the same content).
        english = (language == "English")
        system = _SYSTEM_PROMPT if english else (
            _SYSTEM_PROMPT + f"\n\nIMPORTANT: Write your ENTIRE answer in "
            f"{language}. Keep equipment tags (P-7, FD-2…) and citation markers "
            f"[Source 1] exactly as-is. Do not answer in English.")
        lead = "" if english else f"[Respond ONLY in {language}] "
        max_tokens = 700 if english else 1400

        # Fold recent conversation in so the model handles follow-ups naturally.
        convo = ""
        if history:
            recent = history[-6:]   # keep last few turns
            convo = "Conversation so far:\n" + "\n".join(
                f"{'User' if r == 'user' else 'Assistant'}: {c}" for r, c in recent
            ) + "\n\n"

        user_msg = (
            f"{lead}{convo}"
            f"Question: {question}\n\n"
            f"Document excerpts:\n{excerpts}\n\n"
            f"Answer using only these excerpts, with inline [Source N] citations."
            + ("" if english else f" Write the entire answer in {language}.")
        )

        answer_text = chat(system, user_msg, max_tokens=max_tokens)

        abstained = "don't have a documented answer" in answer_text.lower()
        return Answer(
            text=answer_text,
            sources=[c for c, _ in hits],
            confident=not abstained,
            score=best_score,
        )
