"""Domain-aware natural language query parser with optional Gemini/OpenAI LLM enrichment."""

from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse

import httpx

from bookrecsys.catalog.text_utils import clean_text
from bookrecsys.models import ParsedQuery

# Known metadata vs reading platforms for smart default role assignment
METADATA_DOMAINS = {
    "novelupdates.com",
    "goodreads.com",
    "anilist.co",
    "myanimelist.net",
    "ranobedb.org",
}
READING_DOMAINS = {
    "novelphoenix.com",
    "wuxiaworld.com",
    "royalroad.com",
    "webnovel.com",
    "scribblehub.com",
    "openlibrary.org",
    "gutenberg.org",
}

# Canonical genre synonyms -> (canonical_genre, optional_soft_genres)
GENRE_PATTERNS: list[tuple[re.Pattern[str], str, list[str]]] = [
    (re.compile(r"\bxianxia\b", re.I), "xianxia", ["xuanhuan", "martial arts"]),
    (re.compile(r"\bxuanhuan\b", re.I), "xuanhuan", ["xianxia", "fantasy"]),
    (re.compile(r"\bwuxia\b", re.I), "wuxia", ["martial arts", "xianxia"]),
    (re.compile(r"\bmartial\s+arts?\b", re.I), "martial arts", ["wuxia", "xianxia"]),
    (re.compile(r"\bsci[\-\s]?fi\b|\bscience\s+fiction\b", re.I), "sci-fi", ["science fiction"]),
    (re.compile(r"\bfantasy\b", re.I), "fantasy", []),
    (re.compile(r"\bromance\b|\bromantic\b", re.I), "romance", []),
    (re.compile(r"\bmystery\b|\bdetective\b", re.I), "mystery", ["thriller"]),
    (re.compile(r"\bthriller\b|\bsuspense\b", re.I), "thriller", ["mystery"]),
    (re.compile(r"\bhorror\b|\bsupernatural\s+horror\b", re.I), "horror", ["supernatural", "mystery"]),
    (re.compile(r"\bpsychological\b", re.I), "psychological", []),
    (re.compile(r"\bhistorical\b", re.I), "historical", []),
    (re.compile(r"\bcomedy\b|\bhumou?r(?:ous)?\b|\bfunny\b", re.I), "comedy", []),
    (re.compile(r"\btragedy\b|\btragic\b", re.I), "tragedy", []),
    (re.compile(r"\bslice\s+of\s+life\b", re.I), "slice of life", []),
    (re.compile(r"\bmecha\b", re.I), "mecha", ["sci-fi"]),
    (re.compile(r"\bsmut\b", re.I), "smut", []),
    (re.compile(r"\bharem\b", re.I), "harem", []),
    (re.compile(r"\byaoi\b|\bdanmei\b|\boys['\s]*love\b", re.I), "yaoi", ["shounen ai"]),
    (re.compile(r"\byuri\b|\bgirls['\s]*love\b", re.I), "yuri", ["shoujo ai"]),
    (re.compile(r"\bphilosophy\b|\bphilosophical\b", re.I), "philosophy", ["nonfiction", "classics"]),
    (re.compile(r"\bclassic(?:s)?\b", re.I), "classics", ["fiction"]),
    (re.compile(r"\byoung\s+adult\b|\bya\b", re.I), "young adult", []),
]

# Canonical trope/tag patterns -> (required_tag, soft_related_tags)
TAG_PATTERNS: list[tuple[re.Pattern[str], str, list[str]]] = [
    (
        re.compile(r"\btransmigrat(?:ed|ion|or|ing|e)\b|\bcrossed\s+over\b|\bsoul\s+transfer\b", re.I),
        "transmigration",
        ["transported to another world", "reincarnation"],
    ),
    (
        re.compile(r"\bisekai\b|\btransported\s+to\s+another\s+world\b|\banother\s+world\b", re.I),
        "transported to another world",
        ["transmigration", "reincarnation"],
    ),
    (
        re.compile(r"\breincarnat(?:ed|ion|or)\b|\brebirth\b|\breborn\b", re.I),
        "reincarnation",
        ["second chance", "transmigration"],
    ),
    (
        re.compile(r"\bregress(?:or|ed|ion)\b|\bsecond\s+chance\b|\bback\s+in\s+time\b", re.I),
        "second chance",
        ["time travel", "reincarnation"],
    ),
    (re.compile(r"\btime\s+loop\b", re.I), "time loop", ["time travel"]),
    (re.compile(r"\bcultivat(?:ion|or|ing)\b|\bimmortal(?:s|ity)?\b", re.I), "cultivation", ["immortals"]),
    (re.compile(r"\bsystem\b|\bstatus\s+window\b|\bcheat\s+system\b", re.I), "system", ["game elements"]),
    (re.compile(r"\blitrpg\b|\bgame\s+elements\b|\blevel(?:ing)?\s+up\b", re.I), "game elements", ["litrpg"]),
    (re.compile(r"\bdungeon(?:s)?\b", re.I), "dungeons", ["monsters"]),
    (re.compile(r"\bnecromancer\b|\bnecromancy\b", re.I), "necromancer", ["magic"]),
    (re.compile(r"\bruthless\b", re.I), "ruthless protagonist", ["cunning protagonist"]),
    (re.compile(r"\bcautious\b|\bsteady\b|\blow[\-\s]key\b", re.I), "cautious protagonist", ["calm protagonist"]),
    (re.compile(r"\bscheming\b|\bcunning\b|\bmanipulative\b", re.I), "scheming", ["cunning protagonist"]),
    (re.compile(r"\bsmart\s+(?:mc|protagonist|main\s+character)\b|\bintelligent\b", re.I), "clever protagonist", []),
    (re.compile(r"\bevil\s+(?:mc|protagonist|main\s+character)\b|\bvillain(?:ous)?\s+(?:mc|protagonist|lead)\b", re.I), "evil protagonist", ["antihero protagonist"]),
    (re.compile(r"\banti[\-\s]?hero\b", re.I), "antihero protagonist", []),
    (re.compile(r"\boverpowered\b|\bop\s+(?:mc|protagonist|lead)\b", re.I), "overpowered protagonist", ["strong lead"]),
    (re.compile(r"\bweak\s+to\s+strong\b", re.I), "weak to strong", ["character growth"]),
    (re.compile(r"\bkingdom[\-\s]building\b|\bterritory\s+management\b|\bfaction\s+building\b", re.I), "kingdom building", ["management"]),
    (re.compile(r"\bacademy\b|\bschool\s+life\b", re.I), "academy", []),
    (re.compile(r"\bapocalyp(?:se|tic)\b|\bpost[\-\s]apocalyp(?:se|tic)\b", re.I), "apocalypse", ["post-apocalyptic", "survival"]),
    (re.compile(r"\bsteampunk\b", re.I), "steampunk", []),
    (re.compile(r"\blovecraft(?:ian)?\b|\beldritch\b|\bcosmic\s+horror\b", re.I), "cosmic horror", ["lovecraftian", "cults"]),
    (re.compile(r"\bdystopia(?:n)?\b", re.I), "dystopia", ["science fiction"]),
    (re.compile(r"\bspy\b|\bespionage\b|\bsecret\s+agent\b", re.I), "espionage", ["thriller", "action"]),
    (re.compile(r"\bfemale\s+(?:mc|protagonist|lead|main\s+character)\b|\bfmc\b", re.I), "female protagonist", ["female lead"]),
    (re.compile(r"\bmale\s+(?:mc|protagonist|lead|main\s+character)\b|\bmmc\b", re.I), "male protagonist", ["male lead"]),
    (re.compile(r"\balchemy\b|\balchemist\b", re.I), "alchemy", ["pills"]),
    (re.compile(r"\bsword(?:sman|smanship)?\b", re.I), "sword", ["swordsmanship", "sword and magic"]),
]

_URL_RE = re.compile(r"https?://[^\s,)>\"']+", re.I)
_BARE_DOMAIN_RE = re.compile(
    r"\b([a-z0-9\-]+\.(?:com|org|net|co|io|me))\b",
    re.I,
)
_NEGATION_CLAUSE_RE = re.compile(
    r"\b(?:no|without|not\s+a|not|exclude|excluding|avoid|zero)\s+([a-z0-9\-\s]{3,35}?)(?=[,.;]|\band\b|\bor\b|\bwith\b|\bthat\b|$)",
    re.I,
)
_MIN_CH_RE = re.compile(
    r"(?:at\s+least|minimum|min|over|more\s+than|>\s*|>=?\s*)\s*(\d{2,5})\s*(?:chapters?|ch\b)",
    re.I,
)
_MAX_CH_RE = re.compile(
    r"(?:under|less\s+than|fewer\s+than|max|maximum|<\s*|<=?\s*)\s*(\d{2,5})\s*(?:chapters?|ch\b)",
    re.I,
)
_SIMILAR_TO_RE = re.compile(
    r"(?:similar\s+to|books?\s+like|novels?\s+like|such\s+as|fans\s+of)\s+[\"']?([A-Z][A-Za-z0-9'’:\-\s]{2,55})(?=[\"']?(?:[,.;]|\band\b|\bwith\b|\bbut\b|$))"
)


def _normalize_domain(raw_url_or_domain: str) -> str:
    s = raw_url_or_domain.strip().rstrip(".,;:!?)")
    if "://" in s:
        host = urlparse(s).netloc.lower()
    else:
        host = s.split("/")[0].lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _extract_domains_and_roles(text: str) -> tuple[list[str], list[str]]:
    """
    Identify domains mentioned in natural language and classify each into:
    - required_metadata_domains (e.g. 'have an entry on https://www.novelupdates.com')
    - preferred_reading_domains (e.g. 'I want to read the novel on https://novelphoenix.com')
    """
    meta_domains: list[str] = []
    read_domains: list[str] = []

    # Split sentences only on whitespace-bounded punctuation or newlines (never dots inside URLs!)
    sentences = re.split(r"(?:[.!?]+(?=\s|$)|\n+)", text)
    for sent in sentences:
        sent_low = sent.lower()
        found: list[str] = []
        # Mask out full URLs before scanning bare domains so 'https://www.novelupdates.com' never yields 'www'
        for m in _URL_RE.finditer(sent):
            d = _normalize_domain(m.group(0))
            if d and "." in d and d not in found:
                found.append(d)
        sent_without_urls = _URL_RE.sub(" ", sent)
        for m in _BARE_DOMAIN_RE.finditer(sent_without_urls):
            d = _normalize_domain(m.group(1))
            if d and "." in d and d not in found:
                found.append(d)

        for dom in found:
            is_read_context = bool(
                re.search(
                    rf"(?:read(?:ing)?|available|hosted|chapters?)\b[^.]*?\b{re.escape(dom.split('.')[0])}",
                    sent_low,
                )
            )
            is_meta_context = bool(
                re.search(
                    rf"(?:entry|listed|page|profile|database|catalog|on\s+novelupdates)\b[^.]*?\b{re.escape(dom.split('.')[0])}",
                    sent_low,
                )
            )

            if is_read_context or (dom in READING_DOMAINS and not is_meta_context):
                if dom not in read_domains:
                    read_domains.append(dom)
            elif is_meta_context or dom in METADATA_DOMAINS:
                if dom not in meta_domains:
                    meta_domains.append(dom)
            else:
                # Default unknown website mentioned with 'read' anywhere -> read_domains, else both/read
                if "read" in sent_low:
                    if dom not in read_domains:
                        read_domains.append(dom)
                else:
                    if dom not in meta_domains:
                        meta_domains.append(dom)

    # Also handle shorthand names without TLD (e.g. "on novelupdates" / "on novelphoenix")
    low_all = text.lower()
    if "novelupdates" in low_all or "novel updates" in low_all:
        if "novelupdates.com" not in meta_domains:
            meta_domains.append("novelupdates.com")
    if "novelphoenix" in low_all or "novel phoenix" in low_all:
        if "novelphoenix.com" not in read_domains:
            read_domains.append("novelphoenix.com")
    if "royalroad" in low_all or "royal road" in low_all:
        if "royalroad.com" not in read_domains and "royalroad.com" not in meta_domains:
            read_domains.append("royalroad.com")

    return meta_domains, read_domains


def parse_natural_language_query(query: str, allow_llm: bool = True) -> ParsedQuery:
    """
    Parse a natural language recommendation prompt into structured filters, soft boosts,
    target domains, and a clean semantic query string.
    """
    raw = clean_text(query)
    if not raw:
        return ParsedQuery()

    parsed = ParsedQuery(raw_query=raw)
    parsed.required_metadata_domains, parsed.preferred_reading_domains = _extract_domains_and_roles(raw)

    # 1. Extract negated clauses FIRST so "no harem" or "not ongoing" never triggers positive match!
    negated_spans: list[tuple[int, int]] = []
    for m in _NEGATION_CLAUSE_RE.finditer(raw):
        phrase = m.group(1).strip().lower()
        negated_spans.append(m.span())
        if "ongoing" in phrase or "incomplete" in phrase or "hiatus" in phrase:
            parsed.completion_status = "completed"
        if "completed" in phrase or "finished" in phrase:
            parsed.completion_status = "ongoing"

        for pat, g_canon, _ in GENRE_PATTERNS:
            if pat.search(phrase) and g_canon not in parsed.excluded_genres:
                parsed.excluded_genres.append(g_canon)
                # 'harem' / 'smut' / 'tragedy' can appear in tags too
                if g_canon not in parsed.excluded_tags:
                    parsed.excluded_tags.append(g_canon)
        for pat, t_canon, _ in TAG_PATTERNS:
            if pat.search(phrase) and t_canon not in parsed.excluded_tags:
                parsed.excluded_tags.append(t_canon)

    # Mask out negated spans & URLs for positive extraction
    masked_chars = list(raw)
    for s, e in negated_spans:
        for i in range(s, e):
            masked_chars[i] = " "
    positive_text = "".join(masked_chars)
    positive_no_urls = _URL_RE.sub(" ", positive_text)

    # 2. Media format detection
    low = positive_no_urls.lower()
    if re.search(r"\bjapanese\s+light\s+novels?\s+only\b|\bstrictly\s+light\s+novels?\b", low):
        parsed.media_filter = "light_novel_strict"
    elif re.search(
        r"\b(?:light\s+novels?|web\s*novels?|wn\b|ln\b|xianxia|xuanhuan|wuxia|litrpg|cultivation)\b",
        low,
    ) or "novelupdates.com" in parsed.required_metadata_domains or "novelphoenix.com" in parsed.preferred_reading_domains:
        parsed.media_filter = "asian_novel"
    elif re.search(
        r"\b(?:physical|print|paperbacks?|hardcovers?|published\s+books?)\b", low
    ):
        parsed.media_filter = "print_book"

    # 3. Language origin detection
    if re.search(r"\bchinese\b", low):
        parsed.original_language = "chinese"
    elif re.search(r"\bjapanese\b", low):
        parsed.original_language = "japanese"
    elif re.search(r"\bkorean\b", low):
        parsed.original_language = "korean"
    elif re.search(r"\benglish\s+original\b", low):
        parsed.original_language = "english"

    # 4. Completion status
    if re.search(r"\b(?:fully|completely)\s+translated\b", low):
        parsed.completion_status = "completed"
        parsed.require_complete_translation = True
    elif re.search(r"\b(?:completed|complete|finished)\b", low):
        parsed.completion_status = "completed"
    elif re.search(r"\b(?:ongoing|currently\s+releasing|airing)\b", low) and not parsed.completion_status:
        parsed.completion_status = "ongoing"

    # 5. Chapter bounds
    m_min = _MIN_CH_RE.search(positive_no_urls)
    if m_min:
        parsed.min_chapters = int(m_min.group(1))
    m_max = _MAX_CH_RE.search(positive_no_urls)
    if m_max:
        parsed.max_chapters = int(m_max.group(1))

    # 6. Genres & Tags
    for pat, g_canon, soft_gs in GENRE_PATTERNS:
        if pat.search(positive_no_urls) and g_canon not in parsed.excluded_genres:
            if g_canon not in parsed.required_genres:
                parsed.required_genres.append(g_canon)
            for sg in soft_gs:
                if sg not in parsed.soft_genres and sg not in parsed.excluded_genres:
                    parsed.soft_genres.append(sg)

    for pat, t_canon, soft_ts in TAG_PATTERNS:
        if pat.search(positive_no_urls) and t_canon not in parsed.excluded_tags:
            if t_canon not in parsed.required_tags:
                parsed.required_tags.append(t_canon)
            for st in soft_ts:
                if st not in parsed.soft_tags and st not in parsed.excluded_tags:
                    parsed.soft_tags.append(st)

    # 7. Seed titles ("similar to X" / "books like X")
    for m in _SIMILAR_TO_RE.finditer(raw):
        cand = clean_text(m.group(1))
        if cand and cand.lower() not in {"my history", "my past reads", "my physical books"}:
            parsed.similar_to_titles.append(cand)

    # 8. Clean semantic query (strip URLs, domain boilerplate, meta instructions)
    sem = _URL_RE.sub(" ", raw)
    sem = _BARE_DOMAIN_RE.sub(" ", sem)
    sem = re.sub(
        r"\b(?:find\s+me|give\s+me|recommend(?:\s+me)?|i\s+want\s+to\s+read(?:\s+the\s+novel)?\s+on|"
        r"also\s+have\s+an\s+entry\s+on|have\s+an\s+entry\s+on|the\s+novel\s+should\s+be|etc\.?)\b",
        " ",
        sem,
        flags=re.I,
    )
    sem = clean_text(sem)
    parsed.semantic_query = sem or raw

    # 9. Optional LLM refinement when GEMINI_API_KEY or OPENAI_API_KEY is configured
    if allow_llm:
        _maybe_enrich_with_llm(parsed)

    return parsed


def _maybe_enrich_with_llm(parsed: ParsedQuery) -> None:
    """Optionally enrich subtle query nuances via Gemini or OpenAI API if configured in .env."""
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not gemini_key and not openai_key:
        return

    prompt = (
        "Extract structured book/novel search parameters from the user's request as JSON with keys: "
        "semantic_query (string), extra_genres (list of lowercase strings), "
        "extra_tags (list of lowercase strings), excluded_tags (list of lowercase strings). "
        f"User request: {parsed.raw_query!r}"
    )
    try:
        with httpx.Client(timeout=6.0) as client:
            if gemini_key:
                resp = client.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                    params={"key": gemini_key},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"responseMimeType": "application/json"},
                    },
                )
                resp.raise_for_status()
                txt = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(txt)
                parsed.parser_backend = "rule+gemini"
            else:
                resp = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {openai_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "response_format": {"type": "json_object"},
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                resp.raise_for_status()
                txt = resp.json()["choices"][0]["message"]["content"]
                data = json.loads(txt)
                parsed.parser_backend = "rule+openai"

        for tg in data.get("extra_tags") or []:
            tg_c = clean_text(tg).lower()
            if tg_c and tg_c not in parsed.soft_tags and tg_c not in parsed.excluded_tags:
                parsed.soft_tags.append(tg_c)
        for ex in data.get("excluded_tags") or []:
            ex_c = clean_text(ex).lower()
            if ex_c and ex_c not in parsed.excluded_tags:
                parsed.excluded_tags.append(ex_c)
    except Exception:
        # Gracefully keep deterministic parse on any network/key failure
        pass
