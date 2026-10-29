"""
Live Fact-Checking & Encyclopedic Claim Verification Service
============================================================
Verifies factual claims (sports results, awards, historical events, political
offices, scientific facts, biographical claims, geography) in real time using
Wikipedia's official MediaWiki API (en.wikipedia.org/w/api.php):
  1. Encyclopedic Myth / Pseudoscience / Conspiracy Theory Detection
  2. Award / Championship / Tournament Winner Verification ("X won Y")
  3. Office / Role / Capital / Superlative Verification ("X is the <role> of Y")
  4. Biographical & Scientific Contradiction Analysis (Dead/Alive, East/West, Entity-Fact matching)
"""
import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

_WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
_USER_AGENT = "VeritiScanAI/2.0 (https://github.com/ankanganguli/mini-project; forensic-verifier)"

# Normalize common spelling variations of famous awards, tournaments, and entities
_CLAIM_NORMALIZATIONS = [
    (re.compile(r"\bball[ao]n\s*d'?or\b", re.IGNORECASE), "Ballon d'Or"),
    (re.compile(r"\bmbapp[eé]\b", re.IGNORECASE), "Kylian Mbappé"),
    (re.compile(r"\bucl\b", re.IGNORECASE), "UEFA Champions League"),
]

_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "of", "for", "to", "from", "by", "with",
    "and", "or", "is", "was", "are", "were", "has", "have", "had", "been", "be",
    "won", "win", "wins", "winner", "that", "this", "it", "as", "his", "her", "their"
}

_DEBUNK_MARKERS = re.compile(
    r"\b(scientifically\s+disproven|archaic\s+and\s+scientifically\s+disproven|"
    r"conspiracy\s+theor(?:y|ies)|pseudoscience|pseudoscientific|common\s+misconception|"
    r"urban\s+legend|hoax|debunked|discredited|fraudulent|false\s+claim|"
    r"no\s+scientific\s+evidence|myth|unfounded|fabricated|parody)\b",
    re.IGNORECASE,
)


def normalize_claim_text(text: str) -> str:
    out = text.strip()
    for pattern, repl in _CLAIM_NORMALIZATIONS:
        out = pattern.sub(repl, out)
    return out


def _clean_tokens(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9éáíóúñü]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


class LiveFactCheckService:
    """
    Retrieves live encyclopedic evidence from Wikipedia and evaluates whether
    a factual claim is supported, contradicted (fake), or unverified.
    """

    def is_factual_claim(self, text: str) -> bool:
        words = text.strip().split()
        if len(words) < 3 or len(words) > 100:
            return False
        if len(words) <= 40:
            return True
        return False

    def _wiki_search_and_extracts(self, query: str, limit: int = 4) -> List[Dict[str, str]]:
        """
        Queries Wikipedia search + intro extracts for a query string.
        """
        results: List[Dict[str, str]] = []
        try:
            search_q = " ".join(query.strip().split()[:16])
            params = urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": search_q,
                "srlimit": limit,
                "utf8": 1,
                "format": "json",
            })
            req = urllib.request.Request(
                f"{_WIKI_API_URL}?{params}",
                headers={"User-Agent": _USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))

            search_items = data.get("query", {}).get("search", [])
            if not search_items:
                return []

            titles = [item["title"] for item in search_items[:limit]]
            snippets = {
                item["title"]: re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                for item in search_items[:limit]
            }

            ext_params = urllib.parse.urlencode({
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "exchars": 1100,
                "redirects": 1,
                "titles": "|".join(titles),
                "format": "json",
            })
            ext_req = urllib.request.Request(
                f"{_WIKI_API_URL}?{ext_params}",
                headers={"User-Agent": _USER_AGENT},
            )
            with urllib.request.urlopen(ext_req, timeout=3.5) as resp:
                ext_data = json.loads(resp.read().decode("utf-8", errors="ignore"))

            pages = ext_data.get("query", {}).get("pages", {})
            pages_by_title = {p.get("title", ""): (p.get("extract") or "").strip() for p in pages.values()}

            for t in titles:
                results.append({
                    "title": t,
                    "snippet": snippets.get(t, ""),
                    "extract": pages_by_title.get(t, ""),
                })
        except Exception as e:
            logger.debug(f"Wikipedia API lookup error for '{query}': {e}")
        return results

    def _first_sentence(self, text: str) -> str:
        if not text:
            return ""
        s = re.split(r"(?<=[.!?])\s+", text.strip())
        return s[0][:240] if s else text[:240]

    def verify_claim(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Verifies a factual claim using live Wikipedia evidence and deterministic
        encyclopedic rules.
        """
        if not self.is_factual_claim(text):
            return None

        norm_claim = normalize_claim_text(text)
        lower_claim = norm_claim.lower()
        is_negated = bool(re.search(r"\b(never|not|didn'?t|hasn'?t|haven'?t|isn'?t|wasn'?t|myth|fake|hoax|false)\b", lower_claim))

        # ── 1. Instant High-Precision Checks (Common Factual Traps) ─────────
        # Directional / Astronomical / Basic Scientific facts
        if re.search(r"\bsun\s+(rises|rose)\s+in\s+the\s+(west|north|south)\b", lower_claim) and not is_negated:
            return {
                "is_fake": True,
                "p_fake": 0.96,
                "reason": "Astronomical fact check: The Sun rises in the east and sets in the west due to Earth's rotation.",
            }
        if re.search(r"\bearth\s+is\s+flat\b", lower_claim) and not is_negated:
            return {
                "is_fake": True,
                "p_fake": 0.97,
                "reason": "Scientific fact check: The Earth is an oblate spheroid; Flat Earth is a scientifically disproven misconception.",
            }

        # Ballon d'Or winners check
        if "ballon d'or" in lower_claim and re.search(r"\b(won|winner|wins|awarded)\b", lower_claim) and not is_negated:
            actual_winners = [
                "messi", "ronaldo", "modric", "modrić", "benzema", "rodri",
                "kaka", "kaká", "cannavaro", "ronaldinho", "shevchenko", "nedved",
                "owen", "figo", "rivaldo", "zidane", "cruyff", "platini", "van basten",
                "di stefano", "beckenbauer", "weah", "baggio", "papin",
                "matthaus", "gullit", "belanov", "rossi", "rumenigge", "keegan",
                "simonsen", "blokhin", "muller", "charlton", "best", "eusebio",
                "dembele", "dembélé", "putellas", "bonmati", "bonmatí"
            ]
            if not any(w in lower_claim for w in actual_winners):
                return {
                    "is_fake": True,
                    "p_fake": 0.93,
                    "reason": "Encyclopedic records show this player has never won the Ballon d'Or.",
                }

        # ── 2. Fetch Live Wikipedia Evidence for the Claim ──────────────────
        evidence = self._wiki_search_and_extracts(norm_claim, limit=4)
        if not evidence:
            return None

        combined_wiki = " ".join(
            f"{item['title']} {item['snippet']} {item['extract']}" for item in evidence
        )
        combined_wiki_lower = combined_wiki.lower()

        # ── 3. Myth / Conspiracy / Disproven Claim Check ────────────────────
        top_doc_text = f"{evidence[0]['title']} {evidence[0]['snippet']} {evidence[0]['extract'][:500]}"
        if _DEBUNK_MARKERS.search(top_doc_text) and not is_negated:
            return {
                "is_fake": True,
                "p_fake": 0.94,
                "reason": f"Wikipedia ({evidence[0]['title']}): {self._first_sentence(evidence[0]['extract'] or evidence[0]['snippet'])}",
            }

        # ── 4. Pattern A: "<Subject> won <Event/Award/Tournament>" ──────────
        won_match = re.match(
            r"^([a-zA-Z0-9éáíóúñüÉÁÍÓÚÑÜ\s.'-]+?)\s+(?:has\s+|have\s+)?(?:won|wins|is\s+the\s+winner\s+of|was\s+the\s+winner\s+of|won\s+the)\s+(?:the\s+)?(.+?)[.!?]*$",
            norm_claim.strip(),
            re.IGNORECASE,
        )
        if won_match and not is_negated:
            subj = won_match.group(1).strip()
            obj = won_match.group(2).strip()
            subj_tokens = _clean_tokens(subj)
            obj_tokens = _clean_tokens(obj)

            # Fetch the Wikipedia page specifically for the Event/Award/Tournament (e.g. "2022 FIFA World Cup")
            obj_evidence = self._wiki_search_and_extracts(obj, limit=2)
            subj_evidence = self._wiki_search_and_extracts(subj, limit=2)

            obj_text = " ".join(f"{x['title']} {x['snippet']} {x['extract']}" for x in obj_evidence).lower()
            subj_text = " ".join(f"{x['title']} {x['snippet']} {x['extract']}" for x in subj_evidence).lower()

            # Check if Subject is mentioned in the Event/Award's intro extract OR Event is in Subject's intro
            subj_in_obj = all(t in obj_text for t in subj_tokens) if subj_tokens else False
            obj_in_subj = sum(1 for t in obj_tokens if t in subj_text) >= max(1, int(len(obj_tokens) * 0.6))

            # Check if Subject's snippet/extract explicitly says "never won", "runner-up", "third", "finalist" for this award
            neg_context = re.search(
                r"\b(never\s+won|yet\s+to\s+win|finished\s+third|runner-up|runners-up|eliminated\s+in|lost\s+the\s+final)\b",
                combined_wiki_lower,
            )

            if obj_evidence and not subj_in_obj and not obj_in_subj:
                ref_title = obj_evidence[0]["title"]
                ref_sent = self._first_sentence(obj_evidence[0]["extract"] or obj_evidence[0]["snippet"])
                return {
                    "is_fake": True,
                    "p_fake": 0.90,
                    "reason": f"Contradicts Wikipedia ({ref_title}): '{subj}' is not recorded as the winner. {ref_sent}",
                }
            if neg_context and not subj_in_obj:
                ref_title = evidence[0]["title"]
                return {
                    "is_fake": True,
                    "p_fake": 0.88,
                    "reason": f"Wikipedia ({ref_title}) indicates '{subj}' did not win '{obj}' ({neg_context.group(0)}).",
                }
            if subj_in_obj or obj_in_subj:
                ref_title = (obj_evidence or evidence)[0]["title"]
                return {
                    "is_fake": False,
                    "p_fake": 0.12,
                    "reason": f"Verified by Wikipedia ({ref_title}).",
                }

        # ── 5. Pattern B: "<Subject> is/was the <Role/Capital/Leader> of <Target>" ──
        role_match = re.match(
            r"^([a-zA-Z0-9éáíóúñüÉÁÍÓÚÑÜ\s.'-]+?)\s+(?:is|was|are)\s+(?:the\s+)?(capital|president|prime\s+minister|king|queen|chancellor|ceo|founder|inventor|author|director|largest\s+city|tallest\s+mountain|longest\s+river|richest\s+person)\s+(?:of|in|for)\s+(.+?)[.!?]*$",
            norm_claim.strip(),
            re.IGNORECASE,
        )
        if role_match and not is_negated:
            subj = role_match.group(1).strip()
            role = role_match.group(2).strip()
            target = role_match.group(3).strip()
            subj_tokens = _clean_tokens(subj)

            role_query = f"{role} of {target}"
            role_ev = self._wiki_search_and_extracts(role_query, limit=2)
            subj_ev = self._wiki_search_and_extracts(subj, limit=1)

            role_text = " ".join(f"{x['title']} {x['snippet']} {x['extract']}" for x in (role_ev + evidence[:2])).lower()
            subj_intro = (subj_ev[0]["extract"].lower() if subj_ev else "")

            subj_matches_role = any(t in role_text for t in subj_tokens) if subj_tokens else False
            target_tokens = _clean_tokens(target)
            role_tokens = _clean_tokens(role)
            subj_intro_confirms = (
                all(t in subj_intro for t in role_tokens)
                and any(t in subj_intro for t in target_tokens)
            ) if (subj_intro and role_tokens and target_tokens) else False

            if not subj_matches_role and not subj_intro_confirms:
                ref_doc = (role_ev or evidence)[0]
                return {
                    "is_fake": True,
                    "p_fake": 0.91,
                    "reason": f"Contradicts Wikipedia ({ref_doc['title']}): {self._first_sentence(ref_doc['extract'] or ref_doc['snippet'])}",
                }
            else:
                ref_doc = (role_ev or evidence)[0]
                return {
                    "is_fake": False,
                    "p_fake": 0.10,
                    "reason": f"Confirmed by Wikipedia ({ref_doc['title']}): {self._first_sentence(ref_doc['extract'] or ref_doc['snippet'])}",
                }

        # ── 6. Pattern C: Dead / Alive Hoax Check ───────────────────────────
        if re.search(r"\b(is\s+dead|died|passed\s+away|was\s+killed|assassinated)\b", lower_claim) and not is_negated:
            first_extract = (evidence[0]["extract"] or "").lower()
            if re.search(r"\bis\s+a[n]?\s+", first_extract[:200]) and "died" not in first_extract[:350]:
                return {
                    "is_fake": True,
                    "p_fake": 0.90,
                    "reason": f"Wikipedia ({evidence[0]['title']}) biography uses present tense with no record of death.",
                }

        return None


fact_check_service = LiveFactCheckService()
