"""
Live Fact-Checking & Encyclopedic Claim Verification Service
============================================================
Verifies factual claims (sports results, awards, historical events, political
offices, scientific facts, biographical claims, geography) in real time using
Wikipedia's official MediaWiki API (en.wikipedia.org/w/api.php):
  1. Encyclopedic Myth / Pseudoscience / Conspiracy Theory Detection
  2. Award / Championship / Tournament Winner Verification ("X won Y")
  3. Office / Role / Capital / Superlative Verification ("X is the <role> of Y")
  4. Location / Profession / Founder / Creator Verification ("X is in Y", "X founded Y")
  5. Biographical & Scientific Contradiction Analysis (Dead/Alive, East/West)
"""
import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Dict, Any, List, Optional

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

    def _wiki_search_and_extracts(self, query: str, limit: int = 3) -> List[Dict[str, str]]:
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
                "exchars": 1200,
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

        # ── 2. Pattern A: "<Subject> won <Event/Award/Tournament>" ──────────
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

            obj_evidence = self._wiki_search_and_extracts(obj, limit=2)
            subj_evidence = self._wiki_search_and_extracts(subj, limit=1)

            if obj_evidence:
                # Primary event/award page extract (e.g. "2022 FIFA World Cup")
                primary_obj_doc = obj_evidence[0]
                obj_extract_lower = f"{primary_obj_doc['title']} {primary_obj_doc['extract']}".lower()
                subj_in_obj_extract = all(t in obj_extract_lower for t in subj_tokens) if subj_tokens else False

                # Check if Subject's own intro explicitly has a winning sentence with ALL obj_tokens
                subj_won_sentence = False
                if subj_evidence and obj_tokens:
                    subj_sentences = re.split(r"(?<=[.!?])\s+", subj_evidence[0]["extract"].lower())
                    for sent in subj_sentences:
                        if re.search(r"\b(won|winner|champion|champions|titles?|awarded|recipient)\b", sent):
                            if not re.search(r"\b(never|runner-up|runners-up|third|fourth|lost|eliminated)\b", sent):
                                if all(t in sent for t in obj_tokens):
                                    subj_won_sentence = True
                                    break

                if not subj_in_obj_extract and not subj_won_sentence:
                    ref_title = primary_obj_doc["title"]
                    ref_sent = self._first_sentence(primary_obj_doc["extract"] or primary_obj_doc["snippet"])
                    return {
                        "is_fake": True,
                        "p_fake": 0.91,
                        "reason": f"Contradicts Wikipedia ({ref_title}): '{subj}' is not recorded as the winner of '{obj}'. {ref_sent}",
                    }
                else:
                    ref_title = primary_obj_doc["title"]
                    return {
                        "is_fake": False,
                        "p_fake": 0.12,
                        "reason": f"Verified by Wikipedia ({ref_title}): {self._first_sentence(primary_obj_doc['extract'])}",
                    }

        # ── 3. Pattern B: "<Subject> is/was the <Role/Capital/Leader> of <Target>" ──
        role_match = re.match(
            r"^([a-zA-Z0-9éáíóúñüÉÁÍÓÚÑÜ\s.'-]+?)\s+(?:is|was|are)\s+(?:the\s+)?(capital|capital\s+city|president|prime\s+minister|king|queen|chancellor|ceo|chief\s+executive\s+officer|founder|co-founder|inventor|author|director|largest\s+city|tallest\s+mountain|longest\s+river|richest\s+person)\s+(?:of|in|for)\s+(.+?)[.!?]*$",
            norm_claim.strip(),
            re.IGNORECASE,
        )
        if role_match and not is_negated:
            subj = role_match.group(1).strip()
            role = role_match.group(2).strip().lower()
            target = role_match.group(3).strip()
            subj_tokens = _clean_tokens(subj)
            target_tokens = _clean_tokens(target)

            role_query = f"{role} of {target}"
            role_ev = self._wiki_search_and_extracts(role_query, limit=2)
            subj_ev = self._wiki_search_and_extracts(subj, limit=1)
            if not role_ev and not subj_ev:
                return None

            # Check ONLY the primary Role Wikipedia article extract (never mix with subject search)
            role_primary_extract = (role_ev[0]["extract"].lower() if role_ev else "")
            subj_intro = (subj_ev[0]["extract"].lower() if subj_ev else "")
            subj_first_sent = self._first_sentence(subj_intro)

            if "capital" in role:
                # For capitals, check if Subject's first sentence explicitly says "capital (city) of <target>"
                # and NOT "capital city of the ... state of <other>"
                exact_cap_pattern = re.compile(
                    r"\bcapital(?:\s+city)?\s+of\s+(?:the\s+)?" + r"\s+".join(re.escape(t) for t in target_tokens) + r"\b",
                    re.IGNORECASE,
                )
                is_valid_capital = bool(exact_cap_pattern.search(subj_first_sent))
                if not is_valid_capital:
                    ref_doc = (subj_ev or role_ev)[0] if (subj_ev or role_ev) else {"title": target, "extract": ""}
                    return {
                        "is_fake": True,
                        "p_fake": 0.92,
                        "reason": f"Contradicts Wikipedia ({ref_doc['title']}): {self._first_sentence(ref_doc['extract'])}",
                    }
                else:
                    ref_doc = (subj_ev or role_ev)[0]
                    return {
                        "is_fake": False,
                        "p_fake": 0.10,
                        "reason": f"Confirmed by Wikipedia ({ref_doc['title']}): {self._first_sentence(ref_doc['extract'])}",
                    }

            subj_in_role_page = all(t in role_primary_extract for t in subj_tokens) if (subj_tokens and role_primary_extract) else False
            role_in_subj_intro = (
                role in subj_intro
                and all(t in subj_intro for t in target_tokens)
            ) if (subj_intro and target_tokens) else False

            if not subj_in_role_page and not role_in_subj_intro:
                ref_doc = (subj_ev or role_ev)[0] if (subj_ev or role_ev) else {"title": role_query, "extract": ""}
                return {
                    "is_fake": True,
                    "p_fake": 0.92,
                    "reason": f"Contradicts Wikipedia ({ref_doc['title']}): {self._first_sentence(ref_doc['extract'])}",
                }
            else:
                ref_doc = (role_ev or subj_ev)[0]
                return {
                    "is_fake": False,
                    "p_fake": 0.10,
                    "reason": f"Confirmed by Wikipedia ({ref_doc['title']}): {self._first_sentence(ref_doc['extract'])}",
                }

        # ── 4. Pattern C: "<Subject> founded / invented / discovered / located in <Object>" ──
        rel_match = re.match(
            r"^([a-zA-Z0-9éáíóúñüÉÁÍÓÚÑÜ\s.'-]+?)\s+(?:was\s+|is\s+)?(founded|invented|discovered|created|built\s+in|located\s+in|situated\s+in|plays\s+for)\s+(.+?)[.!?]*$",
            norm_claim.strip(),
            re.IGNORECASE,
        )
        if rel_match and not is_negated:
            subj = rel_match.group(1).strip()
            verb = rel_match.group(2).strip().lower()
            obj = rel_match.group(3).strip()
            subj_tokens = _clean_tokens(subj)
            obj_tokens = _clean_tokens(obj)

            subj_ev = self._wiki_search_and_extracts(subj, limit=1)
            obj_ev = self._wiki_search_and_extracts(obj, limit=1)

            subj_ext = (subj_ev[0]["extract"].lower() if subj_ev else "")
            obj_ext = (obj_ev[0]["extract"].lower() if obj_ev else "")

            obj_in_subj = all(t in subj_ext for t in obj_tokens) if (obj_tokens and subj_ext) else False
            subj_in_obj = all(t in obj_ext for t in subj_tokens) if (subj_tokens and obj_ext) else False

            if (subj_ev or obj_ev) and not obj_in_subj and not subj_in_obj:
                ref_doc = (subj_ev or obj_ev)[0]
                return {
                    "is_fake": True,
                    "p_fake": 0.90,
                    "reason": f"Contradicts Wikipedia ({ref_doc['title']}): {self._first_sentence(ref_doc['extract'])}",
                }

        # ── 5. Fetch General Evidence & Check Myth / Dead-Alive Hoaxes ──────
        evidence = self._wiki_search_and_extracts(norm_claim, limit=3)
        if not evidence:
            return None

        top_doc_text = f"{evidence[0]['title']} {evidence[0]['snippet']} {evidence[0]['extract'][:500]}"
        if _DEBUNK_MARKERS.search(top_doc_text) and not is_negated:
            return {
                "is_fake": True,
                "p_fake": 0.94,
                "reason": f"Wikipedia ({evidence[0]['title']}): {self._first_sentence(evidence[0]['extract'] or evidence[0]['snippet'])}",
            }

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
