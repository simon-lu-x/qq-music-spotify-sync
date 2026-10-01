"""Match QQ Music songs to Spotify tracks using two-phase search + scoring."""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from opencc import OpenCC

from .qq_music import QQSong
from .spotify_client import SpotifyClient, SpotifyTrack

logger = logging.getLogger(__name__)

# Tags that indicate a special version we want to exclude unless the source also has them
_SPECIAL_VERSION_TAGS = re.compile(
    r"\b(live|remaster(?:ed)?|karaoke|instrumental|dj\s*version|remix|cover|acoustic|demo|acapella)\b|伴奏|现场版|現場版|韩文版|韓文版|日文版|英文版|粤语版|粵語版",
    re.IGNORECASE,
)

# Patterns to strip from titles during normalization
_BRACKET_CONTENT = re.compile(r"[(\[（【《「『][^)\]）】》」』]*[)\]）】》」』]")
_FEAT_SUFFIX = re.compile(r"\s+(feat\.?|ft\.?|featuring)\s+.*", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_LATIN_ARTIST_PHRASE = re.compile(r"[a-z0-9][a-z0-9 .&'/-]*[a-z0-9]", re.IGNORECASE)
_CJK_ARTIST_PHRASE = re.compile(r"[\u4e00-\u9fff]{2,}")
_VERSION_SUFFIX = re.compile(
    r"\s*[-–—:：]\s*(live|remix|acoustic|demo|acapella|伴奏|说唱版|說唱版|现场版|現場版|live版伴奏|韩文版|韓文版|日文版|英文版|粤语版|粵語版)\b.*$",
    re.IGNORECASE,
)
_META_SUFFIX = re.compile(
    r'\s*[-–—:：]\s*(第\d+波|from the first take|電影.*主題曲|电影.*主题曲|影视剧.*片尾曲|影視劇.*片尾曲|影视剧.*插曲|影視劇.*插曲|ost)\b.*$',
    re.IGNORECASE,
)
_PROMO_SUFFIX = re.compile(r"\s*第\d+波$", re.IGNORECASE)
_TRAILING_DESCRIPTOR = re.compile(r"\s+(说唱版|說唱版)$", re.IGNORECASE)
_SOUNDTRACK_METADATA = re.compile(
    r"\s*(网络剧|網絡劇|电视剧|電視劇|电影|電影)?\s*(主题曲|主題曲|片尾曲|插曲|等待曲)\s*",
    re.IGNORECASE,
)
_PUNCT_TO_SPACE = str.maketrans({
    ".": " ",
    "·": " ",
    "•": " ",
    "・": " ",
    "_": " ",
    "-": " ",
    "/": " ",
    "&": " ",
})

_MAX_DURATION_DIFF_MS = 15_000   # 15 seconds
_TITLE_SIMILARITY_THRESHOLD = 0.8
_RELAXED_PRIMARY_TITLE_THRESHOLD = 0.95
_SEARCH_PACING = 0.1             # seconds between API calls
_TOTAL_BUDGET_SECONDS = 300      # 5-minute total budget for all searches
_MAX_RETRIES_PER_SONG = 2

_T2S = OpenCC("t2s")


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _contains_latin(text: str) -> bool:
    return bool(re.search(r"[a-z]", text, re.IGNORECASE))


@dataclass
class UnmatchedReason:
    query: str
    candidates: list[dict]       # Raw candidate info for artifact logging
    reason: str                  # Human-readable explanation


@dataclass
class MatchResult:
    matched: list[tuple[QQSong, SpotifyTrack]] = field(default_factory=list)
    unmatched: list[tuple[QQSong, UnmatchedReason]] = field(default_factory=list)
    timed_out: bool = False      # True if budget was exhausted mid-run


def _normalize(text: str) -> str:
    """Normalize a track/artist name for fuzzy comparison."""
    # Convert full-width characters to ASCII equivalents
    text = unicodedata.normalize("NFKC", text)
    text = _T2S.convert(text)
    text = _VERSION_SUFFIX.sub("", text)
    text = _META_SUFFIX.sub("", text)
    text = _PROMO_SUFFIX.sub("", text)
    # Remove content inside brackets
    text = _BRACKET_CONTENT.sub("", text)
    # Remove feat. / ft. suffixes
    text = _FEAT_SUFFIX.sub("", text)
    text = _TRAILING_DESCRIPTOR.sub("", text)
    text = _SOUNDTRACK_METADATA.sub(" ", text)
    text = re.sub(r"[()（）]", " ", text)
    # Normalize punctuation variants that often differ across providers
    text = text.translate(_PUNCT_TO_SPACE)
    # Lowercase and collapse whitespace
    text = _WHITESPACE.sub(" ", text.lower()).strip()
    return text


_TRAILING_LATIN_SUBTITLE = re.compile(r"(?<=[\u4e00-\u9fff])\s*[a-z0-9][a-z0-9 ]*$")


def _is_latin_subtitle_variant(a: str, b: str) -> bool:
    """
    True when two normalized titles differ only by an English subtitle glued onto
    the Chinese title, e.g. "甲乙丙丁" vs "甲乙丙丁strangers". Spotify often
    lists Chinese songs this way while QQ Music uses the Chinese title alone.
    """
    for short, long_ in ((a, b), (b, a)):
        core = _TRAILING_LATIN_SUBTITLE.sub("", long_).strip()
        if core != long_ and core == short and _contains_cjk(core) and len(core) >= 2:
            return True
    return False


def _title_similarity(a: str, b: str) -> float:
    norm_a, norm_b = _normalize(a), _normalize(b)
    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    if _is_latin_subtitle_variant(norm_a, norm_b):
        return max(ratio, _RELAXED_PRIMARY_TITLE_THRESHOLD)
    return ratio


def _artists_overlap(qq_artists: list[str], sp_artists: list[str]) -> int:
    """Return count of matching artists (case-insensitive, whitespace-collapsed)."""
    norm_qq = set().union(*(_artist_aliases(a) for a in qq_artists)) if qq_artists else set()
    norm_sp = set().union(*(_artist_aliases(a) for a in sp_artists)) if sp_artists else set()
    return len(norm_qq & norm_sp)


def _artist_aliases(name: str) -> set[str]:
    """
    Expand an artist string into a small alias set.

    Examples:
      - "Dizzy Dizzo (蔡诗芸)" -> {"dizzy dizzo", "蔡诗芸", ...}
      - "G.E.M. 邓紫棋" -> {"g e m", "邓紫棋", ...}
    """
    aliases: set[str] = set()
    normalized_full = _normalize(name)
    if normalized_full:
        aliases.add(normalized_full)

    full_text = unicodedata.normalize("NFKC", _T2S.convert(name))
    bracket_chunks = re.findall(r"[(\[（【]([^)\]）】]+)[)\]）】]", full_text)
    for chunk in bracket_chunks:
        normalized_chunk = _normalize(chunk)
        if normalized_chunk:
            aliases.add(normalized_chunk)

    outside_brackets = _BRACKET_CONTENT.sub(" ", full_text)
    normalized_outside = _normalize(outside_brackets)
    if normalized_outside:
        aliases.add(normalized_outside)

    for match in _LATIN_ARTIST_PHRASE.findall(full_text):
        normalized_match = _normalize(match)
        if normalized_match:
            aliases.add(normalized_match)

    for match in _CJK_ARTIST_PHRASE.findall(full_text):
        normalized_match = _normalize(match)
        if normalized_match:
            aliases.add(normalized_match)

    return aliases


def _has_special_version_tag(title: str) -> bool:
    return bool(_SPECIAL_VERSION_TAGS.search(title))


def _duration_score(qq_ms: int, sp_ms: int) -> float:
    """Return a 0..1 score; 1 = perfect match, 0 = >= MAX_DURATION_DIFF_MS apart."""
    if qq_ms == 0 or sp_ms == 0:
        return 0.5  # Unknown: neutral score
    diff = abs(qq_ms - sp_ms)
    if diff >= _MAX_DURATION_DIFF_MS:
        return 0.0
    return 1.0 - diff / _MAX_DURATION_DIFF_MS


def _score_candidate(
    song: QQSong,
    candidate: SpotifyTrack,
    source_has_special: bool,
    *,
    allow_primary_artist_fallback: bool = False,
) -> float | None:
    """
    Return a composite score [0..1] for the candidate, or None if it fails
    the acceptance criteria.
    """
    # Acceptance: title similarity
    title_sim = _title_similarity(song.title, candidate.name)
    if title_sim < _TITLE_SIMILARITY_THRESHOLD:
        return None

    # Acceptance: artist overlap
    overlap = _artists_overlap(song.artists, candidate.artists)
    if overlap == 0 and not (
        allow_primary_artist_fallback and _is_strong_primary_match(song, candidate)
    ):
        return None

    candidate_has_special = _has_special_version_tag(candidate.name)
    # Acceptance: only match special/non-special versions consistently.
    if not source_has_special and candidate_has_special:
        return None
    if source_has_special and not candidate_has_special:
        return None

    # Acceptance: duration check (only when we have QQ duration)
    if song.duration_ms > 0:
        diff = abs(song.duration_ms - candidate.duration_ms)
        if diff > _MAX_DURATION_DIFF_MS:
            return None
        dur_score = _duration_score(song.duration_ms, candidate.duration_ms)
    else:
        dur_score = 0.5

    # Composite score
    if overlap == 0:
        normalized_overlap = 0.35
        score = title_sim * 0.65 + normalized_overlap * 0.1 + dur_score * 0.25
    else:
        max_overlap = max(len(song.artists), len(candidate.artists), 1)
        normalized_overlap = min(overlap / max_overlap, 1.0)
        score = title_sim * 0.4 + normalized_overlap * 0.3 + dur_score * 0.3
    return score


def _best_candidate(
    song: QQSong,
    candidates: list[SpotifyTrack],
    *,
    allow_primary_artist_fallback: bool = False,
) -> SpotifyTrack | None:
    source_has_special = _has_special_version_tag(song.title)
    scored = []
    for c in candidates:
        s = _score_candidate(
            song,
            c,
            source_has_special,
            allow_primary_artist_fallback=allow_primary_artist_fallback,
        )
        if s is not None:
            scored.append((s, c))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _is_strong_primary_match(song: QQSong, candidate: SpotifyTrack) -> bool:
    """
    Allow a cautious fallback when Spotify's top result clearly matches the title,
    but artist names differ because one side uses English stage names / aliases.
    Only used for the primary query that already contains artist text.
    """
    title_sim = _title_similarity(song.title, candidate.name)
    if title_sim < _RELAXED_PRIMARY_TITLE_THRESHOLD:
        return False

    # Only use this fallback for cross-script artist names such as:
    #   周杰伦 -> Jay Chou, 李荣浩 -> Ronghao Li
    source_has_cjk = any(_contains_cjk(artist) for artist in song.artists)
    candidate_has_latin = any(_contains_latin(artist) for artist in candidate.artists)
    candidate_has_cjk = any(_contains_cjk(artist) for artist in candidate.artists)
    if not source_has_cjk or not candidate_has_latin or candidate_has_cjk:
        return False

    if _has_special_version_tag(candidate.name) and not _has_special_version_tag(song.title):
        return False

    if song.duration_ms > 0 and abs(song.duration_ms - candidate.duration_ms) > _MAX_DURATION_DIFF_MS:
        return False

    return True


def _candidates_for_log(candidates: list[SpotifyTrack]) -> list[dict]:
    return [
        {"uri": c.uri, "name": c.name, "artists": c.artists, "duration_ms": c.duration_ms}
        for c in candidates
    ]


def match_songs(songs: list[QQSong], spotify: SpotifyClient) -> MatchResult:
    """
    For each QQ song, run two-phase Spotify search and pick the best match.
    Respects a 5-minute total budget.
    """
    result = MatchResult()
    deadline = time.monotonic() + _TOTAL_BUDGET_SECONDS

    for i, song in enumerate(songs):
        if time.monotonic() > deadline:
            logger.warning(
                "Search budget exhausted after %d/%d songs. Skipping remainder.",
                i,
                len(songs),
            )
            result.timed_out = True
            # Remaining songs are unmatched due to timeout
            for remaining in songs[i:]:
                result.unmatched.append(
                    (remaining, UnmatchedReason(query="", candidates=[], reason="budget_timeout"))
                )
            break

        primary_query = f"{song.title} {song.artists[0]}" if song.artists else song.title
        fallback_query = song.title

        track, reason = _search_with_retry(song, primary_query, fallback_query, spotify)

        if track:
            result.matched.append((song, track))
        else:
            result.unmatched.append((song, reason))

        # Pacing between songs
        time.sleep(_SEARCH_PACING)

    return result


def _search_with_retry(
    song: QQSong,
    primary_query: str,
    fallback_query: str,
    spotify: SpotifyClient,
) -> tuple[SpotifyTrack | None, UnmatchedReason]:
    """Run primary and fallback searches, return the best match or an unmatched reason."""
    all_candidates: list[SpotifyTrack] = []

    for query in [primary_query, fallback_query]:
        for attempt in range(_MAX_RETRIES_PER_SONG):
            try:
                candidates = spotify.search_tracks(query)
                break
            except Exception as exc:
                if attempt < _MAX_RETRIES_PER_SONG - 1:
                    logger.warning(
                        "Search attempt %d failed for '%s': %s. Retrying...",
                        attempt + 1, query, exc,
                    )
                    time.sleep(1.5 ** attempt)
                else:
                    logger.error("Search failed for '%s' after %d attempts: %s", query, _MAX_RETRIES_PER_SONG, exc)
                    candidates = []

        all_candidates.extend(candidates)
        best = _best_candidate(
            song,
            candidates,
            allow_primary_artist_fallback=(query == primary_query),
        )
        if best:
            logger.debug("Matched '%s' via query '%s' -> '%s'", song.title, query, best.name)
            return best, None  # type: ignore[return-value]

    # Both phases failed
    reason = UnmatchedReason(
        query=primary_query,
        candidates=_candidates_for_log(all_candidates),
        reason="no_candidate_passed_acceptance_criteria"
        if all_candidates
        else "no_search_results",
    )
    logger.info(
        "No match for '%s' by %s (%d candidates checked)",
        song.title,
        ", ".join(song.artists),
        len(all_candidates),
    )
    return None, reason
