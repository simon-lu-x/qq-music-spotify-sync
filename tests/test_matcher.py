"""Tests for the matcher module: normalization, scoring, and matching logic."""
import pytest
from unittest.mock import MagicMock

from qq_spotify_sync.matcher import (
    MatchResult,
    _artists_overlap,
    _artist_aliases,
    _best_candidate,
    _has_special_version_tag,
    _is_strong_primary_match,
    _normalize,
    _score_candidate,
    _title_similarity,
    match_songs,
)
from qq_spotify_sync.qq_music import QQSong
from qq_spotify_sync.spotify_client import SpotifyTrack


def make_song(title="漠河舞厅", artists=None, duration_ms=270_000) -> QQSong:
    return QQSong(title=title, artists=artists or ["柳爽"], album="", duration_ms=duration_ms)


def make_track(name="漠河舞厅", artists=None, duration_ms=270_000, uri="spotify:track:abc") -> SpotifyTrack:
    return SpotifyTrack(uri=uri, name=name, artists=artists or ["柳爽"], duration_ms=duration_ms)


class TestNormalize:
    def test_strips_parentheses(self):
        assert _normalize("漠河舞厅 (Live)") == "漠河舞厅"

    def test_strips_chinese_brackets(self):
        assert _normalize("漠河舞厅（现场版）") == "漠河舞厅"

    def test_strips_feat(self):
        assert _normalize("Song Title feat. Another Artist") == "song title"

    def test_strips_ft(self):
        assert _normalize("Song ft. Someone") == "song"

    def test_lowercases(self):
        assert _normalize("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize("hello   world") == "hello world"

    def test_full_width_to_ascii(self):
        result = _normalize("ａｂｃ")  # full-width letters
        assert result == "abc"

    def test_converts_traditional_to_simplified(self):
        assert _normalize("戀人") == "恋人"

    def test_strips_book_title_metadata(self):
        assert _normalize("一念《逐玉》") == "一念"

    def test_strips_movie_theme_suffix(self):
        assert _normalize('如果可以 - 電影"月老"主題曲') == "如果可以"

    def test_strips_wave_suffix(self):
        assert _normalize("愛情訊息 - 第6波") == "爱情讯息"

    def test_normalizes_live_suffix_style(self):
        assert _normalize("慢冷 - Live") == "慢冷"

    def test_normalizes_rap_suffix_style(self):
        assert _normalize("茶花开了，该回家了 - 《茶花开了》说唱版") == "茶花开了,该回家了"

    def test_strips_soundtrack_metadata_inside_brackets(self):
        assert _normalize("天赋 (《锦绣未央》电视剧片尾曲)") == "天赋"


class TestArtistAliases:
    def test_extracts_cjk_alias_from_parentheses(self):
        aliases = _artist_aliases("Dizzy Dizzo (蔡诗芸)")
        assert "蔡诗芸" in aliases
        assert "dizzy dizzo" in aliases

    def test_extracts_mixed_script_aliases(self):
        aliases = _artist_aliases("G.E.M. 邓紫棋")
        assert "g e m" in aliases
        assert "邓紫棋" in aliases


class TestArtistsOverlap:
    def test_exact_match(self):
        assert _artists_overlap(["柳爽"], ["柳爽"]) == 1

    def test_case_insensitive(self):
        assert _artists_overlap(["Artist A"], ["artist a"]) == 1

    def test_no_overlap(self):
        assert _artists_overlap(["Artist A"], ["Artist B"]) == 0

    def test_partial_overlap(self):
        assert _artists_overlap(["A", "B", "C"], ["B", "D"]) == 1

    def test_matches_alias_in_parentheses(self):
        assert _artists_overlap(["Dizzy Dizzo (蔡诗芸)"], ["蔡詩蕓"]) == 1


class TestHasSpecialVersionTag:
    def test_live(self):
        assert _has_special_version_tag("Song (Live)")

    def test_remastered(self):
        assert _has_special_version_tag("Song - Remastered 2021")

    def test_remix(self):
        assert _has_special_version_tag("Song (Remix)")

    def test_normal_title(self):
        assert not _has_special_version_tag("漠河舞厅")


class TestScoreCandidate:
    def test_perfect_match_scores_high(self):
        song = make_song()
        track = make_track()
        score = _score_candidate(song, track, source_has_special=False)
        assert score is not None
        assert score >= 0.9

    def test_rejects_low_title_similarity(self):
        song = make_song(title="完全不同的歌")
        track = make_track(name="Another Song Entirely")
        score = _score_candidate(song, track, source_has_special=False)
        assert score is None

    def test_rejects_no_artist_overlap(self):
        song = make_song(artists=["柳爽"])
        track = make_track(artists=["邓紫棋"])
        score = _score_candidate(song, track, source_has_special=False)
        assert score is None

    def test_allows_strong_primary_match_without_artist_overlap(self):
        song = make_song(title="恋人", artists=["李荣浩"], duration_ms=276_000)
        track = make_track(name="戀人", artists=["Ronghao Li"], duration_ms=275_912)
        score = _score_candidate(
            song,
            track,
            source_has_special=False,
            allow_primary_artist_fallback=True,
        )
        assert score is not None

    def test_rejects_live_version_when_source_is_not_live(self):
        song = make_song(title="漠河舞厅")
        track = make_track(name="漠河舞厅 (Live)")
        score = _score_candidate(song, track, source_has_special=False)
        assert score is None

    def test_allows_live_version_when_source_is_live(self):
        song = make_song(title="漠河舞厅 (Live)")
        track = make_track(name="漠河舞厅 (Live)")
        score = _score_candidate(song, track, source_has_special=True)
        assert score is not None

    def test_rejects_studio_version_when_source_is_live(self):
        song = make_song(title="漠河舞厅 (Live)")
        track = make_track(name="漠河舞厅")
        score = _score_candidate(song, track, source_has_special=True)
        assert score is None

    def test_rejects_foreign_language_version_when_source_is_plain(self):
        song = make_song(title="如果可以", artists=["韦礼安"], duration_ms=274_000)
        track = make_track(name="如果可以 (韓文版)", artists=["WeiBird"], duration_ms=274_323)
        score = _score_candidate(
            song,
            track,
            source_has_special=False,
            allow_primary_artist_fallback=True,
        )
        assert score is None

    def test_rejects_duration_too_different(self):
        song = make_song(duration_ms=270_000)
        track = make_track(duration_ms=270_000 + 20_000)  # 20s difference
        score = _score_candidate(song, track, source_has_special=False)
        assert score is None

    def test_neutral_score_when_duration_unknown(self):
        song = make_song(duration_ms=0)  # Unknown
        track = make_track(duration_ms=999_000)
        # Should not be rejected due to duration, but may fail on other criteria
        song2 = make_song(duration_ms=0, title="漠河舞厅", artists=["柳爽"])
        track2 = make_track(duration_ms=999_000)
        score = _score_candidate(song2, track2, source_has_special=False)
        assert score is not None  # Not rejected by duration


class TestBestCandidate:
    def test_picks_best_scoring_candidate(self):
        song = make_song()
        worse = make_track(name="漠河舞厅 (Remix)", uri="uri:worse")
        better = make_track(name="漠河舞厅", uri="uri:better")
        result = _best_candidate(song, [worse, better])
        assert result is not None
        assert result.uri == "uri:better"

    def test_returns_none_when_no_candidates_pass(self):
        song = make_song()
        bad = make_track(name="Completely Different Song", artists=["Nobody"])
        assert _best_candidate(song, [bad]) is None

    def test_returns_none_for_empty_candidates(self):
        song = make_song()
        assert _best_candidate(song, []) is None

    def test_accepts_primary_query_alias_match(self):
        song = make_song(title="那天下雨了", artists=["周杰伦"], duration_ms=223_000)
        candidate = make_track(name="那天下雨了", artists=["Jay Chou"], duration_ms=223_333, uri="uri:jay")
        result = _best_candidate(song, [candidate], allow_primary_artist_fallback=True)
        assert result is not None
        assert result.uri == "uri:jay"


class TestStrongPrimaryMatch:
    def test_requires_near_exact_title(self):
        song = make_song(title="恋人", artists=["李荣浩"], duration_ms=276_000)
        track = make_track(name="恋人未满", artists=["Ronghao Li"], duration_ms=275_000)
        assert not _is_strong_primary_match(song, track)

    def test_rejects_same_script_wrong_artist_fallback(self):
        song = make_song(title="爱情讯息", artists=["郭静"], duration_ms=250_000)
        track = make_track(name="爱情讯息", artists=["浠然"], duration_ms=250_000)
        assert not _is_strong_primary_match(song, track)


class TestMatchSongs:
    def _make_spotify(self, search_returns: list[SpotifyTrack]):
        sp = MagicMock()
        sp.search_tracks.return_value = search_returns
        return sp

    def test_matches_song(self):
        song = make_song()
        track = make_track()
        sp = self._make_spotify([track])
        result = match_songs([song], sp)
        assert len(result.matched) == 1
        assert result.matched[0][0] is song

    def test_records_unmatched(self):
        song = make_song()
        sp = self._make_spotify([])
        result = match_songs([song], sp)
        assert len(result.unmatched) == 1
        assert result.unmatched[0][0] is song
        assert result.unmatched[0][1].reason == "no_search_results"

    def test_unmatched_reason_with_candidates(self):
        song = make_song()
        bad_track = make_track(name="Completely Different", artists=["Nobody"])
        sp = self._make_spotify([bad_track])
        result = match_songs([song], sp)
        assert len(result.unmatched) == 1
        assert result.unmatched[0][1].reason == "no_candidate_passed_acceptance_criteria"
        # Both phases return candidates (same bad track), so we get >= 1
        assert len(result.unmatched[0][1].candidates) >= 1


class TestLatinSubtitleTitle:
    def test_chinese_title_matches_title_with_english_subtitle(self):
        assert _title_similarity("甲乙丙丁 (你我怎么两清)", "甲乙丙丁Strangers") >= 0.95

    def test_subtitle_rule_works_in_either_direction(self):
        assert _title_similarity("甲乙丙丁Strangers", "甲乙丙丁") >= 0.95

    def test_different_chinese_titles_not_affected(self):
        assert _title_similarity("异想天开", "异想天开了") < 0.95
        assert _title_similarity("甲乙丙丁", "甲乙丙丁戊己") < 0.95

    def test_pure_english_titles_not_affected(self):
        assert _title_similarity("Dear John", "Dear John Letter") < 0.95

    def test_matches_english_stage_name_artist_with_subtitle_title(self):
        song = make_song("甲乙丙丁 (你我怎么两清)", ["李佳薇"], duration_ms=210_000)
        track = make_track("甲乙丙丁Strangers", ["Jess Lee"], duration_ms=210_461)
        assert _best_candidate(song, [track], allow_primary_artist_fallback=True) == track

    def test_duration_still_must_match(self):
        song = make_song("甲乙丙丁 (你我怎么两清)", ["李佳薇"], duration_ms=210_000)
        track = make_track("甲乙丙丁Strangers", ["Jess Lee"], duration_ms=260_000)
        assert _best_candidate(song, [track], allow_primary_artist_fallback=True) is None

    def test_cantonese_version_still_rejected(self):
        song = make_song("甲乙丙丁 (你我怎么两清)", ["李佳薇"], duration_ms=210_000)
        track = make_track("甲乙丙丁Strangers - 粵語版", ["Jess Lee"], duration_ms=209_000)
        assert _best_candidate(song, [track], allow_primary_artist_fallback=True) is None
