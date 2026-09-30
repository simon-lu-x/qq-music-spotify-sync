"""Tests for SpotifyClient."""
from unittest.mock import MagicMock, patch, call

import pytest

from qq_spotify_sync.config import Config
from qq_spotify_sync.spotify_client import (
    SpotifyClient,
    SpotifyError,
    SpotifyTrack,
    _playlist_description,
    _build_client,
)


def _make_config(**overrides) -> Config:
    defaults = dict(
        spotify_client_id="cid",
        spotify_client_secret="csecret",
        spotify_redirect_uri="http://localhost:8888/callback",
        spotify_refresh_token="rtoken",
        spotify_playlist_id="",
        spotify_playlist_name="QQ音乐热歌榜",
        qq_music_top_id=26,
        qq_music_num=100,
        telegram_bot_token="",
        telegram_chat_id="",
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_client(sp_mock: MagicMock, **config_overrides) -> SpotifyClient:
    config = _make_config(**config_overrides)
    client = SpotifyClient.__new__(SpotifyClient)
    client._config = config
    client._sp = sp_mock
    client._current_user_id = "testuser"
    return client


class TestSearchTracks:
    def test_returns_tracks(self):
        sp = MagicMock()
        sp.search.return_value = {
            "tracks": {
                "items": [
                    {
                        "uri": "spotify:track:abc",
                        "name": "漠河舞厅",
                        "artists": [{"name": "柳爽"}],
                        "duration_ms": 270_000,
                        "popularity": 80,
                    }
                ]
            }
        }
        client = _make_client(sp)
        results = client.search_tracks("漠河舞厅 柳爽")
        assert len(results) == 1
        assert results[0].uri == "spotify:track:abc"
        assert results[0].artists == ["柳爽"]

    def test_returns_empty_list_when_no_results(self):
        sp = MagicMock()
        sp.search.return_value = {"tracks": {"items": []}}
        client = _make_client(sp)
        assert client.search_tracks("nonexistent song xyz") == []


class TestEnsurePlaylist:
    def test_returns_configured_playlist_id_directly(self):
        sp = MagicMock()
        client = _make_client(sp, spotify_playlist_id="existing-id")
        result = client.ensure_playlist()
        assert result == "existing-id"
        sp.current_user_playlists.assert_not_called()

    def test_finds_existing_managed_playlist(self):
        sp = MagicMock()
        sp.current_user_playlists.return_value = {
            "items": [
                {
                    "id": "managed-pl-id",
                    "name": "QQ音乐热歌榜",
                    "owner": {"id": "testuser"},
                    "description": "最近更新：2026-04-02",
                }
            ],
            "next": None,
        }
        client = _make_client(sp)
        result = client.ensure_playlist()
        assert result == "managed-pl-id"
        sp.current_user_playlist_create.assert_not_called()

    def test_ignores_playlist_owned_by_other_user(self):
        sp = MagicMock()
        sp.current_user_playlists.return_value = {
            "items": [
                {
                    "id": "other-pl-id",
                    "name": "QQ音乐热歌榜",
                    "owner": {"id": "someone_else"},
                    "description": "最近更新：2026-04-02",
                }
            ],
            "next": None,
        }
        sp.current_user_playlist_create.return_value = {"id": "new-pl-id"}
        client = _make_client(sp)
        result = client.ensure_playlist()
        assert result == "new-pl-id"

    def test_creates_playlist_when_none_found(self):
        sp = MagicMock()
        sp.current_user_playlists.return_value = {"items": [], "next": None}
        sp.current_user_playlist_create.return_value = {"id": "brand-new-id"}
        client = _make_client(sp)
        result = client.ensure_playlist()
        assert result == "brand-new-id"
        sp.current_user_playlist_create.assert_called_once()
        call_kwargs = sp.current_user_playlist_create.call_args
        assert call_kwargs.kwargs.get("description", "").startswith("最近更新：")

    def test_paginates_through_playlists(self):
        sp = MagicMock()
        sp.current_user_playlists.side_effect = [
            {
                "items": [{"id": "other", "name": "other", "owner": {"id": "testuser"}, "description": ""}],
                "next": "page2",
            },
            {
                "items": [
                    {
                        "id": "target-pl",
                        "name": "QQ音乐热歌榜",
                        "owner": {"id": "testuser"},
                        "description": "最近更新：2026-04-02",
                    }
                ],
                "next": None,
            },
        ]
        client = _make_client(sp)
        result = client.ensure_playlist()
        assert result == "target-pl"

    def test_updates_playlist_metadata_with_date(self):
        sp = MagicMock()
        client = _make_client(sp)
        client.update_playlist_metadata("pl-id", "2026-04-02")
        sp.playlist_change_details.assert_called_once_with(
            "pl-id",
            description="最近更新：2026-04-02",
        )


class TestReplacePlaylistTracks:
    def test_replaces_tracks(self):
        sp = MagicMock()
        client = _make_client(sp)
        uris = [f"spotify:track:{i}" for i in range(10)]
        client.replace_playlist_tracks("pl-id", uris)
        sp.playlist_replace_items.assert_called_once_with("pl-id", uris)

    def test_truncates_to_100(self):
        sp = MagicMock()
        client = _make_client(sp)
        uris = [f"spotify:track:{i}" for i in range(150)]
        client.replace_playlist_tracks("pl-id", uris)
        actual_uris = sp.playlist_replace_items.call_args[0][1]
        assert len(actual_uris) == 100


class TestBuildClient:
    @patch("qq_spotify_sync.spotify_client.SpotifyOAuth")
    def test_invalid_grant_includes_reauth_hint(self, oauth_cls):
        from spotipy.oauth2 import SpotifyOauthError

        oauth_cls.return_value.refresh_access_token.side_effect = SpotifyOauthError(
            "error: invalid_grant, error_description: Refresh token revoked",
            error="invalid_grant",
            error_description="Refresh token revoked",
        )
        with pytest.raises(SpotifyError, match="get_refresh_token.py"):
            _build_client(_make_config())

    @patch("qq_spotify_sync.spotify_client.SpotifyOAuth")
    def test_other_errors_have_no_reauth_hint(self, oauth_cls):
        oauth_cls.return_value.refresh_access_token.side_effect = RuntimeError("network down")
        with pytest.raises(SpotifyError) as excinfo:
            _build_client(_make_config())
        assert "get_refresh_token.py" not in str(excinfo.value)
