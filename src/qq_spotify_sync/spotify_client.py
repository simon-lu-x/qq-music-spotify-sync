"""Spotify client: OAuth, track search, playlist management."""
from __future__ import annotations

import logging
import time
from datetime import date
from dataclasses import dataclass, field

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from .config import Config

logger = logging.getLogger(__name__)

_SEARCH_LIMIT = 10       # Spotify Feb 2026 max
_BASE_PACING = 0.1       # seconds between search calls


class SpotifyError(Exception):
    """System-level Spotify failure."""


_REAUTH_HINT = (
    "The refresh token is no longer valid. Re-authorize by running "
    "`python scripts/get_refresh_token.py` locally, then update the "
    "SPOTIFY_REFRESH_TOKEN GitHub secret with the new value."
)


@dataclass
class SpotifyTrack:
    uri: str
    name: str
    artists: list[str]
    duration_ms: int
    popularity: int = 0


def _playlist_description(updated_on: str | None = None) -> str:
    date_text = updated_on or date.today().isoformat()
    return f"最近更新：{date_text}"


def _build_client(config: Config) -> spotipy.Spotify:
    auth_manager = SpotifyOAuth(
        client_id=config.spotify_client_id,
        client_secret=config.spotify_client_secret,
        redirect_uri=config.spotify_redirect_uri,
        scope="playlist-modify-public playlist-modify-private playlist-read-private",
    )
    try:
        token_info = auth_manager.refresh_access_token(config.spotify_refresh_token)
    except Exception as exc:
        message = f"Failed to refresh Spotify access token: {exc}"
        if "invalid_grant" in str(exc):
            message = f"{message}. {_REAUTH_HINT}"
        raise SpotifyError(message) from exc

    return spotipy.Spotify(auth=token_info["access_token"])


class SpotifyClient:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._sp = _build_client(config)
        self._current_user_id: str | None = None

    @property
    def current_user_id(self) -> str:
        if self._current_user_id is None:
            try:
                self._current_user_id = self._sp.current_user()["id"]
            except Exception as exc:
                raise SpotifyError(f"Failed to get current Spotify user: {exc}") from exc
        return self._current_user_id

    def search_tracks(self, query: str) -> list[SpotifyTrack]:
        """Search Spotify for tracks. Returns up to SEARCH_LIMIT candidates.

        Handles 429 Retry-After with exponential backoff.
        Raises SpotifyError on non-recoverable HTTP failures.
        """
        backoff = 1.0
        for attempt in range(3):
            try:
                result = self._sp.search(q=query, type="track", limit=_SEARCH_LIMIT)
                items = result.get("tracks", {}).get("items", [])
                return [
                    SpotifyTrack(
                        uri=t["uri"],
                        name=t["name"],
                        artists=[a["name"] for a in t["artists"]],
                        duration_ms=t["duration_ms"],
                        popularity=t.get("popularity", 0),
                    )
                    for t in items
                ]
            except spotipy.SpotifyException as exc:
                if exc.http_status == 429:
                    retry_after = int(
                        getattr(exc, "headers", {}).get("Retry-After", backoff)
                    )
                    logger.warning(
                        "Spotify 429 on search (attempt %d). Waiting %ds.",
                        attempt + 1,
                        retry_after,
                    )
                    time.sleep(retry_after)
                    backoff = min(backoff * 2, 60)
                elif exc.http_status and exc.http_status >= 500:
                    logger.warning(
                        "Spotify 5xx (%d) on search (attempt %d). Retrying in %ds.",
                        exc.http_status,
                        attempt + 1,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                else:
                    raise SpotifyError(f"Spotify search error: {exc}") from exc
            except Exception as exc:
                raise SpotifyError(f"Unexpected error during search: {exc}") from exc

        raise SpotifyError(f"Search failed after 3 attempts for query: {query!r}")

    def ensure_playlist(self) -> str:
        """
        Find or create the managed playlist. Priority:
          1. SPOTIFY_PLAYLIST_ID if configured
          2. Playlist owned by current user whose name exactly matches
          3. Create new playlist

        Returns the playlist ID.
        """
        config = self._config

        # Priority 1: explicit ID
        if config.spotify_playlist_id:
            logger.info("Using configured playlist ID: %s", config.spotify_playlist_id)
            return config.spotify_playlist_id

        # Priority 2: find by owner + exact name
        user_id = self.current_user_id
        playlist_id = self._find_managed_playlist(user_id, config.spotify_playlist_name)
        if playlist_id:
            logger.info("Found existing managed playlist: %s", playlist_id)
            return playlist_id

        # Priority 3: create
        return self._create_managed_playlist(user_id, config.spotify_playlist_name)

    def _find_managed_playlist(self, user_id: str, name: str) -> str | None:
        """Paginate through all user playlists looking for an exact-name match."""
        offset = 0
        while True:
            try:
                result = self._sp.current_user_playlists(limit=50, offset=offset)
            except Exception as exc:
                raise SpotifyError(f"Failed to list playlists: {exc}") from exc

            items = result.get("items", [])
            for pl in items:
                if pl is None:
                    continue
                owner_id = (pl.get("owner") or {}).get("id", "")
                pl_name = pl.get("name", "")
                if owner_id == user_id and pl_name == name:
                    return pl["id"]

            if result.get("next") is None:
                break
            offset += len(items)

        return None

    def _create_managed_playlist(self, user_id: str, name: str) -> str:
        try:
            pl = self._sp.current_user_playlist_create(
                name=name,
                public=True,
                description=_playlist_description(),
            )
        except Exception as exc:
            raise SpotifyError(f"Failed to create playlist: {exc}") from exc

        playlist_id = pl["id"]
        logger.info(
            "Created new managed playlist '%s' (id=%s). "
            "Set SPOTIFY_PLAYLIST_ID=%s to skip lookup next time.",
            name,
            playlist_id,
            playlist_id,
        )
        return playlist_id

    def update_playlist_metadata(self, playlist_id: str, updated_on: str) -> None:
        try:
            self._sp.playlist_change_details(
                playlist_id,
                description=_playlist_description(updated_on),
            )
        except Exception as exc:
            raise SpotifyError(f"Failed to update playlist metadata: {exc}") from exc

        logger.info("Updated playlist %s description to %s", playlist_id, updated_on)

    def replace_playlist_tracks(self, playlist_id: str, track_uris: list[str]) -> None:
        """Replace all tracks in the playlist. track_uris must have <= 100 entries."""
        if len(track_uris) > 100:
            logger.warning(
                "track_uris count %d exceeds 100; truncating.", len(track_uris)
            )
            track_uris = track_uris[:100]

        try:
            self._sp.playlist_replace_items(playlist_id, track_uris)
        except Exception as exc:
            raise SpotifyError(f"Failed to replace playlist tracks: {exc}") from exc

        logger.info(
            "Updated playlist %s with %d tracks.", playlist_id, len(track_uris)
        )

    def get_playlist_url(self, playlist_id: str) -> str:
        return f"https://open.spotify.com/playlist/{playlist_id}"
