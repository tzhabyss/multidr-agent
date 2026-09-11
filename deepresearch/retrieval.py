"""Search backends and a redirect-aware safe web fetcher."""

from __future__ import annotations

import asyncio
import gc
import html
import io
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, Field

from .schemas import FetchedDocument, SearchHit
from .security import SecurityError, validate_public_url


MULTIPART_PUBLIC_SUFFIXES = frozenset(
    {
        "co.uk",
        "org.uk",
        "ac.uk",
        "com.cn",
        "net.cn",
        "org.cn",
        "gov.cn",
        "edu.cn",
        "com.au",
        "net.au",
        "org.au",
        "co.jp",
        "co.kr",
    }
)

OPEN_ACADEMIC_DOMAINS = (
    "semanticscholar.org",
    "openalex.org",
    "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "biorxiv.org",
    "medrxiv.org",
    "crossref.org",
    "unpaywall.org",
    "doaj.org",
    "econstor.eu",
    "handle.net",
    "core.ac.uk",
    "zenodo.org",
    "figshare.com",
)

RESTRICTIVE_ACADEMIC_DOMAINS = (
    "onlinelibrary.wiley.com",
    "sciencedirect.com",
    "scopus.com",
    "webofscience.com",
    "link.springer.com",
    "tandfonline.com",
    "mdpi.com",
    "ieeexplore.ieee.org",
    "jstor.org",
    "dl.acm.org",
    "journals.sagepub.com",
    "brill.com",
    "academic.oup.com",
    "cambridge.org",
    "nature.com",
    "researchgate.net",
    "academia.edu",
    "ssrn.com",
)

ACADEMIC_RESOLUTION_DOMAINS = (
    "semanticscholar.org",
    "openalex.org",
    "crossref.org",
    "doi.org",
    "onlinelibrary.wiley.com",
    "sciencedirect.com",
    "link.springer.com",
    "tandfonline.com",
    "mdpi.com",
    "ieeexplore.ieee.org",
    "jstor.org",
    "dl.acm.org",
    "journals.sagepub.com",
    "brill.com",
    "academic.oup.com",
    "cambridge.org",
    "nature.com",
    "researchgate.net",
    "academia.edu",
    "ssrn.com",
)


def academic_access_rank(url: str) -> int:
    """Prefer open scholarly indexes and repositories over publisher pages."""

    host = (urlsplit(url).hostname or "").lower()
    if any(host == domain or host.endswith("." + domain) for domain in OPEN_ACADEMIC_DOMAINS):
        return 0
    if any(
        host == domain or host.endswith("." + domain)
        for domain in RESTRICTIVE_ACADEMIC_DOMAINS
    ):
        return 2
    return 1


def normalize_academic_text(value: Any) -> str:
    """Normalize API text while removing Crossref/JATS markup."""

    if isinstance(value, list):
        value = " ".join(str(item) for item in value if item is not None)
    if value is None:
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(value)))
    return re.sub(r"\s+", " ", text).strip()


def abstract_from_inverted_index(value: Any) -> str:
    """Reconstruct OpenAlex's abstract text from its public inverted index."""

    if not isinstance(value, dict):
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned.append((position, str(word)))
    return " ".join(word for _, word in sorted(positioned))


def independent_domain(url: str) -> str:
    """Return a conservative registrable-domain identity for diversity checks."""
    parsed = urlsplit(url)
    if parsed.scheme.lower() == "file":
        # Local documents have no network domain. Treat each canonical path as
        # one independent source identity so local-only research can satisfy
        # the same structural evidence floor as a web source.
        return f"file:{parsed.path.lower()}" if parsed.path else ""
    host = (parsed.hostname or "").rstrip(".").lower()
    labels = [item for item in host.split(".") if item]
    if len(labels) <= 2 or all(item.isdigit() for item in labels):
        return host
    suffix = ".".join(labels[-2:])
    if suffix in MULTIPART_PUBLIC_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return suffix


class SearchBackend(Protocol):
    name: str

    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        ...


class SearchToolInput(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=10)


class FetchToolInput(BaseModel):
    url: str
    title: str | None = None


def url_matches_domain_policy(
    url: str,
    *,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> bool:
    """Apply exact-or-subdomain matching; a block rule always wins."""
    host = (urlsplit(url).hostname or "").rstrip(".").lower()
    if not host:
        return False

    def matches(rule: str) -> bool:
        normalized = rule.strip().rstrip(".").lower()
        if normalized.startswith("*."):
            normalized = normalized[2:]
        return bool(normalized) and (host == normalized or host.endswith("." + normalized))

    blocked = blocked_domains or []
    allowed = allowed_domains or []
    if any(matches(rule) for rule in blocked):
        return False
    return not allowed or any(matches(rule) for rule in allowed)


def url_matches_exact_blocklist(
    url: str,
    *,
    blocked_urls: list[str] | None = None,
) -> bool:
    """Return False when a canonical URL is explicitly prohibited."""
    try:
        candidate = canonicalize_url(url).rstrip("/")
    except (TypeError, ValueError):
        return False
    blocked: set[str] = set()
    for item in blocked_urls or []:
        try:
            blocked.add(canonicalize_url(item).rstrip("/"))
        except (TypeError, ValueError):
            continue
    return candidate not in blocked


class DuckDuckGoSearchBackend:
    name = "duckduckgo"

    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise RuntimeError("Install the 'ddgs' package to use DuckDuckGo search") from exc

        def run() -> list[dict[str, Any]]:
            return list(DDGS().text(query, max_results=max_results))

        rows = await asyncio.to_thread(run)
        return [
            SearchHit(
                title=str(row.get("title") or ""),
                url=str(row.get("href") or row.get("url") or ""),
                snippet=str(row.get("body") or row.get("snippet") or ""),
                provider=self.name,
                rank=index,
            )
            for index, row in enumerate(rows, 1)
            if row.get("href") or row.get("url")
        ]


class TavilySearchBackend:
    name = "tavily"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is required")

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        topic: str = "general",
        time_range: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        country: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        chunks_per_source: int | None = None,
    ) -> list[SearchHit]:
        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "topic": topic,
        }
        if time_range:
            payload["time_range"] = time_range
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date
        if country:
            payload["country"] = country
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        if chunks_per_source is not None:
            payload["chunks_per_source"] = chunks_per_source
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json=payload,
            )
            response.raise_for_status()
            rows = response.json().get("results", [])
        return [
            SearchHit(
                title=str(row.get("title") or ""),
                url=str(row.get("url") or ""),
                snippet=str(row.get("content") or ""),
                provider=self.name,
                rank=index,
                score=(float(row["score"]) if row.get("score") is not None else None),
            )
            for index, row in enumerate(rows, 1)
            if row.get("url")
        ]


class WikipediaSearchBackend:
    """Keyless fallback that returns full Wikipedia article URLs."""

    name = "wikipedia"

    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
            response = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "utf8": 1,
                    "srlimit": max_results,
                },
            )
            response.raise_for_status()
            rows = response.json().get("query", {}).get("search", [])
        return [
            SearchHit(
                title=str(row.get("title") or ""),
                url="https://en.wikipedia.org/wiki/" + str(row.get("title") or "").replace(" ", "_"),
                snippet=re.sub(r"<[^>]+>", "", str(row.get("snippet") or "")),
                provider=self.name,
                rank=index,
            )
            for index, row in enumerate(rows, 1)
            if row.get("title")
        ]


class FallbackSearchBackend:
    name = "fallback"

    def __init__(self, backends: list[SearchBackend]):
        if not backends:
            raise ValueError("At least one search backend is required")
        self.backends = backends

    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        errors: list[str] = []
        for backend in self.backends:
            try:
                results = await backend.search(query, max_results=max_results)
                if results:
                    return results
                errors.append(f"{backend.name}: empty")
            except Exception as exc:
                errors.append(f"{backend.name}: {type(exc).__name__}: {exc}")
        raise RuntimeError("All search backends failed: " + "; ".join(errors))


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs):
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str):
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str):
        if self._ignored_depth:
            return
        value = data.strip()
        if value:
            self.parts.append(value)
            if self._in_title:
                self.title += (" " if self.title else "") + value


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    netloc = host
    if port and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        netloc = f"{host}:{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


class SafeWebFetcher:
    ALLOWED_MIME = {
        "text/html",
        "text/plain",
        "application/json",
        "application/xml",
        "text/xml",
    }

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        max_bytes: int = 1_500_000,
        max_redirects: int = 4,
        retries: int = 0,
        retry_backoff_seconds: float = 0.0,
        allow_pdf: bool = False,
        reader_fallback: bool = False,
        academic_fallback: bool = False,
        dynamic_fallback: bool = False,
        dynamic_timeout_seconds: float = 15.0,
        dynamic_probe_timeout_seconds: float = 8.0,
        dynamic_max_concurrency: int = 2,
        academic_max_concurrency: int = 2,
        dynamic_executable_path: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.allow_pdf = allow_pdf
        self.reader_fallback = reader_fallback
        self.academic_fallback = academic_fallback
        self.dynamic_fallback = dynamic_fallback
        self.dynamic_timeout_seconds = max(1.0, float(dynamic_timeout_seconds))
        self.dynamic_probe_timeout_seconds = max(
            1.0, float(dynamic_probe_timeout_seconds)
        )
        self.dynamic_max_concurrency = max(1, int(dynamic_max_concurrency))
        self.academic_max_concurrency = max(1, int(academic_max_concurrency))
        self._dynamic_slots = asyncio.Semaphore(self.dynamic_max_concurrency)
        self._dynamic_exception_handler_guard = asyncio.Lock()
        self._dynamic_exception_handler_users = 0
        self._dynamic_exception_handler_loop: asyncio.AbstractEventLoop | None = None
        self._dynamic_previous_exception_handler: Any = None
        self._dynamic_exception_handler: Any = None
        self._academic_slots = asyncio.Semaphore(self.academic_max_concurrency)
        self.dynamic_executable_path = (
            dynamic_executable_path or os.getenv("SCRAPLING_EXECUTABLE_PATH") or None
        )
        self.transport = transport

    async def fetch(
        self,
        url: str,
        title: str | None = None,
        render: bool = False,
    ) -> FetchedDocument:
        navigation_url = await validate_public_url(url)
        requested_url = canonicalize_url(navigation_url)
        static_attempts = 0
        dynamic_attempted = bool(render)

        def finalize(document: FetchedDocument) -> FetchedDocument:
            return document.model_copy(
                update={
                    "static_attempts": static_attempts,
                    "dynamic_attempted": (
                        dynamic_attempted
                        or document.retrieval_method == "scrapling_dynamic"
                    ),
                }
            )

        if render:
            return finalize(
                await self._fetch_dynamic(
                    requested_url,
                    navigation_url=navigation_url,
                )
            )
        errors: list[Exception] = []
        reader_candidates: list[tuple[str, Exception]] = []
        academic_documents: list[FetchedDocument] = []
        candidate_urls = self._candidate_urls(requested_url)
        academic_resolved = False
        if self.academic_fallback and self._prefers_academic_resolution(requested_url):
            alternatives, academic_documents = await self._academic_candidate_urls(
                requested_url,
                title=title or self._title_from_academic_url(requested_url),
            )
            if alternatives:
                candidate_urls = [
                    *alternatives,
                    *(item for item in candidate_urls if item not in alternatives),
                ]
            academic_resolved = True
        candidate_index = 0
        while candidate_index < len(candidate_urls):
            candidate_url = candidate_urls[candidate_index]
            candidate_errors: list[Exception] = []
            for attempt in range(self.retries + 1):
                try:
                    static_attempts += 1
                    candidate_timeout = (
                        self.timeout_seconds
                        if candidate_index == 0
                        else min(self.timeout_seconds, 6.0)
                    )
                    if (
                        candidate_index == 0
                        and self.dynamic_fallback
                        and not dynamic_attempted
                    ):
                        candidate_timeout = min(
                            candidate_timeout,
                            self.dynamic_probe_timeout_seconds,
                        )
                    document = await asyncio.wait_for(
                        self._fetch_once(
                            requested_url,
                            candidate_url,
                            retrieval_method=(
                                "direct"
                                if candidate_url == requested_url
                                else "alternative"
                            ),
                            timeout_seconds=candidate_timeout,
                        ),
                        timeout=candidate_timeout + 1.0,
                    )
                    if document.content:
                        if self.dynamic_fallback and self._looks_like_dynamic_shell(document):
                            dynamic_attempted = True
                            try:
                                rendered = await self._fetch_dynamic(
                                    requested_url,
                                    navigation_url=navigation_url,
                                )
                                if self._prefer_rendered(document, rendered):
                                    return finalize(rendered)
                            except Exception as exc:
                                errors.append(exc)
                        return finalize(document)
                    raise ValueError("Fetched page contains no readable text")
                except Exception as exc:
                    candidate_errors.append(exc)
                    if (
                        candidate_index == 0
                        and self.dynamic_fallback
                        and not dynamic_attempted
                        and self._dynamic_fallback_allowed(exc)
                    ):
                        dynamic_attempted = True
                        try:
                            return finalize(
                                await self._fetch_dynamic(
                                    requested_url,
                                    navigation_url=navigation_url,
                                )
                            )
                        except Exception as dynamic_exc:
                            errors.append(dynamic_exc)
                    if (
                        not self._retryable(exc)
                        or attempt >= self._retry_limit(exc)
                        or (
                            self.academic_fallback
                            and academic_access_rank(candidate_url) == 2
                        )
                        or candidate_index > 0
                    ):
                        break
                    delay = self._retry_delay(exc, attempt)
                    if delay > 0:
                        await asyncio.sleep(delay)
            errors.extend(candidate_errors)
            if candidate_errors:
                last_candidate_error = candidate_errors[-1]
                if isinstance(last_candidate_error, SecurityError) and not self._reader_fallback_allowed(
                    last_candidate_error
                ):
                    raise last_candidate_error
                if (
                    self._reader_fallback_allowed(last_candidate_error)
                    and not (
                        self.academic_fallback
                        and academic_access_rank(candidate_url) == 2
                    )
                ):
                    reader_candidates.append((candidate_url, last_candidate_error))
            if self.academic_fallback and not academic_resolved:
                academic_resolved = True
                alternatives, academic_documents = await self._academic_candidate_urls(
                    requested_url,
                    title=title,
                )
                candidate_urls.extend(
                    item for item in alternatives if item not in candidate_urls
                )
            candidate_index += 1

        last_fetch_error = errors[-1] if errors else None
        if (
            self.dynamic_fallback
            and not dynamic_attempted
            and last_fetch_error is not None
            and self._dynamic_fallback_allowed(last_fetch_error)
        ):
            dynamic_attempted = True
            try:
                return finalize(
                    await self._fetch_dynamic(
                        requested_url,
                        navigation_url=navigation_url,
                    )
                )
            except Exception as exc:
                errors.append(exc)
        if academic_documents:
            return finalize(academic_documents[0])
        if self.reader_fallback:
            for candidate_url, _ in reader_candidates:
                try:
                    fallback = await self._reader_document(requested_url, candidate_url)
                    if fallback.content:
                        return finalize(fallback)
                except Exception as exc:
                    errors.append(exc)
        last_error = errors[-1] if errors else RuntimeError("Fetch returned no document")
        raise last_error

    @staticmethod
    def _looks_like_dynamic_shell(document: FetchedDocument) -> bool:
        """Identify HTML shells that contain too little rendered page content.

        This is deliberately conservative. A browser is expensive, so ordinary
        readable HTML, JSON, plain text, PDFs, and access challenges never take
        this path.
        """

        if document.content_type != "text/html":
            return False
        sample = document.content[:4_000].casefold()
        if SafeWebFetcher._looks_like_access_challenge(document.title, sample):
            return False
        shell_markers = (
            "please enable javascript",
            "javascript is required",
            "requires javascript",
            "enable javascript to run",
            "loading…",
            "loading...",
        )
        return len(document.content.strip()) < 300 or any(
            marker in sample for marker in shell_markers
        )

    @staticmethod
    def _prefer_rendered(
        static: FetchedDocument, rendered: FetchedDocument
    ) -> bool:
        """Use browser output only when it materially improves the static page."""

        return bool(rendered.content.strip()) and len(rendered.content) > len(
            static.content.strip()
        )

    @staticmethod
    def _is_dynamic_cleanup_warning(context: dict[str, Any]) -> bool:
        diagnostic = " ".join(
            str(item)
            for item in (context.get("exception"), context.get("message"))
            if item
        ).casefold()
        return any(
            marker in diagnostic
            for marker in (
                "targetclosederror",
                "target page, context or browser has been closed",
                "frame was detached",
                "net::err_aborted",
            )
        )

    async def _retain_dynamic_exception_filter(
        self,
    ) -> asyncio.AbstractEventLoop:
        """Install one shared cleanup filter without serializing browser work."""

        loop = asyncio.get_running_loop()
        async with self._dynamic_exception_handler_guard:
            if (
                self._dynamic_exception_handler_users
                and self._dynamic_exception_handler_loop is not loop
            ):
                raise RuntimeError("Dynamic fetcher cannot span event loops")
            if (
                self._dynamic_exception_handler_loop is loop
                and loop.get_exception_handler()
                is self._dynamic_exception_handler
            ):
                self._dynamic_exception_handler_users += 1
                return loop
            if self._dynamic_exception_handler_users == 0:
                previous_handler = loop.get_exception_handler()

                def dynamic_exception_handler(
                    current_loop: asyncio.AbstractEventLoop,
                    context: dict[str, Any],
                ) -> None:
                    if self._is_dynamic_cleanup_warning(context):
                        return
                    if previous_handler is not None:
                        previous_handler(current_loop, context)
                    else:
                        current_loop.default_exception_handler(context)

                self._dynamic_exception_handler_loop = loop
                self._dynamic_previous_exception_handler = previous_handler
                self._dynamic_exception_handler = dynamic_exception_handler
                loop.set_exception_handler(dynamic_exception_handler)
            self._dynamic_exception_handler_users += 1
        return loop

    async def _release_dynamic_exception_filter(
        self, loop: asyncio.AbstractEventLoop
    ) -> None:
        async with self._dynamic_exception_handler_guard:
            if self._dynamic_exception_handler_loop is not loop:
                return
            self._dynamic_exception_handler_users = max(
                0, self._dynamic_exception_handler_users - 1
            )
            if self._dynamic_exception_handler_users:
                return
        # Playwright/Scrapling can retain completed callback futures until a
        # later collection cycle. Drain and collect them while the narrowly
        # scoped cleanup-warning handler is still active.
        await asyncio.sleep(0.25)
        gc.collect()
        await asyncio.sleep(0)
        async with self._dynamic_exception_handler_guard:
            if self._dynamic_exception_handler_users:
                return
            handler = self._dynamic_exception_handler
            if loop.get_exception_handler() is handler:
                loop.set_exception_handler(
                    self._dynamic_previous_exception_handler
                )
            self._dynamic_exception_handler_loop = None
            self._dynamic_previous_exception_handler = None
            self._dynamic_exception_handler = None

    async def _fetch_dynamic(
        self,
        requested_url: str,
        *,
        navigation_url: str | None = None,
    ) -> FetchedDocument:
        """Render one public page with Scrapling's normal Chromium fetcher.

        Scrapling is an optional dependency. We intentionally use
        ``DynamicFetcher`` rather than its stealth variant: challenges and
        CAPTCHAs remain failures and are never bypassed.
        """

        navigation_url = navigation_url or requested_url
        await validate_public_url(navigation_url)
        try:
            from scrapling.fetchers import DynamicFetcher
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Dynamic rendering requires the 'browser' extra and a browser: "
                "pip install -e \".[browser]\" && scrapling install"
            ) from exc

        timeout_ms = max(1_000, int(self.dynamic_timeout_seconds * 1_000))
        blocked_requests: list[Exception] = []

        async def page_setup(page: Any) -> None:
            """Guard browser requests before Chromium sends them."""

            async def guard_route(route: Any, request: Any) -> None:
                target = str(getattr(request, "url", ""))
                try:
                    if getattr(request, "resource_type", "") in {
                        "image",
                        "media",
                        "font",
                    }:
                        await route.abort()
                        return
                    if target.startswith(("http://", "https://")):
                        await validate_public_url(target)
                    elif target.startswith(("ws://", "wss://")):
                        raise SecurityError("Blocked browser websocket request")
                    await route.continue_()
                except Exception as exc:
                    is_navigation = getattr(request, "is_navigation_request", None)
                    if callable(is_navigation) and is_navigation():
                        blocked_requests.append(exc)
                    try:
                        await route.abort()
                    except Exception:
                        pass

            await page.route("**/*", guard_route)

        async def page_action(page: Any) -> None:
            """Wait for a dynamic shell to acquire useful visible text."""

            try:
                await page.wait_for_function(
                    "document.body && document.body.innerText.trim().length >= 300",
                    # Leave enough time for Scrapling to build the response and
                    # close its isolated session before the outer hard timeout.
                    timeout=min(timeout_ms, 5_000),
                )
            except Exception:
                # The normal post-render validation decides whether a short
                # page is useful; a page-specific wait miss is not fatal.
                pass

        async with self._dynamic_slots:
            # Every DynamicFetcher call owns an AsyncDynamicSession. Only the
            # loop-global cleanup-warning filter is shared and reference-counted.
            loop = await self._retain_dynamic_exception_filter()
            try:
                browser_kwargs: dict[str, Any] = {}
                executable_path = self.dynamic_executable_path
                if not executable_path and os.name == "nt":
                    for candidate in (
                        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                    ):
                        if os.path.isfile(candidate):
                            executable_path = candidate
                            break
                if executable_path:
                    browser_kwargs["executable_path"] = executable_path
                fetch_task = asyncio.create_task(
                    DynamicFetcher.async_fetch(
                        navigation_url,
                        headless=True,
                        # Keep scripts/XHR/styles but drop heavy media in the
                        # guarded route above.
                        disable_resources=False,
                        block_ads=False,
                        network_idle=False,
                        load_dom=True,
                        timeout=timeout_ms,
                        wait=1_500,
                        google_search=False,
                        page_setup=page_setup,
                        page_action=page_action,
                        **browser_kwargs,
                    ),
                    name="scrapling-dynamic-fetch",
                )
                try:
                    response = await asyncio.wait_for(
                        fetch_task,
                        timeout=self.dynamic_timeout_seconds + 2.0,
                    )
                except BaseException:
                    if not fetch_task.done():
                        fetch_task.cancel()
                    try:
                        await fetch_task
                    except BaseException:
                        pass
                    raise
            finally:
                # The last active session drains delayed Playwright callbacks
                # before restoring the original loop exception handler.
                await self._release_dynamic_exception_filter(loop)

        if blocked_requests:
            raise blocked_requests[0]

        status = int(getattr(response, "status", 0) or 0)
        if status >= 400:
            raise RuntimeError(f"Dynamic fetch returned HTTP {status}")
        final_url = canonicalize_url(str(getattr(response, "url", requested_url)))
        await validate_public_url(final_url)
        redirect_chain: list[str] = []
        for historical in getattr(response, "history", ()) or ():
            historical_url = getattr(historical, "url", None)
            if not historical_url:
                continue
            safe_historical_url = canonicalize_url(str(historical_url))
            await validate_public_url(safe_historical_url)
            redirect_chain.append(safe_historical_url)

        body = getattr(response, "body", b"")
        if isinstance(body, str):
            raw = body.encode("utf-8")
        else:
            raw = bytes(body)
        if len(raw) > self.max_bytes:
            raise SecurityError("Response body exceeds size limit")
        headers = getattr(response, "headers", {}) or {}
        content_type = str(headers.get("content-type", "text/html")).split(";")[0].lower()
        if content_type not in self.ALLOWED_MIME:
            # Browser responses occasionally omit or normalize the main
            # document's MIME. HTML bytes are still accepted after sniffing.
            content_type = self._sniff_content_type(final_url, raw) or content_type
        if content_type not in self.ALLOWED_MIME:
            raise SecurityError(f"Blocked MIME type: {content_type or 'unknown'}")

        decoded = raw.decode(getattr(response, "encoding", None) or "utf-8", errors="replace")
        title = final_url
        if content_type == "text/html":
            parser = _TextExtractor()
            parser.feed(decoded)
            text = "\n".join(parser.parts)
            title = html.unescape(parser.title) or final_url
        else:
            text = decoded
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            raise ValueError("Dynamic page contains no readable text")
        if self._looks_like_access_challenge(title, text):
            raise ValueError("Access challenge page returned instead of content")
        return FetchedDocument(
            requested_url=requested_url,
            final_url=final_url,
            title=title,
            content=text,
            content_type=content_type,
            http_status=status or 200,
            redirect_chain=redirect_chain,
            retrieval_method="scrapling_dynamic",
        )

    @staticmethod
    def _retryable(error: Exception) -> bool:
        if isinstance(error, (TimeoutError, httpx.TimeoutException, httpx.TransportError)):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in {408, 425, 429, 500, 502, 503, 504}
        return False

    def _retry_limit(self, error: Exception) -> int:
        """Return the highest zero-based retry index allowed for this error."""

        if not self._retryable(error):
            return 0
        if isinstance(
            error,
            (TimeoutError, httpx.TimeoutException, httpx.TransportError),
        ):
            return min(self.retries, 1)
        if isinstance(error, httpx.HTTPStatusError):
            if error.response.status_code == 408:
                return min(self.retries, 1)
            return self.retries
        return self.retries

    def _retry_delay(self, error: Exception, attempt: int) -> float:
        """Respect bounded server backoff without changing fetch accounting."""

        fallback = self.retry_backoff_seconds * (attempt + 1)
        if not isinstance(error, httpx.HTTPStatusError):
            return fallback
        raw = error.response.headers.get("retry-after", "").strip()
        if not raw:
            return fallback
        try:
            delay = float(raw)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(raw)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return fallback
        return max(fallback, min(60.0, max(0.0, delay)))

    @staticmethod
    def _dynamic_fallback_allowed(error: Exception) -> bool:
        """Use a browser only when rendering can plausibly change the result."""

        if isinstance(error, ValueError):
            return not isinstance(error, SecurityError)
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in {403, 405}
        return False

    @staticmethod
    def _reader_fallback_allowed(error: Exception) -> bool:
        if isinstance(error, SecurityError):
            message = str(error).casefold()
            return "blocked mime type" in message or "response body exceeds" in message
        if isinstance(
            error,
            (TimeoutError, httpx.TimeoutException, httpx.TransportError, ValueError),
        ):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in {
                403, 405, 408, 425, 429, 500, 502, 503, 504
            }
        return False

    @staticmethod
    def _candidate_urls(url: str) -> list[str]:
        """Keep the source identity but try a canonical academic entry when needed."""

        candidates = [url]
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path
        arxiv = re.search(r"/(?:abs|html|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", path)
        if host in {"ar5iv.labs.arxiv.org", "arxiv.org", "www.arxiv.org"} and arxiv:
            paper_id = arxiv.group(1)
            candidates.extend(
                [
                    f"https://arxiv.org/abs/{paper_id}",
                    f"https://arxiv.org/pdf/{paper_id}",
                ]
            )
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _prefers_academic_resolution(url: str) -> bool:
        """Resolve indexes and restrictive publisher pages before direct crawling."""

        host = (urlsplit(url).hostname or "").lower()
        return any(
            host == domain or host.endswith("." + domain)
            for domain in ACADEMIC_RESOLUTION_DOMAINS
        )

    @staticmethod
    def _title_from_academic_url(url: str) -> str | None:
        parsed = urlsplit(unquote(url))
        host = (parsed.hostname or "").lower()
        if host == "semanticscholar.org" or host.endswith(".semanticscholar.org"):
            match = re.search(
                r"/paper/([^/]+)/[0-9a-f]{20,}", parsed.path, re.IGNORECASE
            )
            if match:
                return normalize_academic_text(match.group(1).replace("-", " "))
        return None

    @staticmethod
    def _doi_from_url(url: str) -> str | None:
        parsed = urlsplit(unquote(url))
        path = parsed.path.lstrip("/")
        if (parsed.hostname or "").lower() in {"doi.org", "dx.doi.org"}:
            candidate = path
        else:
            match = re.search(
                r"(?:^|/)doi(?:/(?:abs|full|pdf|epdf))?/(10\.\d{4,9}/[^?#]+)",
                parsed.path,
                flags=re.IGNORECASE,
            )
            candidate = match.group(1) if match else ""
        match = re.search(
            r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", candidate, flags=re.IGNORECASE
        )
        return match.group(0).rstrip(".,;)") if match else None

    @staticmethod
    def _normalise_paper_title(value: Any) -> str:
        text = normalize_academic_text(value).casefold()
        return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()

    @classmethod
    def _select_openalex_title_match(
        cls,
        title: str | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Select only a strong title match to avoid fetching the wrong paper."""

        expected = cls._normalise_paper_title(title)
        expected_tokens = set(expected.split())
        if (
            len(expected) < 16
            or len(expected_tokens) < 3
            or expected in {"untitled", "publication", "research paper"}
            or expected.startswith(("http ", "https "))
        ):
            return {}
        best: dict[str, Any] = {}
        best_score = 0.0
        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue
            candidate = cls._normalise_paper_title(
                item.get("display_name") or item.get("title")
            )
            if not candidate:
                continue
            candidate_tokens = set(candidate.split())
            union = expected_tokens | candidate_tokens
            token_score = (
                len(expected_tokens & candidate_tokens) / len(union) if union else 0.0
            )
            sequence_score = SequenceMatcher(None, expected, candidate).ratio()
            containment_score = 0.0
            if expected in candidate or candidate in expected:
                containment_score = min(len(expected), len(candidate)) / max(
                    len(expected), len(candidate)
                )
            score = max(token_score, sequence_score, containment_score)
            if score > best_score:
                best = item
                best_score = score
        return best if best_score >= 0.84 else {}

    async def _academic_candidate_urls(
        self,
        source_url: str,
        *,
        title: str | None = None,
    ) -> tuple[list[str], list[FetchedDocument]]:
        """Resolve DOI or a strong title match to open academic alternatives."""

        async def resolve(name: str, endpoint: str) -> tuple[str, dict[str, Any]]:
            try:
                await validate_public_url(endpoint)
                async with self._academic_slots:
                    async with httpx.AsyncClient(
                        timeout=min(self.timeout_seconds, 20.0),
                        trust_env=False,
                        headers={"User-Agent": "EvidenceDeepResearch/0.1"},
                        transport=self.transport,
                    ) as client:
                        response = await client.get(endpoint)
                        response.raise_for_status()
                        value = response.json()
                return name, value if isinstance(value, dict) else {}
            except Exception:
                return name, {}

        doi = self._doi_from_url(source_url)
        endpoints: dict[str, str] = {}
        payloads: dict[str, dict[str, Any]] = {}
        resolved_from_title = False

        if not doi and title:
            openalex_title = re.sub(
                r"[^\w\s-]+",
                " ",
                normalize_academic_text(title)[:500],
                flags=re.UNICODE,
            ).strip()
            openalex_params: dict[str, Any] = {
                "search": openalex_title,
                "per-page": 5,
            }
            openalex_email = os.getenv("OPENALEX_EMAIL", "").strip()
            if openalex_email:
                openalex_params["mailto"] = openalex_email
            openalex_api_key = os.getenv("OPENALEX_API_KEY", "").strip()
            if openalex_api_key:
                openalex_params["api_key"] = openalex_api_key
            openalex_endpoint = (
                "https://api.openalex.org/works?" + urlencode(openalex_params)
            )
            _, openalex_search = await resolve("openalex_search", openalex_endpoint)
            openalex_match = self._select_openalex_title_match(title, openalex_search)
            if openalex_match:
                resolved_from_title = True
                endpoints["openalex"] = openalex_endpoint
                payloads["openalex"] = openalex_match
                doi = self._doi_from_url(str(openalex_match.get("doi") or ""))

        if not doi and "openalex" not in payloads:
            return [], []

        if doi:
            encoded_doi = quote(doi, safe="")
            if "openalex" not in payloads:
                endpoints["openalex"] = (
                    f"https://api.openalex.org/works/https://doi.org/{encoded_doi}"
                )
            if not resolved_from_title:
                endpoints.update(
                    {
                        "semantic_scholar": (
                            "https://api.semanticscholar.org/graph/v1/paper/"
                            f"DOI:{encoded_doi}?fields=title,abstract,year,authors,venue,url,openAccessPdf"
                        ),
                        "crossref": f"https://api.crossref.org/works/{encoded_doi}",
                        "europe_pmc": (
                            "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                            f"?query=DOI:{encoded_doi}&format=json&resultType=core"
                        ),
                    }
                )
                unpaywall_email = os.getenv("UNPAYWALL_EMAIL", "").strip()
                if unpaywall_email:
                    endpoints["unpaywall"] = (
                        f"https://api.unpaywall.org/v2/{encoded_doi}"
                        f"?email={quote(unpaywall_email, safe='@')}"
                    )

        resolved = await asyncio.gather(
            *(
                resolve(name, endpoint)
                for name, endpoint in endpoints.items()
                if name not in payloads
            )
        )
        payloads.update(dict(resolved))
        candidates: list[str] = []
        documents: list[FetchedDocument] = []

        def add(value: Any) -> None:
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                candidates.append(value)

        def add_document(
            provider: str,
            *,
            title: Any,
            abstract: Any,
            metadata: list[str],
        ) -> None:
            if provider not in endpoints:
                return
            endpoint = endpoints[provider]
            clean_title = normalize_academic_text(title)
            clean_abstract = normalize_academic_text(abstract)
            clean_metadata = [item for item in metadata if item]
            if not clean_title and not clean_abstract and not clean_metadata:
                return
            content = "\n".join(
                [
                    "Evidence scope: abstract and bibliographic metadata only; full text was not accessed.",
                    f"DOI: {doi}" if doi else "",
                    f"Title: {clean_title}" if clean_title else "",
                    f"Abstract: {clean_abstract}" if clean_abstract else "",
                    *clean_metadata,
                ]
            ).strip()
            documents.append(
                FetchedDocument(
                    requested_url=canonicalize_url(source_url),
                    final_url=canonicalize_url(endpoint),
                    title=clean_title or (f"DOI {doi}" if doi else "Academic record"),
                    content=content,
                    content_type="application/json",
                    http_status=200,
                    retrieval_method=(
                        "academic_abstract" if clean_abstract else "academic_metadata"
                    ),
                )
            )

        openalex = payloads.get("openalex", {})
        openalex_abstract = abstract_from_inverted_index(
            openalex.get("abstract_inverted_index")
        )
        add_document(
            "openalex",
            title=openalex.get("display_name") or openalex.get("title"),
            abstract=openalex_abstract,
            metadata=[
                f"Publication year: {openalex.get('publication_year')}"
                if openalex.get("publication_year")
                else "",
                f"Type: {openalex.get('type')}" if openalex.get("type") else "",
            ],
        )
        locations = [
            openalex.get("best_oa_location"),
            openalex.get("primary_location"),
            *openalex.get("locations", []),
        ]
        for location in locations:
            if isinstance(location, dict):
                add(location.get("pdf_url"))
                add(location.get("landing_page_url"))

        unpaywall = payloads.get("unpaywall", {})
        for location in (
            unpaywall.get("best_oa_location"),
            *unpaywall.get("oa_locations", []),
        ):
            if isinstance(location, dict):
                add(location.get("url_for_pdf"))
                add(location.get("url"))

        semantic_scholar = payloads.get("semantic_scholar", {})
        semantic_authors = ", ".join(
            str(item.get("name"))
            for item in semantic_scholar.get("authors", [])
            if isinstance(item, dict) and item.get("name")
        )
        add_document(
            "semantic_scholar",
            title=semantic_scholar.get("title"),
            abstract=semantic_scholar.get("abstract"),
            metadata=[
                f"Authors: {semantic_authors}" if semantic_authors else "",
                f"Year: {semantic_scholar.get('year')}"
                if semantic_scholar.get("year")
                else "",
                f"Venue: {semantic_scholar.get('venue')}"
                if semantic_scholar.get("venue")
                else "",
            ],
        )
        open_access_pdf = semantic_scholar.get("openAccessPdf")
        if isinstance(open_access_pdf, dict):
            add(open_access_pdf.get("url"))
        add(semantic_scholar.get("url"))

        crossref = payloads.get("crossref", {}).get("message", {})
        if isinstance(crossref, dict):
            crossref_titles = crossref.get("title", [])
            crossref_title = (
                crossref_titles[0]
                if isinstance(crossref_titles, list) and crossref_titles
                else crossref_titles
            )
            add_document(
                "crossref",
                title=crossref_title,
                abstract=crossref.get("abstract"),
                metadata=[
                    f"Publisher: {crossref.get('publisher')}"
                    if crossref.get("publisher")
                    else "",
                    f"Type: {crossref.get('type')}" if crossref.get("type") else "",
                ],
            )
            for link in crossref.get("link", []):
                if isinstance(link, dict):
                    add(link.get("URL"))
            resource = crossref.get("resource", {})
            if isinstance(resource, dict):
                primary = resource.get("primary", {})
                if isinstance(primary, dict):
                    add(primary.get("URL"))

        europe_pmc = payloads.get("europe_pmc", {})
        result_list = europe_pmc.get("resultList", {})
        if isinstance(result_list, dict):
            for result in result_list.get("result", []):
                if not isinstance(result, dict):
                    continue
                pmcid = str(result.get("pmcid") or "").strip()
                pmid = str(result.get("pmid") or "").strip()
                if pmcid:
                    add(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/")
                if pmid:
                    add(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
                add_document(
                    "europe_pmc",
                    title=result.get("title"),
                    abstract=result.get("abstractText"),
                    metadata=[
                        f"Authors: {result.get('authorString')}"
                        if result.get("authorString")
                        else "",
                        f"Journal: {result.get('journalTitle')}"
                        if result.get("journalTitle")
                        else "",
                        f"Publication year: {result.get('pubYear')}"
                        if result.get("pubYear")
                        else "",
                    ],
                )

        if doi and doi.casefold().startswith("10.1101/"):
            add(f"https://www.biorxiv.org/content/{doi}")
            add(f"https://www.medrxiv.org/content/{doi}")

        canonical_source = canonicalize_url(source_url)
        unique: list[str] = []
        for candidate in candidates:
            try:
                normalized = canonicalize_url(candidate)
            except (TypeError, ValueError):
                continue
            if (
                normalized != canonical_source
                and academic_access_rank(normalized) != 2
                and normalized not in unique
            ):
                unique.append(normalized)
        unique.sort(
            key=lambda item: (
                academic_access_rank(item),
                0 if urlsplit(item).path.lower().endswith(".pdf") else 1,
            )
        )
        unique = unique[:3]
        documents.sort(
            key=lambda item: (
                item.retrieval_method != "academic_abstract",
                academic_access_rank(item.final_url),
            )
        )
        return unique, documents

    async def _fetch_once(
        self,
        requested_url: str,
        initial_url: str,
        *,
        retrieval_method: str,
        timeout_seconds: float | None = None,
    ) -> FetchedDocument:
        current = initial_url
        redirects: list[str] = []
        async with httpx.AsyncClient(
            timeout=timeout_seconds or self.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": "EvidenceDeepResearch/0.1"},
            transport=self.transport,
        ) as client:
            for _ in range(self.max_redirects + 1):
                await validate_public_url(current)
                async with client.stream("GET", current) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise SecurityError("Redirect response omitted Location")
                        target = canonicalize_url(urljoin(current, location))
                        await validate_public_url(target)
                        redirects.append(target)
                        current = target
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";")[0].lower()
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise SecurityError("Response body exceeds size limit")
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    allowed_mime = set(self.ALLOWED_MIME)
                    if self.allow_pdf:
                        allowed_mime.add("application/pdf")
                    sniffed_type = self._sniff_content_type(current, raw)
                    if sniffed_type and (
                        content_type not in allowed_mime
                        or sniffed_type == "application/pdf"
                        or content_type == "application/pdf"
                    ):
                        content_type = sniffed_type
                    if content_type not in allowed_mime:
                        raise SecurityError(f"Blocked MIME type: {content_type or 'unknown'}")
                    encoding = response.encoding or "utf-8"
                    text = raw.decode(encoding, errors="replace")
                    title = current
                    if content_type == "application/pdf":
                        text = self._extract_pdf_text(raw)
                    elif content_type == "text/html":
                        parser = _TextExtractor()
                        parser.feed(text)
                        text = "\n".join(parser.parts)
                        title = html.unescape(parser.title) or current
                        if self._looks_like_access_challenge(title, text):
                            raise ValueError("Access challenge page returned instead of content")
                    text = re.sub(r"[ \t]+", " ", text)
                    text = re.sub(r"\n{3,}", "\n\n", text).strip()
                    final_url = canonicalize_url(str(response.url))
                    return FetchedDocument(
                        requested_url=requested_url,
                        final_url=final_url,
                        title=title,
                        content=text,
                        content_type=content_type,
                        http_status=response.status_code,
                        redirect_chain=redirects,
                        retrieval_method=retrieval_method,
                    )
        raise SecurityError("Redirect limit exceeded")

    @staticmethod
    def _sniff_content_type(url: str, raw: bytes) -> str:
        """Recover common mislabeled responses without trusting arbitrary bytes."""

        sample = raw[:512].lstrip().lower()
        if raw.startswith(b"%PDF-") or (
            urlsplit(url).path.lower().endswith(".pdf") and raw.startswith(b"%PDF")
        ):
            return "application/pdf"
        if sample.startswith(b"<!doctype html") or sample.startswith(b"<html"):
            return "text/html"
        if sample.startswith(b"<?xml"):
            return "application/xml"
        return ""

    @staticmethod
    def _extract_pdf_text(raw: bytes) -> str:
        errors: list[str] = []
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                return text
            errors.append("pypdf returned no text")
        except Exception as exc:
            errors.append(f"pypdf: {type(exc).__name__}: {exc}")

        # PyMuPDF is optional. When installed (for example alongside Docling),
        # it provides a second parser for PDFs that pypdf cannot decode.
        try:
            import fitz

            document = fitz.open(stream=raw, filetype="pdf")
            text = "\n\n".join(page.get_text("text") for page in document)
            if text.strip():
                return text
            errors.append("pymupdf returned no text")
        except Exception as exc:
            errors.append(f"pymupdf: {type(exc).__name__}: {exc}")
        raise RuntimeError("PDF parsers failed: " + "; ".join(errors))

    @staticmethod
    def _looks_like_access_challenge(title: str, text: str) -> bool:
        sample = f"{title}\n{text[:2000]}".casefold()
        markers = (
            "just a moment",
            "checking your browser",
            "verify you are human",
            "enable javascript and cookies",
            "cf-chl-",
            "captcha",
        )
        return any(marker in sample for marker in markers)

    async def _reader_document(
        self, requested_url: str, source_url: str
    ) -> FetchedDocument:
        content = await self._reader_text(source_url)
        if not content:
            raise ValueError("Reader fallback returned no readable text")
        final_url = canonicalize_url(source_url)
        return FetchedDocument(
            requested_url=requested_url,
            final_url=final_url,
            title=final_url,
            content=content,
            content_type="text/plain",
            http_status=200,
            redirect_chain=[],
            retrieval_method="reader_fallback",
        )

    async def _reader_text(self, source_url: str) -> str:
        """Bounded access fallback for empty pages; never a search-summary source."""

        reader_url = "https://r.jina.ai/http://" + source_url.removeprefix("https://").removeprefix("http://")
        await validate_public_url(reader_url)
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            trust_env=False,
            headers={"User-Agent": "EvidenceDeepResearch/0.1"},
            transport=self.transport,
        ) as client:
            response = await client.get(reader_url)
            response.raise_for_status()
            content = response.text.strip()
        if self._looks_like_access_challenge("", content):
            raise ValueError("Reader returned an access challenge page")
        return content[: self.max_bytes].strip()


class EvaluationWebFetcher(SafeWebFetcher):
    """Shared hardened retrieval used by daily and DRB evaluation runs."""

    def __init__(self, **kwargs: Any) -> None:
        defaults = {
            "timeout_seconds": 30.0,
            "max_bytes": 8_000_000,
            "retries": 2,
            "retry_backoff_seconds": 0.6,
            "allow_pdf": True,
            "reader_fallback": True,
            "academic_fallback": True,
            "dynamic_fallback": kwargs.get("transport") is None,
            "dynamic_max_concurrency": 2,
            "academic_max_concurrency": 2,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
