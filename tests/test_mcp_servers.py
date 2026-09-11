from __future__ import annotations

import pytest
import httpx

from deepresearch.mcp_servers import github_search, paper_search


@pytest.fixture(autouse=True)
def _reset_paper_search_runtime(monkeypatch):
    paper_search._reset_runtime_state_for_tests()
    monkeypatch.setattr(
        paper_search,
        "_PROVIDER_MIN_INTERVAL_SECONDS",
        {"arxiv": 0.0, "semantic_scholar": 0.0, "openalex": 0.0, "serpapi": 0.0},
    )
    yield
    paper_search._reset_runtime_state_for_tests()


class _Response:
    def __init__(self, *, text: str = "", payload: dict | None = None):
        self.text = text
        self._payload = payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    def __init__(self, response: _Response):
        self.response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


@pytest.mark.asyncio
async def test_arxiv_server_parses_atom_entries(monkeypatch):
    response = _Response(
        text="""
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/1234.5678</id>
            <title> A useful paper </title>
            <summary> A concise abstract. </summary>
            <author><name>A. Researcher</name></author>
            <published>2026-01-01T00:00:00Z</published>
          </entry>
        </feed>
        """
    )
    client = _Client(response)
    monkeypatch.setattr(paper_search.httpx, "AsyncClient", lambda **kwargs: client)

    rows = await paper_search.search_arxiv("retrieval", max_results=2)

    assert rows[0]["title"] == "A useful paper"
    assert rows[0]["url"] == "http://arxiv.org/abs/1234.5678"
    assert rows[0]["authors"] == ["A. Researcher"]


@pytest.mark.asyncio
async def test_openalex_server_prefers_open_pdf_and_applies_filters(monkeypatch):
    response = _Response(
        payload={
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "doi": "https://doi.org/10.1234/openalex",
                    "display_name": "OpenAlex paper",
                    "publication_year": 2024,
                    "abstract_inverted_index": {"Open": [0], "abstract": [1]},
                    "best_oa_location": {
                        "pdf_url": "https://repository.example/openalex.pdf"
                    },
                    "authorships": [
                        {"author": {"display_name": "Open Author"}}
                    ],
                    "cited_by_count": 12,
                }
            ]
        }
    )
    client = _Client(response)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("OPENALEX_EMAIL", raising=False)
    monkeypatch.setattr(paper_search.httpx, "AsyncClient", lambda **kwargs: client)

    rows = await paper_search.search_openalex(
        "retrieval",
        max_results=2,
        from_year=2020,
        to_year=2025,
        open_access_only=True,
    )

    assert rows[0]["url"] == "https://repository.example/openalex.pdf"
    assert rows[0]["doi"] == "https://doi.org/10.1234/openalex"
    assert rows[0]["abstract"] == "Open abstract"
    params = client.calls[0][1]["params"]
    assert params["per-page"] == 2
    assert params["filter"] == (
        "from_publication_date:2020-01-01,"
        "to_publication_date:2025-12-31,is_oa:true"
    )


@pytest.mark.asyncio
async def test_scholar_server_uses_semantic_scholar_fallback(monkeypatch):
    response = _Response(
        payload={
            "data": [
                {
                    "title": "Scholar paper",
                    "abstract": "Scholar abstract",
                    "year": 2025,
                    "url": "https://doi.org/10.1234/example",
                    "authors": [{"name": "A. Author"}],
                }
            ]
        }
    )
    client = _Client(response)
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.setattr(paper_search.httpx, "AsyncClient", lambda **kwargs: client)

    rows = await paper_search.search_google_scholar("retrieval", max_results=2)

    assert rows[0]["title"] == "Scholar paper"
    assert rows[0]["url"] == "https://doi.org/10.1234/example"
    assert rows[0]["provider"] == "semantic_scholar"


@pytest.mark.asyncio
async def test_scholar_server_falls_back_to_openalex_after_rate_limit(monkeypatch):
    class _RateLimitedResponse(_Response):
        def raise_for_status(self):
            request = httpx.Request("GET", "https://api.semanticscholar.org/search")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    responses = [
        _RateLimitedResponse(),
        _RateLimitedResponse(),
        _RateLimitedResponse(),
        _Response(
            payload={
                "results": [
                    {
                        "display_name": "OpenAlex paper",
                        "doi": "https://doi.org/10.1234/openalex",
                        "publication_year": 2024,
                        "abstract_inverted_index": {
                            "OpenAlex": [0],
                            "abstract": [1],
                        },
                        "authorships": [
                            {"author": {"display_name": "Open Author"}}
                        ],
                    }
                ]
            }
        ),
    ]

    class _ClientSequence(_Client):
        async def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return responses.pop(0)

    client = _ClientSequence(_Response())
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.setattr(paper_search.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(
        paper_search,
        "_retry_after_seconds",
        lambda response, *, provider, attempt: 0.0,
    )

    rows = await paper_search.search_google_scholar("retrieval", max_results=2)

    assert rows[0]["title"] == "OpenAlex paper"
    assert rows[0]["abstract"] == "OpenAlex abstract"
    assert rows[0]["provider"] == "openalex"
    assert len(client.calls) == 4


@pytest.mark.asyncio
async def test_semantic_scholar_retries_retry_after_then_succeeds(monkeypatch):
    request = httpx.Request("GET", "https://api.semanticscholar.org/search")

    class _RetryAfterResponse(_Response):
        def raise_for_status(self):
            response = httpx.Response(
                429,
                request=request,
                headers={"Retry-After": "0"},
            )
            raise httpx.HTTPStatusError(
                "rate limited",
                request=request,
                response=response,
            )

    responses = [
        _RetryAfterResponse(),
        _Response(
            payload={
                "data": [
                    {
                        "title": "Recovered scholar paper",
                        "url": "https://doi.org/10.1234/recovered",
                    }
                ]
            }
        ),
    ]

    class _ClientSequence(_Client):
        async def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return responses.pop(0)

    client = _ClientSequence(_Response())
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.setattr(paper_search.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(
        paper_search,
        "_retry_after_seconds",
        lambda response, *, provider, attempt: 0.0,
    )

    rows = await paper_search.search_google_scholar("recovered", max_results=2)

    assert rows[0]["title"] == "Recovered scholar paper"
    assert rows[0]["provider"] == "semantic_scholar"
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_openalex_query_cache_avoids_duplicate_provider_calls(monkeypatch):
    response = _Response(
        payload={
            "results": [
                {
                    "display_name": "Cached OpenAlex paper",
                    "id": "https://openalex.org/WCACHE",
                }
            ]
        }
    )
    client = _Client(response)
    monkeypatch.setattr(paper_search.httpx, "AsyncClient", lambda **kwargs: client)

    first = await paper_search.search_openalex("cache query", max_results=2)
    second = await paper_search.search_openalex("cache query", max_results=2)

    assert first == second
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_scholar_server_falls_back_to_openalex_after_empty_result(monkeypatch):
    responses = [
        _Response(payload={"data": []}),
        _Response(
            payload={
                "results": [
                    {
                        "display_name": "OpenAlex fallback",
                        "doi": "https://doi.org/10.1234/fallback",
                    }
                ]
            }
        ),
    ]

    class _ClientSequence(_Client):
        async def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return responses.pop(0)

    client = _ClientSequence(_Response())
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.setattr(paper_search.httpx, "AsyncClient", lambda **kwargs: client)

    rows = await paper_search.search_google_scholar("retrieval", max_results=2)

    assert rows[0]["title"] == "OpenAlex fallback"
    assert rows[0]["provider"] == "openalex"


@pytest.mark.asyncio
async def test_github_server_parses_repository_and_code_results(monkeypatch):
    responses = [
        _Response(
            payload={
                "items": [
                    {
                        "full_name": "org/research",
                        "html_url": "https://github.com/org/research",
                        "description": "Research agent",
                        "stargazers_count": 10,
                        "language": "Python",
                    }
                ]
            }
        ),
        _Response(
            payload={
                "items": [
                    {
                        "name": "agent.py",
                        "html_url": "https://github.com/org/research/blob/main/agent.py",
                        "path": "agent.py",
                        "repository": {"full_name": "org/research"},
                    }
                ]
            }
        ),
    ]

    class _ClientSequence(_Client):
        async def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return responses.pop(0)

    client = _ClientSequence(_Response())
    monkeypatch.setattr(github_search.httpx, "AsyncClient", lambda **kwargs: client)

    repositories = await github_search.search_repositories("research", max_results=1)
    code = await github_search.search_code("file:agent.py", max_results=1)

    assert repositories[0]["title"] == "org/research"
    assert repositories[0]["url"] == "https://github.com/org/research"
    assert code[0]["title"] == "agent.py"
    assert code[0]["repository"] == "org/research"
