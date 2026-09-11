---
name: safe-fetch-evaluation
description: Fetch evaluation sources with retries, optional browser rendering, academic-source alternatives, PDF extraction, readable-page fallback, and final-URL evidence controls.
---

# Evaluation Safe Fetch

This fetcher is available only in `drb_evaluation`. It preserves the normal
public-URL and redirect safety checks while adding bounded retries, PDF text
extraction, academic source alternatives, and an optional reader fallback for
empty pages or recoverable public-access failures such as 403, 405, timeout,
transport errors, and mislabeled document MIME types. Returned content is still untrusted and must pass the
ordinary evidence-extraction and citation pipeline.

When static HTML is only a JavaScript shell, one bounded Scrapling
`DynamicFetcher` render is allowed. Dynamic rendering remains subject to the
same public initial/final URL, MIME, response-size, timeout, and challenge-page
checks. The first recoverable static timeout, transport failure, 403/429, or
challenge response switches directly to the browser instead of consuming the
remaining static retries. Stealth mode and CAPTCHA/Cloudflare bypass are not
used.

For academic URLs, prefer open scholarly indexes and repositories in this
order: Semantic Scholar, OpenAlex, arXiv, PubMed/PMC, bioRxiv/medRxiv,
Crossref, and Unpaywall. Publisher pages known to spend requests on bot or
subscription gates (including Wiley, ScienceDirect, Scopus, and Web of
Science) should fail over immediately instead of consuming repeated retries.
When a failed academic URL has no DOI, its search-result title may be used for
a strong-title OpenAlex match. At most three non-restrictive alternative URLs
are attempted with bounded wall-clock time before the fetcher returns a clearly
labelled abstract or bibliographic record.

Reader fallback content is an access aid only. It is recorded as a retrieval
method while the source URI remains the original public URL; it never permits
search summaries or model-generated text to become evidence.
