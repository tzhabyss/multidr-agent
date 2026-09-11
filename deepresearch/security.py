"""Network and untrusted-content security primitives."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit


class SecurityError(ValueError):
    pass


BLOCKED_HOSTS = {"localhost", "localhost.localdomain"}


def _is_forbidden_ip(value: str) -> bool:
    ip = ipaddress.ip_address(value)
    return any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        )
    )


async def validate_public_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise SecurityError("Only http and https URLs are allowed")
    if not parsed.hostname or parsed.username or parsed.password:
        raise SecurityError("URL must contain a public host and no embedded credentials")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in BLOCKED_HOSTS:
        raise SecurityError(f"Blocked host: {hostname}")
    try:
        parsed_ip = ipaddress.ip_address(hostname)
    except ValueError:
        parsed_ip = None
    if parsed_ip is not None:
        if _is_forbidden_ip(hostname):
            raise SecurityError(f"Blocked IP address: {hostname}")
        return url

    loop = asyncio.get_running_loop()
    try:
        addresses = await loop.run_in_executor(
            None,
            lambda: socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM),
        )
    except socket.gaierror as exc:
        raise SecurityError(f"DNS resolution failed for {hostname}") from exc
    for address in {item[4][0] for item in addresses}:
        if _is_forbidden_ip(address):
            raise SecurityError(f"Host {hostname} resolves to blocked address {address}")
    return url


def wrap_untrusted_content(content: str, source_uri: str) -> str:
    return (
        "<untrusted_content source="
        + repr(source_uri)
        + ">\n"
        + content
        + "\n</untrusted_content>"
    )
