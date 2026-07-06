"""Unit test for the Windows TLS workaround helper (scripts/common/network.py).

See that module's docstring for the two issues it works around (an
OPENSSL_Uplink native crash and a certificate-verification failure), both
observed on a Windows dev machine with Avast/AVG installed.
"""
from __future__ import annotations

import os

from scripts.common.network import apply_windows_tls_workarounds


def test_apply_windows_tls_workarounds_clears_sslkeylogfile(monkeypatch) -> None:
    monkeypatch.setenv("SSLKEYLOGFILE", r"\\.\aswMonFltProxy\FFFFBA87D4D65870")
    apply_windows_tls_workarounds()
    assert "SSLKEYLOGFILE" not in os.environ


def test_apply_windows_tls_workarounds_is_a_noop_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    apply_windows_tls_workarounds()  # must not raise
    assert "SSLKEYLOGFILE" not in os.environ


def test_apply_windows_tls_workarounds_idempotent() -> None:
    # Injecting truststore twice (e.g. called from multiple modules) must not error.
    apply_windows_tls_workarounds()
    apply_windows_tls_workarounds()
