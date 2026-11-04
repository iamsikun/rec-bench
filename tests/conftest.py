"""Test setup: a private dataset cache, and no network access outside ``-m network`` tests."""

import socket

import pytest


@pytest.fixture(autouse=True, scope="session")
def _data_root(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("RECSYS_DATA", str(tmp_path_factory.mktemp("recsys-data")))
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    if any(request.node.get_closest_marker(m) for m in ("network", "reference", "perf")):
        return

    def guard(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise RuntimeError(f"network access in an offline test: {address}")
        return real_connect(self, address, *args, **kwargs)

    real_connect = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", guard)
