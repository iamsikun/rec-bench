"""Test setup: a private dataset cache, and no network access outside ``-m network`` tests.

``reference`` and ``perf`` tests use the real cache so they reuse downloaded datasets.
"""

import os
import socket

import pytest
import torch

torch.set_num_threads(1)  # one thread per xdist worker: oversubscription slows the suite
REAL_DATA = os.environ.get("RECSYS_DATA")


@pytest.fixture(autouse=True, scope="session")
def _data_root(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("RECSYS_DATA", str(tmp_path_factory.mktemp("recsys-data")))
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _real_data_for_slow_tests(request, monkeypatch):
    if any(request.node.get_closest_marker(m) for m in ("reference", "perf")):
        if REAL_DATA is None:
            monkeypatch.delenv("RECSYS_DATA")
        else:
            monkeypatch.setenv("RECSYS_DATA", REAL_DATA)


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
