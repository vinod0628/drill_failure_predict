"""Heavy block model work in worker processes, so the window never stops responding.

Python threads share one interpreter lock: long number crunching in a thread can still starve the window
("Not Responding"). Here the work runs in a pool of separate processes (one per core, less one for the
window). The caller - a thread of the GUI - only hands out pieces and collects them, reports progress with
the time left, and can cancel between pieces.

* ``points_inside``   blocks inside a solid, in chunks (New block model from a solid)
* ``run``             any other single task

Small jobs run in the calling thread: starting worker processes takes a second or two.
The pool is started on first use and kept until ``shutdown`` (manager closed, app closed).
(Grade estimation is not part of this application: use GeoViewer.)
"""

from __future__ import annotations

import concurrent.futures as cf
import math
import multiprocessing as mp
import os
import threading
import time

import numpy as np

from ..i18n import tr

SMALL_INSIDE = 300_000       # points: fewer are tested in the calling thread
HEARTBEAT = 0.25             # seconds between progress reports / cancel checks while waiting

_pool: cf.ProcessPoolExecutor | None = None


class Cancelled(Exception):
    """The user cancelled a long block model task."""


_lock = threading.Lock()


def n_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _watch_parent() -> None:
    """End this worker when GeoViewer is gone (closed, crashed or killed): on Windows an
    orphaned worker would otherwise wait for work forever."""
    parent = mp.parent_process()
    while parent is not None:
        try:
            parent.join(2.0)
            if not parent.is_alive():
                break
        except Exception:  # noqa: BLE001 - cannot tell: assume it is gone
            break
    os._exit(0)


def _init_worker(lang: str) -> None:
    from .. import i18n

    try:
        i18n.set_language(lang)
    except Exception:  # noqa: BLE001 - messages in English then
        pass
    threading.Thread(target=_watch_parent, daemon=True, name="parent-watch").start()


def pool() -> cf.ProcessPoolExecutor:
    global _pool
    with _lock:
        if _pool is None:
            from .. import i18n

            for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")          # one core per worker: the workers are the parallelism
            _pool = cf.ProcessPoolExecutor(n_workers(), mp_context=mp.get_context("spawn"),
                                           initializer=_init_worker, initargs=(i18n.language().code,))
        return _pool


def shutdown() -> None:
    """Stop the worker processes now - work still running in them is dropped (they are
    started again when needed)."""
    global _pool
    with _lock:
        p, _pool = _pool, None
    if p is None:
        return
    procs = list((getattr(p, "_processes", None) or {}).values())
    p.shutdown(wait=False, cancel_futures=True)
    for proc in procs:
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:  # noqa: BLE001 - already gone
            pass
    for proc in procs:
        try:
            proc.join(2.0)
        except Exception:  # noqa: BLE001
            pass


def _broken() -> None:
    shutdown()                  # a worker died (e.g. out of memory): a new pool next time


def _eta(start: float, frac: float) -> str:
    if frac <= 0.02:
        return ""
    left = (time.time() - start) * (1 - frac) / frac
    if left < 60:
        return tr(" - about {s} s left", s=max(1, int(round(left))))
    return tr(" - about {m} min left", m=int(math.ceil(left / 60)))


def _collect(futures: dict, on_result, progress, cancel, total: float, label) -> None:
    """Wait for ``futures`` ({future: weight}), handing each result to ``on_result(key,
    result)``; ``label(done)`` gives the progress text."""
    start = time.time()
    done_w = 0.0
    pending = set(futures)
    try:
        while pending:
            if cancel is not None and cancel():
                for f in pending:
                    f.cancel()
                raise Cancelled()
            done, pending = cf.wait(pending, timeout=HEARTBEAT, return_when=cf.FIRST_COMPLETED)
            for f in done:
                on_result(futures[f], f.result())
                done_w += futures[f][1]
            if progress is not None:
                frac = done_w / max(total, 1e-9)
                progress(min(frac, 0.999), label(done_w) + _eta(start, frac))
    except cf.process.BrokenProcessPool:
        _broken()
        raise RuntimeError(tr("A worker process stopped (out of memory?). Try again with fewer blocks or "
                              "a smaller search.")) from None


# --- estimation -------------------------------------------------------------------------------
# --- inside a solid -----------------------------------------------------------------------------
def _inside_chunk(points, vertices, triangles):
    from ..scene.solid import points_inside as inside

    return inside(points, vertices, triangles)


def points_inside(points: np.ndarray, vertices, triangles, progress=None, cancel=None) -> np.ndarray:
    """``scene.solid.points_inside`` over all cores for many points."""
    N = len(points)
    if N < SMALL_INSIDE or n_workers() == 1:
        return _inside_chunk(points, vertices, triangles)
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(triangles, dtype=np.int64)
    used = np.unique(F)                                  # send only the mesh's own vertices
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    V, F = V[used], remap[F]
    out = np.zeros(N, dtype=bool)
    chunk = int(max(100_000, math.ceil(N / (n_workers() * 4))))
    ex = pool()
    futures = {}
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        futures[ex.submit(_inside_chunk, np.ascontiguousarray(points[s:e]), V, F)] = ((s, e), e - s)

    def put(key, res):
        s, e = key[0]
        out[s:e] = res

    _collect(futures, put, progress, cancel, N,
             lambda d: tr("Blocks inside the solid: {n} of {total} tested", n=f"{int(d):,}", total=f"{N:,}"))
    return out


# --- any single task -------------------------------------------------------------------------------
def run(fn, *args, progress=None, cancel=None, text: str = ""):
    """``fn(*args)`` in a worker process (``fn`` must be a module-level function); progress
    shows ``text`` with the time spent."""
    ex = pool()
    out = {}
    start = time.time()

    def put(_key, res):
        out["r"] = res

    _collect({ex.submit(fn, *args): (0, 1)}, put,
             (lambda _f, _t: progress(None, tr("{text} ({s} s)", text=text, s=int(time.time() - start))))
             if progress is not None else None, cancel, 1, lambda _d: "")
    return out["r"]
