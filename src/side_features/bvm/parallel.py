"""Evaluate independent operating points on a process pool.

Every expensive thing BVM and RBM do is a *map*: `r_min` pre-scans a reflux
grid, `feasible_band` scans one, `feasibility_map` sweeps (R, S, E/F),
`spectrum` sweeps the feed position, `operating_region` sweeps entrainer ratio.
The points do not talk to each other, and one point is seconds of work on a
non-ideal ternary (3.6 s for one extractive `size_column` before this module
existed), so the sweeps ran for minutes.

PROCESSES, NOT THREADS. A point is spent in Python-level marching loops over
3-element arrays -- numpy never holds an array big enough to release the GIL --
so a `ThreadPoolExecutor` cannot run two of them at once. Marching the
rectifying and stripping sections "in parallel" on threads is the same non-win
one level down, and they are ~10 ms of the seconds a point costs anyway.
`_demo` measures processes against threads rather than arguing about it, but
only asserts the robust half (the pool wins): on Apple Silicon a synthetic
single-threaded loop can land on an efficiency core and make ANY form of
concurrency look like a speedup, so a tight ratio there would be a flaky gate,
not a proof.

`pmap` is the whole interface and it is allowed to be boring:

  * it runs the FIRST item inline, which both seeds the result and times the
    work. Under `SERIAL_BELOW` seconds a pool costs more to start than it saves,
    so the rest is done inline too -- that is why an ideal-thermo BTX sweep
    (14 ms a point) does not pay a 2 s spawn tax to save 80 ms;
  * anything that will not cross a process boundary -- an un-picklable thermo
    closure, a machine that refuses to spawn -- silently falls back to the same
    serial loop. A caller never has to ask;
  * nested calls stay serial (`_IN_WORKER`), so a parallel `operating_region`
    does not have each of its workers try to start a pool of its own;
  * the pool is kept between calls (`get_pool` / `shutdown`) -- `pnarrow` maps
    once per bisection round and re-spawning eight numpy-importing workers each
    round cost more than the round saved.

Measured on the ethanol/water/EG extractive column, same process, back to back:
`bvm.r_min` 144 s -> 43 s (3.4x), `rbm.reflux_band` 24 s -> 10 s (2.3x).

Callables handed to `pmap` must be importable by name (a module-level function
or a `functools.partial` of one) -- a lambda or a local closure is not
picklable, and takes the serial path.
"""

import atexit
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

#: Below this measured cost per item (seconds), a pool is not worth starting.
#: One spawn-based worker costs ~0.2-0.3 s to boot on macOS and the whole point
#: of the pool is to hide seconds, not milliseconds.
SERIAL_BELOW = 0.20

#: Set in the workers, so a sweep nested inside a parallel sweep stays serial.
_IN_WORKER = False


def _init():
    global _IN_WORKER
    _IN_WORKER = True


#: The pool, kept between calls. `pnarrow` maps once per bisection round, and a
#: fresh `ProcessPoolExecutor` each time means re-spawning workers that each
#: re-import numpy and scipy -- measured at ~8 s a round on this project, which
#: was most of what the parallel bisection was supposed to be saving.
_POOL = None


def get_pool():
    """The shared pool, started on first use."""
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=max_workers(), initializer=_init)
    return _POOL


def shutdown():
    """Release the workers. Idle worker processes hold real memory, so the GUI
    calls this when a run finishes rather than keeping eight of them around for
    the life of the window; anything else is covered by `atexit`."""
    global _POOL
    pool, _POOL = _POOL, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


atexit.register(shutdown)


def max_workers():
    """Cores to use. Leaves one for the GUI thread so a sweep does not make the
    window stutter -- the solves are already off the GUI thread, but a pool that
    saturates every core starves the compositor anyway."""
    n = os.cpu_count() or 1
    return max(1, min(n - 1, 8))


def pmap(func, items, on_step=None, cancelled=None):
    """[func(item) for item in items], in parallel when that is worth doing.

    `on_step(done, total)` and `cancelled()` are the same hooks the serial
    sweeps already took. A cancelled run returns what it has, with `None` in
    every slot it never reached -- callers already draw around those.

    Results come back in `items` order regardless of completion order.
    """
    items = list(items)
    if not items:
        return []

    import time
    t0 = time.perf_counter()
    first = func(items[0])
    per_item = time.perf_counter() - t0
    out = [first] + [None] * (len(items) - 1)
    if on_step is not None:
        on_step(1, len(items))
    rest = items[1:]
    if not rest:
        return out

    workers = max_workers()
    if _IN_WORKER or workers < 2 or per_item < SERIAL_BELOW or not _picklable(func):
        for i, item in enumerate(rest, start=1):
            if cancelled is not None and cancelled():
                return out
            out[i] = func(item)
            if on_step is not None:
                on_step(i + 1, len(items))
        return out

    try:
        pool = get_pool()
        futures = {pool.submit(func, item): i
                   for i, item in enumerate(rest, start=1)}
        done = 1
        for fut in _as_completed(futures, cancelled):
            out[futures[fut]] = fut.result()
            done += 1
            if on_step is not None:
                on_step(done, len(items))
    except Exception:
        # A pool that cannot start or has broken (sandbox, no spawn, a worker
        # killed) must not lose the sweep. Drop it so the next call starts a
        # fresh one, and finish here.
        shutdown()
        for i, item in enumerate(rest, start=1):
            if cancelled is not None and cancelled():
                return out
            if out[i] is None:
                out[i] = func(item)
    return out


def pnarrow(feasible_fn, bad, good, tol, cancelled=None, max_rounds=40):
    """Shrink a bracket to `tol` and return the `good` end. K points a round.

    A bisection is one evaluation per round, and here one evaluation is a whole
    column solve -- so the bracket is instead sampled at `max_workers()` interior
    points AT ONCE. A round then divides it by k+1 rather than by 2, which on 8
    cores turns 11 sequential solves into 4 parallel rounds. Same answer, same
    monotonicity assumption plain bisection already made.

    `bad` may lie either side of `good`: `feasible_band` refines its upper edge
    downward. The sample points run from `bad` towards `good` either way, so the
    first one that passes is the new `good` and its predecessor the new `bad`.

    Cheap problems keep the plain bisection. k points a round is k times the
    total work to save wall time, which is a bad trade when a solve is 14 ms and
    `pmap` would run them serially anyway -- so the first evaluation is timed and
    k drops to 1 below `SERIAL_BELOW`.
    """
    import time
    t0 = time.perf_counter()
    mid = 0.5 * (bad + good)
    first = feasible_fn(mid)
    k = max_workers() if (time.perf_counter() - t0) >= SERIAL_BELOW else 1
    if first:
        good = mid
    else:
        bad = mid

    for _ in range(max_rounds):
        if abs(good - bad) <= tol or (cancelled is not None and cancelled()):
            break
        xs = [bad + (good - bad) * (i + 1) / (k + 1) for i in range(k)]
        ok = pmap(feasible_fn, xs, cancelled=cancelled)
        if any(v is None for v in ok):
            break                            # cancelled mid-round
        hit = next((i for i, v in enumerate(ok) if v), None)
        if hit is None:
            bad = xs[-1]                     # the whole sample failed
        else:
            good = xs[hit]
            bad = xs[hit - 1] if hit else bad
    return good


def _as_completed(futures, cancelled):
    """`as_completed`, but abandoning the rest of the pool when asked to stop."""
    from concurrent.futures import as_completed
    for fut in as_completed(futures):
        if cancelled is not None and cancelled():
            for f in futures:
                f.cancel()
            return
        yield fut


def _picklable(obj):
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def _slow_square(x):
    """Module-level so the pool can import it (a lambda cannot be pickled).

    A pure-arithmetic loop on purpose: anything that dips into C (even
    `time.perf_counter`) drops the GIL and would make threads look like they
    scale here, which is exactly the illusion `_demo` exists to disprove.
    """
    total = 0
    for i in range(8_000_000):
        total += i & 7
    return x * x


def _demo():
    import time
    from concurrent.futures import ThreadPoolExecutor

    xs = list(range(6))
    assert pmap(_slow_square, []) == []
    assert pmap(abs, [-1, 2, -3]) == [1, 2, 3], "cheap work stays serial"

    seen = []
    got = pmap(_slow_square, xs, on_step=lambda d, t: seen.append(d))
    assert got == [x * x for x in xs], got
    assert seen[-1] == len(xs) and seen == sorted(seen), seen

    # a lambda cannot cross a process boundary; the answer must still be right
    assert pmap(lambda x: x + 1, [_Slow(), 1, 2])[1:] == [2, 3]

    # pnarrow must land on the same edge a plain bisection would, from either
    # direction, and must not step outside the bracket it was handed
    edge = pnarrow(lambda x: x >= 3.14159, 0.0, 10.0, 1e-4)
    assert 3.14159 <= edge <= 3.14159 + 1e-4, edge
    edge = pnarrow(lambda x: x <= 3.14159, 10.0, 0.0, 1e-4)   # upper edge, downward
    assert 3.14159 - 1e-4 <= edge <= 3.14159, edge

    if max_workers() >= 2:
        def timed(fn):
            t0 = time.perf_counter()
            fn()
            return time.perf_counter() - t0

        serial = timed(lambda: [_slow_square(x) for x in xs])
        par = timed(lambda: pmap(_slow_square, xs))
        thr = timed(lambda: list(ThreadPoolExecutor(max_workers=max_workers())
                                 .map(_slow_square, xs)))
        # The claim this module is built on. Only the first of these is asserted
        # tightly: a wall-clock ratio measured on a machine that is also running
        # something else is a flaky gate, and `thr > par` is the part that would
        # actually have to be false for threads to have been the right answer.
        assert par < 0.75 * serial, f"pool bought nothing: {par:.2f}s vs {serial:.2f}s"
        assert thr > par, f"threads beat the pool: {thr:.2f}s vs {par:.2f}s"
        print(f"parallel self-check OK  serial {serial:.1f}s  "
              f"processes {par:.1f}s  threads {thr:.1f}s  ({max_workers()} workers)")
    else:
        print("parallel self-check OK (single core: everything serial)")


class _Slow:
    """First item of the lambda test: makes `pmap` measure the work as slow, so
    it tries the pool and has to fall back rather than short-cutting to serial."""
    def __add__(self, other):
        import time
        time.sleep(SERIAL_BELOW * 1.5)
        return None


if __name__ == "__main__":
    _demo()
