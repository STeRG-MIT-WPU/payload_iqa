"""Pipeline orchestrator: wires camera -> Coral 1 -> Coral 2 as processes.

Both the headless CLI and the PyQt dashboard drive the pipeline through this
one class:

    pipe = Pipeline(Config(...))
    pipe.start()
    for ev in pipe.poll_events():   # call periodically
        ...
    pipe.stop()

Every event is a dict with a "type" key (capture, iqa, cloud, face, rejected,
passed, util, status, error, done). Final verdicts (passed/rejected) are also
appended to data/results.csv.
"""
import csv
import multiprocessing as mp
import queue
import time

from . import config
from .camera import capture_worker
from .workers import coral1_worker, coral2_worker

CSV_FIELDS = ["timestamp", "image", "result", "stage", "reason",
              "iqa", "cloud", "faces", "path"]


class Pipeline:
    def __init__(self, cfg=None):
        self.cfg = cfg or config.Config()
        self._procs = []
        self._events = None
        self._stop_evt = None

    # --- lifecycle ----------------------------------------------------------

    def start(self):
        if self.running:
            return
        self.cfg.ensure_dirs()
        self._init_csv()

        ctx = mp.get_context("spawn")   # same behaviour on Windows and the CM5
        self._stop_evt = ctx.Event()
        self._events = ctx.Queue()
        q_cap = ctx.Queue(maxsize=self.cfg.queue_size)
        q_face = ctx.Queue(maxsize=self.cfg.queue_size)

        self._procs = [
            ctx.Process(target=capture_worker, name="capture",
                        args=(self.cfg, q_cap, self._events, self._stop_evt),
                        daemon=True),
            ctx.Process(target=coral1_worker, name="coral1",
                        args=(self.cfg, q_cap, q_face, self._events,
                              self._stop_evt), daemon=True),
            ctx.Process(target=coral2_worker, name="coral2",
                        args=(self.cfg, q_face, self._events, self._stop_evt),
                        daemon=True),
        ]
        for p in self._procs:
            p.start()

    def stop(self, timeout=4.0):
        if self._stop_evt is not None:
            self._stop_evt.set()
        deadline = time.monotonic() + timeout
        for p in self._procs:
            p.join(max(0.1, deadline - time.monotonic()))
        for p in self._procs:
            if p.is_alive():
                p.terminate()
        self._procs = []

    @property
    def running(self):
        return any(p.is_alive() for p in self._procs)

    # --- events -------------------------------------------------------------

    def poll_events(self, max_events=500):
        """Drain pending events (non-blocking). Also logs verdicts to CSV."""
        out = []
        if self._events is None:
            return out
        for _ in range(max_events):
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                break
            self._record(ev)
            out.append(ev)
        return out

    # --- results.csv --------------------------------------------------------

    def _init_csv(self):
        if not config.RESULTS_CSV.exists():
            with open(config.RESULTS_CSV, "w", newline="") as f:
                csv.DictWriter(f, CSV_FIELDS).writeheader()

    def _record(self, ev):
        if ev.get("type") == "rejected":
            row = dict(result="rejected", stage=ev.get("stage"),
                       reason=ev.get("reason"), faces="")
        elif ev.get("type") == "passed":
            row = dict(result="passed", stage="face", reason="",
                       faces=ev.get("faces"))
        else:
            return
        row.update(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S",
                                    time.localtime(ev.get("t", time.time()))),
            image=ev.get("image"),
            iqa="" if ev.get("iqa") is None else f"{ev['iqa']:.2f}",
            cloud="" if ev.get("cloud") is None else f"{ev['cloud']:.1f}",
            path=ev.get("path"))
        try:
            with open(config.RESULTS_CSV, "a", newline="") as f:
                csv.DictWriter(f, CSV_FIELDS).writerow(row)
        except OSError:
            pass
