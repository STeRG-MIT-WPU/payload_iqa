#!/usr/bin/env python3
"""Run the full two-Coral pipeline WITHOUT any GUI.

Examples:
    # real thing on the CM5: USB camera + both Corals
    python3 run_headless.py

    # dev machine: feed the bundled test images through CPU models
    python3 run_headless.py --source folder --cpu --interval 0.5

    # camera but no Corals attached
    python3 run_headless.py --cpu

    # run for 60 seconds then stop
    python3 run_headless.py --duration 60

Folder mode exits by itself once every image has been processed; camera mode
runs until Ctrl+C (or --duration). Individual models can be tested with
`python -m pipeline.engines {iqa,cloud,face,all} --image PATH`.
"""
import argparse
import sys
import time

from pipeline.config import Config
from pipeline.monitor import ResourceMonitor
from pipeline.orchestrator import Pipeline


class Counters:
    def __init__(self):
        self.captured = 0
        self.iqa_rejected = 0
        self.cloud_rejected = 0
        self.passed = 0
        self.faces = 0


def handle(ev, c):
    t = ev.get("type")
    if t == "capture":
        c.captured += 1
        print(f"[CAPTURE] {ev['image']}")
    elif t == "iqa":
        verdict = "ok" if ev["ok"] else "REJECT"
        print(f"[IQA    ] {ev['image']}: score {ev['score']:.2f} "
              f"({ev['ms']:.0f} ms) -> {verdict}")
    elif t == "cloud":
        print(f"[CLOUD  ] {ev['image']}: coverage {ev['coverage']:.1f}% "
              f"({ev['ms']:.0f} ms) -> {ev['action']}")
    elif t == "face":
        print(f"[FACE   ] {ev['image']}: {ev['count']} face(s) "
              f"({ev['ms']:.0f} ms)")
        c.faces += ev["count"]
    elif t == "rejected":
        if ev["stage"] == "iqa":
            c.iqa_rejected += 1
        else:
            c.cloud_rejected += 1
        print(f"[REJECT ] {ev['image']}: {ev['reason']}  -> {ev['path']}")
    elif t == "passed":
        c.passed += 1
        print(f"[PASS   ] {ev['image']}: IQA {ev['iqa']:.1f}, "
              f"cloud {ev['cloud']:.1f}%, {ev['faces']} face(s)"
              + (" [cropped]" if ev.get("cropped") else "")
              + f"  -> {ev['path']}")
    elif t == "util":
        pass  # folded into the periodic resource line
    elif t == "status":
        print(f"[status ] {ev['msg']}")
    elif t == "error":
        print(f"[ERROR  ] ({ev.get('stage', '?')}) {ev['msg']}", file=sys.stderr)
    return t


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["camera", "folder"], default="camera")
    ap.add_argument("--input", default=None, help="image folder for --source folder")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between captures (default 2)")
    ap.add_argument("--loop", action="store_true",
                    help="folder mode: keep re-feeding the images forever")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU models (no Edge TPUs)")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after this many seconds")
    ap.add_argument("--iqa-min", type=float, default=60.0)
    ap.add_argument("--cloud-reject", type=float, default=70.0)
    ap.add_argument("--cloud-crop", type=float, default=60.0)
    ap.add_argument("--face-threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = Config(source=args.source, camera_index=args.camera_index,
                 capture_interval=args.interval, loop_folder=args.loop,
                 use_edgetpu=not args.cpu, iqa_min=args.iqa_min,
                 cloud_reject=args.cloud_reject, cloud_crop=args.cloud_crop,
                 face_threshold=args.face_threshold)
    if args.input:
        cfg.input_folder = args.input

    pipe = Pipeline(cfg)
    mon = ResourceMonitor()
    counters = Counters()
    util = {}

    pipe.start()
    print(f"pipeline started (source={cfg.source}, "
          f"{'Edge TPU' if cfg.use_edgetpu else 'CPU'}). Ctrl+C to stop.\n")

    t_start = time.monotonic()
    next_res = t_start + 5.0
    try:
        done = False
        while not done:
            for ev in pipe.poll_events():
                if ev.get("type") == "util":
                    util[ev["tpu"]] = ev
                elif handle(ev, counters) == "done":
                    done = True
            if args.duration and time.monotonic() - t_start >= args.duration:
                break
            if time.monotonic() >= next_res:
                next_res = time.monotonic() + 5.0
                s = mon.sample()
                parts = []
                if s:
                    parts.append(f"CPU {s['cpu']:.0f}%  RAM {s['ram']:.0f}% "
                                 f"({s['ram_used_gb']:.1f}/{s['ram_total_gb']:.1f} GB)")
                for name in ("coral1", "coral2"):
                    if name in util:
                        u = util[name]
                        parts.append(f"{name}[{u['device']}] {u['duty']:.0f}%")
                if parts:
                    print(f"[resmon ] {'  |  '.join(parts)}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        pipe.stop()
        for ev in pipe.poll_events():   # drain stragglers
            if ev.get("type") not in ("util", "status"):
                handle(ev, counters)

    c = counters
    print(f"\nsummary: {c.captured} captured | "
          f"{c.iqa_rejected} rejected by IQA | "
          f"{c.cloud_rejected} rejected by cloud | "
          f"{c.passed} passed | {c.faces} faces found")
    print("results: data/passed, data/rejected, data/results.csv")


if __name__ == "__main__":
    main()
