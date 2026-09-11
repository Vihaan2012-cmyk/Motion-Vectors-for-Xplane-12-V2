"""watch.py - a terminal to stare at while a run trains.

Refreshes every 2 s: GPU load, the last loss lines, the latest quiz table and
the round summaries. Ctrl+C to leave; the run is untouched.

    python tools/nn/watch.py                       # newest log in tools/nn/ckpt
    python tools/nn/watch.py tools/nn/ckpt/train_temporal.log
"""
import glob, os, subprocess, sys, time

here = os.path.dirname(os.path.abspath(__file__))
log = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob(os.path.join(here, "ckpt", "train*.log")), key=os.path.getmtime)

def gpu():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader"], text=True, timeout=3).strip()
        u, m, t = [x.strip() for x in o.split(",")]
        return "GPU %s  %s  %s C" % (u, m, t)
    except Exception:
        return "GPU ?"

while True:
    try:
        lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
    except FileNotFoundError:
        lines = []
    steps = [l for l in lines if " step " in l and " loss " in l]
    rounds = [l for l in lines if l.startswith("round ") and ": train L1" in l]
    quiz_at = max((i for i, l in enumerate(lines) if l.startswith("QUIZ")), default=None)
    quiz = lines[quiz_at:quiz_at + 8] if quiz_at is not None else ["(no quiz yet)"]
    final = [l for l in lines if l.startswith(("PERFECT", "MEMORIS", "Traceback"))]
    os.system("cls" if os.name == "nt" else "clear")
    print("%s   %s   %s" % (time.strftime("%H:%M:%S"), gpu(), os.path.basename(log)))
    print("-" * 78)
    for l in steps[-6:]: print(l)
    print("-" * 78)
    for l in quiz: print(l)
    print("-" * 78)
    for l in rounds[-6:]: print(l)
    for l in final: print(l)
    time.sleep(2)
