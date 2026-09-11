import subprocess, sys, re, bisect
dll = sys.argv[1]; offs = [int(x, 16) for x in sys.argv[2:]]
p = subprocess.run(["objdump", "-p", dll], capture_output=True, text=True).stdout
base = int(re.search(r"ImageBase\s+([0-9a-fA-F]+)", p).group(1), 16)
print("%s ImageBase 0x%x SizeOfImage %s TimeDateStamp %s" % (dll.split("/")[-1], base, re.search(r"SizeOfImage\s+([0-9a-fA-F]+)", p).group(1), (re.search(r"Time/Date\s+(.*)", p) or [None, "?"])[1] if False else re.search(r"Time/Date\s+(.*)", p).group(1).strip()))
syms = []
nm = subprocess.run(["nm", "-n", "--demangle", dll], capture_output=True, text=True).stdout
for line in nm.splitlines():
    parts = line.split(" ", 2)
    if len(parts) == 3 and parts[1] in "tTwW":
        try: syms.append((int(parts[0], 16), parts[2]))
        except ValueError: pass
if not syms:   # fall back to the export table
    sec = p.split("[Ordinal/Name Pointer] Table")[1] if "[Ordinal/Name Pointer] Table" in p else ""
    for m in re.finditer(r"\[\s*\d+\]\s+([0-9a-fA-F]+)\s+(\S+)", sec):
        syms.append((base + int(m.group(1), 16), m.group(2)))
    syms.sort()
addrs = [s[0] for s in syms]
for o in offs:
    v = base + o; i = bisect.bisect_right(addrs, v) - 1
    print("  +0x%x  %s+0x%x" % (o, syms[i][1][:110], v - syms[i][0]) if i >= 0 else "  +0x%x ?" % o)
