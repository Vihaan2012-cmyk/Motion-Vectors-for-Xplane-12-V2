import struct, sys
p = sys.argv[1]
d = open(p, 'rb').read()
assert d[:4] == b'MDMP'
nstreams, rva = struct.unpack_from('<II', d, 8)
streams = {}
for i in range(nstreams):
    t, sz, r = struct.unpack_from('<III', d, rva + i * 12); streams[t] = (sz, r)
mods = []
sz, mr = streams[4]; n = struct.unpack_from('<I', d, mr)[0]; off = mr + 4
for i in range(n):
    base, size, chk, tds, nrva = struct.unpack_from('<QIIII', d, off)
    L = struct.unpack_from('<I', d, nrva)[0]
    nm = d[nrva + 4:nrva + 4 + L].decode('utf-16le', 'replace').split(chr(92))[-1]
    mods.append((base, size, nm)); off += 108
def mod(a):
    for b, s, nm in mods:
        if b <= a < b + s: return nm, a - b
    return None
sz, er = streams[6]
tid = struct.unpack_from('<I', d, er)[0]
code, flags, rec, addr, nparams = struct.unpack_from('<IIQQI', d, er + 8)
info = struct.unpack_from('<15Q', d, er + 8 + 32)
print("thread %d exception 0x%08x at 0x%x %s params=%s" % (tid, code, addr, mod(addr), [hex(x) for x in info[:nparams]]))
csz, crva = struct.unpack_from('<II', d, er + 160)
rsp = struct.unpack_from('<Q', d, crva + 0x98)[0]; rip = struct.unpack_from('<Q', d, crva + 0xF8)[0]
rcx, rdx, r8, r9 = struct.unpack_from('<QQQQ', d, crva + 0x80)[0], struct.unpack_from('<Q', d, crva + 0x88)[0], struct.unpack_from('<Q', d, crva + 0xB8)[0], struct.unpack_from('<Q', d, crva + 0xC0)[0]
print("rip 0x%x %s rsp 0x%x rcx 0x%x rdx 0x%x r8 0x%x r9 0x%x" % (rip, mod(rip), rsp, rcx, rdx, r8, r9))
sz, tr = streams[3]; nt = struct.unpack_from('<I', d, tr)[0]
stack = None
for i in range(nt):
    t = tr + 4 + i * 48
    th = struct.unpack_from('<I', d, t)[0]
    if th == tid:
        sstart, ssize, srva = struct.unpack_from('<QII', d, t + 24)
        stack = (sstart, ssize, srva); break
print("stack range 0x%x size %d" % (stack[0], stack[1]))
sstart, ssize, srva = stack
hits = 0
for o in range(max(0, rsp - sstart), ssize - 8, 8):
    v = struct.unpack_from('<Q', d, srva + o)[0]
    m = mod(v)
    if m and m[1] > 0x1000:
        print("  rsp+0x%05x  %s+0x%x" % (o - (rsp - sstart), m[0], m[1]))
        hits += 1
        if hits >= 70: break
print("modules of interest:")
for b, s, nm in mods:
    if any(k in nm.lower() for k in ("vklayer", "vulkan", "vk_real", "nvoglv", "win.xpl", "x-plane", "ffx", "amd_fidelity")):
        print("  0x%x +0x%x %s" % (b, s, nm))
