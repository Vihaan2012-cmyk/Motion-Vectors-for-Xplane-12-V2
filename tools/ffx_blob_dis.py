# Extract the first SPIR-V blob from a generated FFX permutation header (tools/ffx_permute.cpp
# output) into a .spv file, so spirv-dis can compare two generations of the same pass.
#   python tools/ffx_blob_dis.py <header.h> <out.spv>
import re, sys
hdr, out = sys.argv[1], sys.argv[2]
s = open(hdr, encoding="utf-8").read()
m = re.search(r"static const unsigned char g_\w+_data\[\] = \{(.*?)\};", s, re.S)
if not m:
    sys.exit("no g_*_data[] array in " + hdr)
data = bytes(int(x, 0) for x in re.findall(r"0x[0-9a-fA-F]{1,2}|\b\d+\b", m.group(1)))
open(out, "wb").write(data)
print(out, len(data), "bytes")
