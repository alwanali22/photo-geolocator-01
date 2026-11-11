"""Pure-stdlib JPEG DC-only decoder.

Decodes only the DC coefficient of each 8x8 block (the block average), which is
enough for colour/texture statistics and is ~50x faster than a full IDCT in
pure Python. Handles baseline DCT JPEGs (SOF0), 1-3 components, 4:4:4 / 4:2:2 /
4:2:0 subsampling, restart markers and byte stuffing. Progressive JPEGs
(SOF2) raise ValueError.
"""
import struct


class BitReader:
    def __init__(self, data, start):
        self.data = data
        self.pos = start
        self.acc = 0
        self.nbits = 0

    def _fill(self):
        while self.nbits <= 24:
            if self.pos >= len(self.data):
                self.acc = (self.acc << 8) | 0xFF
                self.nbits += 8
                continue
            b = self.data[self.pos]
            self.pos += 1
            if b == 0xFF:
                # byte stuffing or marker
                if self.pos < len(self.data):
                    nb = self.data[self.pos]
                    if nb == 0x00:
                        self.pos += 1
                    elif 0xD0 <= nb <= 0xD7:
                        # restart marker: realign to byte boundary
                        self.pos += 1
                        self.nbits = 0
                        self.acc = 0
                        return
                    else:
                        # other marker: stop feeding
                        self.pos -= 1
                        self.acc = (self.acc << 8) | 0xFF
                        self.nbits += 8
                        continue
                else:
                    self.acc = (self.acc << 8) | 0xFF
                    self.nbits += 8
                    continue
            self.acc = (self.acc << 8) | b
            self.nbits += 8

    def bits(self, n):
        if n == 0:
            return 0
        self._fill()
        while self.nbits < n:
            self._fill()
        self.nbits -= n
        v = (self.acc >> self.nbits) & ((1 << n) - 1)
        self.acc &= (1 << self.nbits) - 1
        return v

    def align(self):
        self.nbits = 0
        self.acc = 0


class HuffmanTable:
    def __init__(self, counts, symbols):
        self.map = {}
        code = 0
        k = 0
        for length in range(1, 17):
            for _ in range(counts[length - 1]):
                self.map[(code, length)] = symbols[k]
                k += 1
                code += 1
            code <<= 1

    def decode(self, br):
        code = 0
        for length in range(1, 17):
            code = (code << 1) | br.bits(1)
            sym = self.map.get((code, length))
            if sym is not None:
                return sym
        raise ValueError("bad huffman code")


def _extend(v, size):
    if size == 0:
        return 0
    if v < (1 << (size - 1)):
        return v - (1 << size) + 1
    return v


def decode_dc_grid(data):
    """Return (width, height, blocks_w, blocks_h, y_grid, cb_grid, cr_grid).

    Grids are per-8x8-block average values (0..255) at luma-block resolution;
    chroma grids are upsampled by nearest neighbour to the luma grid.
    y_grid is a list of rows of floats; cb/cr the same, or None for grayscale.
    """
    pos = 0
    if data[0:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    pos = 2
    qtab = {}
    huff_dc = {}
    huff_ac = {}
    width = height = 0
    comps = []  # (id, h, v, tq)
    restart_interval = 0
    while pos < len(data):
        if data[pos] != 0xFF:
            raise ValueError("bad marker at %d" % pos)
        marker = data[pos + 1]
        pos += 2
        if marker == 0xD9:
            break
        if marker == 0xDA:  # SOS
            length = struct.unpack(">H", data[pos:pos + 2])[0]
            ncomp = data[pos + 2]
            scan_comps = []
            for i in range(ncomp):
                cid = data[pos + 3 + 2 * i]
                tdta = data[pos + 4 + 2 * i]
                scan_comps.append((cid, tdta >> 4, tdta & 0xF))
            pos += length
            return _decode_scan(data, pos, comps, scan_comps, qtab, huff_dc,
                                huff_ac, width, height, restart_interval)
        if marker in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7):
            continue
        length = struct.unpack(">H", data[pos:pos + 2])[0]
        seg = data[pos + 2:pos + length]
        if marker == 0xDB:  # DQT
            p = 0
            while p < len(seg):
                pq = seg[p] >> 4
                tq = seg[p] & 0xF
                p += 1
                n = 64 * (2 if pq else 1)
                vals = list(struct.unpack(">" + ("H" * 64 if pq else "B" * 64),
                                          seg[p:p + n]))
                p += n
                # zigzag -> natural order
                zz = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
                      12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14,
                      21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
                      30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60,
                      61, 54, 47, 55, 62, 63]
                qtab[tq] = [vals[zz[i]] for i in range(64)]
        elif marker == 0xC0:  # SOF0
            prec = seg[0]
            height = struct.unpack(">H", seg[1:3])[0]
            width = struct.unpack(">H", seg[3:5])[0]
            ncomp = seg[5]
            comps = []
            for i in range(ncomp):
                cid = seg[6 + 3 * i]
                hv = seg[7 + 3 * i]
                tq = seg[8 + 3 * i]
                comps.append((cid, hv >> 4, hv & 0xF, tq))
            if prec != 8:
                raise ValueError("unsupported precision %d" % prec)
        elif marker == 0xC2:
            raise ValueError("progressive JPEG not supported")
        elif marker == 0xC4:  # DHT
            p = 0
            while p < len(seg):
                tc = seg[p] >> 4
                th = seg[p] & 0xF
                p += 1
                counts = list(seg[p:p + 16])
                p += 16
                total = sum(counts)
                symbols = list(seg[p:p + total])
                p += total
                tab = HuffmanTable(counts, symbols)
                if tc == 0:
                    huff_dc[th] = tab
                else:
                    huff_ac[th] = tab
        elif marker == 0xDD:  # DRI
            restart_interval = struct.unpack(">H", seg[0:2])[0]
        pos += length
    raise ValueError("no scan found")


def _decode_scan(data, start, comps, scan_comps, qtab, huff_dc, huff_ac,
                 width, height, restart_interval):
    max_h = max(c[1] for c in comps)
    max_v = max(c[2] for c in comps)
    mcu_w = max_h * 8
    mcu_h = max_v * 8
    mcus_x = (width + mcu_w - 1) // mcu_w
    mcus_y = (height + mcu_h - 1) // mcu_h
    # per-component DC grids at component-block resolution
    grids = {}
    for cid, h, v, tq in comps:
        bw = (width * h + max_h * 8 - 1) // (max_h * 8)
        bh = (height * v + max_v * 8 - 1) // (max_v * 8)
        grids[cid] = [[0.0] * bw for _ in range(bh)]
    comp_by_id = {c[0]: c for c in comps}
    scan = {cid: (huff_dc[td], huff_ac[ta], comp_by_id[cid])
            for cid, td, ta in scan_comps}
    br = BitReader(data, start)
    pred = {cid: 0 for cid in scan}
    mcu_count = 0
    for my in range(mcus_y):
        for mx in range(mcus_x):
            if restart_interval and mcu_count and mcu_count % restart_interval == 0:
                br.align()
                pred = {cid: 0 for cid in scan}
                # skip the RST marker bytes in the stream (only if actually present)
                while (br.pos + 1 < len(data) and data[br.pos] == 0xFF
                       and 0xD0 <= data[br.pos + 1] <= 0xD7):
                    br.pos += 2
                    break
            for cid, (dct, act, (ccid, h, v, tq)) in scan.items():
                q0 = qtab[tq][0]
                for by in range(v):
                    for bx in range(h):
                        cat = dct.decode(br)
                        diff = _extend(br.bits(cat), cat)
                        pred[cid] += diff
                        avg = pred[cid] * q0 / 8.0 + 128.0
                        gx = mx * h + bx
                        gy = my * v + by
                        g = grids[cid]
                        if gy < len(g) and gx < len(g[0]):
                            g[gy][gx] = avg
                        # skip AC coefficients: count COEFFICIENTS, not symbols.
                        # A block that exactly fills all 63 AC coefficients has
                        # no EOB marker; counting iterations over-consumes into
                        # the next block. Track coverage instead.
                        covered = 0
                        while covered < 63:
                            sym = act.decode(br)
                            run = sym >> 4
                            size = sym & 0xF
                            if size == 0:
                                if run == 15:
                                    covered += 16
                                    continue
                                break  # EOB
                            covered += run + 1
                            br.bits(size)
            mcu_count += 1
    # luma grid resolution
    y_id = comps[0][0]
    y_grid = grids[y_id]
    bw = len(y_grid[0])
    bh = len(y_grid)
    if len(comps) == 1:
        return width, height, bw, bh, y_grid, None, None

    def upsample(grid, ch, cv):
        # nearest-neighbour upsample chroma grid to luma-block grid
        out = []
        for gy in range(bh):
            sy = min(gy * cv // max_v, len(grid) - 1)
            row = []
            for gx in range(bw):
                sx = min(gx * ch // max_h, len(grid[0]) - 1)
                row.append(grid[sy][sx])
            out.append(row)
        return out

    cb = cr = None
    for cid, h, v, tq in comps[1:]:
        up = upsample(grids[cid], h, v)
        if cb is None:
            cb = up
        else:
            cr = up
    return width, height, bw, bh, y_grid, cb, cr


def ycbcr_to_rgb_grid(y, cb, cr):
    """Convert block grids to RGB rows (values 0..255, clamped)."""
    rows = []
    for yi, row in enumerate(y):
        out = []
        for xi, v in enumerate(row):
            cbb = (cb[yi][xi] - 128.0) if cb else 0.0
            crr = (cr[yi][xi] - 128.0) if cr else 0.0
            r = v + 1.402 * crr
            g = v - 0.34414 * cbb - 0.71414 * crr
            b = v + 1.772 * cbb
            out.append((max(0.0, min(255.0, r)),
                        max(0.0, min(255.0, g)),
                        max(0.0, min(255.0, b))))
        rows.append(out)
    return rows

# --- agent: model-free photo geolocator -------------------------------------
# Stdlib only. No model calls, no network except the trial env, no clock, no
# unseeded randomness: given the same photos it always posts the same answers.
import json
import os
import sys
import urllib.request
import urllib.error

# Rule centroids, fitted on the kit's 3 public practice trials (75 photos)
# by leave-one-trial-out validation: these three rules beat a fixed guess
# out-of-sample; every other candidate rule made it worse and was dropped.
DEFAULT = (55.0, 25.0)      # best single fixed guess for OpenStreetView-5M
SNOW = (53.0, 45.0)         # snow + bright sky -> Russian winter
TEMPGREEN = (43.0, -88.0)   # temperate green -> N. America interior
URBAN = (50.0, 77.0)        # grey asphalt/urban -> Central Asia


def photo_features(rgb, bw, bh):
    """Colour/texture fractions from the per-8x8-block RGB grid."""
    n = bw * bh
    mr = sum(p[0] for row in rgb for p in row) / n
    mg = sum(p[1] for row in rgb for p in row) / n
    mb = sum(p[2] for row in rgb for p in row) / n
    sky = veg = snow = sand = gray = water = 0
    top_rows = max(1, int(bh * 0.3))
    sky_top = 0
    for yi, row in enumerate(rgb):
        for (r, g, b) in row:
            if b > r + 12 and b > g + 6 and b > 90:
                sky += 1
                if yi < top_rows:
                    sky_top += 1
            if g > r + 10 and g > b + 10 and g > 60:
                veg += 1
            if r > 200 and g > 200 and b > 200 and (max(r, g, b) - min(r, g, b)) < 25:
                snow += 1
            if r > 140 and r > g + 15 and g > b + 5 and r < 230:
                sand += 1
            if (max(r, g, b) - min(r, g, b)) < 22 and 55 < r < 175:
                gray += 1
            if b > r + 25 and 70 < b < 200:
                water += 1
    bright_top = 0.0
    cnt = 0
    for yi in range(top_rows):
        for (r, g, b) in rgb[yi]:
            bright_top += (r + g + b) / 3.0
            cnt += 1
    bright_top = bright_top / cnt if cnt else 0.0
    return {
        "mean": (mr, mg, mb),
        "sky_top": sky_top / max(1, bw * top_rows),
        "veg": veg / n,
        "snow": snow / n,
        "sand": sand / n,
        "gray": gray / n,
        "water": water / n,
        "bright_top": bright_top,
    }


def classify(f):
    """(lat, lon) guess for one photo's features."""
    if f["snow"] > 0.32 and f["bright_top"] > 200:
        return SNOW
    if 0.20 <= f["veg"] <= 0.30:
        return TEMPGREEN
    if f["gray"] > 0.50:
        return URBAN
    return DEFAULT


def http_json(url, data=None, timeout=60):
    req = urllib.request.Request(url)
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_bytes(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def main():
    env = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env:
        print("MAKEFUN_ENV_URL is not set", file=sys.stderr)
        return 1
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    if prompt_file:
        try:
            with open(prompt_file, "r", encoding="utf-8") as fh:
                fh.read()
        except OSError:
            pass
    photos = http_json(env + "/photos")["photos"]
    answers = []
    for photo in photos:
        pid = photo["id"]
        guess = DEFAULT
        try:
            url = photo.get("url") or ("/photos/%s.jpg" % pid)
            if url.startswith("http"):
                raw = http_bytes(url)
            else:
                raw = http_bytes(env + url)
            w, h, bw, bh, y, cb, cr = decode_dc_grid(raw)
            rgb = ycbcr_to_rgb_grid(y, cb, cr)
            guess = classify(photo_features(rgb, bw, bh))
        except Exception as exc:  # never leave a photo unanswered
            print("photo %s: %s; using default guess" % (pid, exc),
                  file=sys.stderr)
            guess = DEFAULT
        answers.append({"id": pid,
                        "lat": round(guess[0], 6),
                        "lon": round(guess[1], 6)})
    # Post in one batch; on a whole-batch refusal, retry per photo so a
    # single bad answer cannot block the rest.
    try:
        res = http_json(env + "/answers", {"answers": answers})
        print("posted %d answers, accepted %s" % (len(answers), res.get("accepted")),
              file=sys.stderr)
    except urllib.error.HTTPError:
        posted = 0
        for ans in answers:
            try:
                http_json(env + "/answers", {"answers": [ans]})
                posted += 1
            except Exception as exc:
                print("photo %s: post failed: %s" % (ans["id"], exc),
                      file=sys.stderr)
        print("posted %d/%d answers one by one" % (posted, len(answers)),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
