"""Photo geolocator entry (model-free).

Reads 25 street-level photos from the competition environment, estimates
where each one was taken from simple colour and texture cues, and posts one
latitude/longitude guess per photo. Pure Python standard library, no model
calls, no network access except to the environment itself.

Method in brief: each JPEG is decoded only as far as the DC coefficient of
every 8x8 block, which is the block's average colour. From that coarse grid
we measure four picture-level features (snow cover, sky brightness, green
cover, grey cover) and apply a small fixed rule set. The rules were picked
by checking them against the practice trials and keeping only the ones that
helped out of sample.
"""

import json
import os
import urllib.request

# Fixed guesses used by the rule set (latitude, longitude).
GUESS_RUSSIAN_WINTER = (53.0, 45.0)
GUESS_NORTH_AMERICA = (43.0, -88.0)
GUESS_CENTRAL_ASIA = (50.0, 77.0)
GUESS_FALLBACK = (55.0, 25.0)


class BitReader:
    """Reads bits from a cleaned JPEG entropy stream.

    The stream is cleaned before reading: stuffed 0x00 bytes after 0xFF are
    removed, and restart markers are recorded as byte positions so the
    decoder can jump to them and drop any padding bits.
    """

    def __init__(self, data, restarts):
        self.data = data
        self.pos = 0
        self.buf = 0
        self.nbits = 0
        self.restarts = list(restarts)
        self.next_restart = 0

    def _fill(self):
        # Top the buffer up with up to 4 bytes at a time. Reading a few
        # bytes past a restart marker is harmless: the bits are only
        # consumed while decoding, and align_restart discards the buffer
        # and rewinds the position to the marker before any of them are used.
        count = min(4, len(self.data) - self.pos)
        for _ in range(count):
            self.buf = (self.buf << 8) | self.data[self.pos]
            self.pos += 1
        self.nbits += 8 * count
        if self.nbits:
            self.buf &= (1 << self.nbits) - 1

    def bit(self):
        if self.nbits == 0:
            self._fill()
        self.nbits -= 1
        return (self.buf >> self.nbits) & 1

    def bits(self, count):
        if count == 0:
            return 0
        while self.nbits < count and self.pos < len(self.data):
            self._fill()
        if self.nbits >= count:
            self.nbits -= count
            value = (self.buf >> self.nbits) & ((1 << count) - 1)
            self.buf &= (1 << self.nbits) - 1 if self.nbits else 0
            return value
        # Ran out of data: pad the missing low bits with zeros.
        value = (self.buf << (count - self.nbits)) & ((1 << count) - 1)
        self.buf = 0
        self.nbits = 0
        return value

    def align_restart(self):
        """Drop buffered padding bits and jump to the next restart marker."""
        self.buf = 0
        self.nbits = 0
        if self.next_restart < len(self.restarts):
            self.pos = self.restarts[self.next_restart]
            self.next_restart += 1


def build_huffman(counts, symbols):
    """Build Huffman lookup structures from JPEG DHT data.

    Returns (table, lut): `table` maps (code, length) to a symbol for the
    slow path, and `lut` maps every 8-bit prefix to (symbol, length) for
    codes of 8 bits or fewer, which covers nearly every code in practice.
    """
    table = {}
    lut = [None] * 256
    code = 0
    index = 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            symbol = symbols[index]
            table[(code, length)] = symbol
            if length <= 8:
                prefix = code << (8 - length)
                step = 1 << (8 - length)
                entry = (symbol, length)
                for peek in range(prefix, prefix + step):
                    lut[peek] = entry
            index += 1
            code += 1
        code <<= 1
    return table, lut


def huffman_decode(reader, table, lut):
    """Decode one symbol, using the 8-bit prefix table when possible."""
    if reader.nbits < 8:
        reader._fill()
    if reader.nbits >= 8:
        hit = lut[(reader.buf >> (reader.nbits - 8)) & 0xFF]
        if hit is not None:
            symbol, length = hit
            reader.nbits -= length
            reader.buf &= (1 << reader.nbits) - 1 if reader.nbits else 0
            return symbol
    # Slow path: codes longer than 8 bits, one bit at a time.
    code = 0
    for length in range(1, 17):
        code = (code << 1) | reader.bit()
        symbol = table.get((code, length))
        if symbol is not None:
            return symbol
    raise ValueError("bad huffman code in entropy data")


def extend_value(bits, size):
    """Undo JPEG's magnitude coding for a coefficient of `size` bits."""
    if size == 0:
        return 0
    if bits < (1 << (size - 1)):
        return bits - ((1 << size) - 1)
    return bits


def parse_segments(data):
    """Walk the JPEG markers and collect tables, frame info and scan start.

    Returns (info, entropy_offset) where info holds the image size, the
    per-component sampling and quant tables, the Huffman tables, the
    restart interval and the scan's per-component table choices.
    """
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG file")
    info = {
        "quant": {},
        "dc_tables": {},
        "ac_tables": {},
        "restart_interval": 0,
    }
    pos = 2
    while pos + 1 < len(data):
        if data[pos] != 0xFF:
            raise ValueError("expected JPEG marker at offset %d" % pos)
        marker = data[pos + 1]
        if marker == 0xD8:  # SOI, no payload
            pos += 2
            continue
        if marker == 0xD9:  # EOI
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        length = (data[pos + 2] << 8) | data[pos + 3]
        body = data[pos + 4:pos + 2 + length]
        if marker == 0xDB:  # DQT: quantisation tables
            i = 0
            while i < len(body):
                precision = body[i] >> 4
                table_id = body[i] & 0x0F
                i += 1
                values = []
                for _ in range(64):
                    if precision:
                        values.append((body[i] << 8) | body[i + 1])
                        i += 2
                    else:
                        values.append(body[i])
                        i += 1
                # Only the DC entry (zigzag position 0) is ever used here.
                info["quant"][table_id] = values[0]
        elif marker == 0xC4:  # DHT: Huffman tables
            i = 0
            while i < len(body):
                table_class = body[i] >> 4
                table_id = body[i] & 0x0F
                counts = list(body[i + 1:i + 17])
                total = sum(counts)
                symbols = list(body[i + 17:i + 17 + total])
                table = build_huffman(counts, symbols)
                if table_class == 0:
                    info["dc_tables"][table_id] = table
                else:
                    info["ac_tables"][table_id] = table
                i += 17 + total
        elif marker == 0xC0:  # SOF0: baseline frame
            height = (body[1] << 8) | body[2]
            width = (body[3] << 8) | body[4]
            ncomp = body[5]
            components = []
            for k in range(ncomp):
                off = 6 + 3 * k
                comp_id = body[off]
                sampling = body[off + 1]
                quant_id = body[off + 2]
                components.append({
                    "id": comp_id,
                    "h": sampling >> 4,
                    "v": sampling & 0x0F,
                    "tq": quant_id,
                })
            info["width"] = width
            info["height"] = height
            info["components"] = components
        elif marker == 0xC2:  # SOF2: progressive, not supported
            raise ValueError("progressive JPEG not supported")
        elif marker == 0xDD:  # DRI: restart interval
            info["restart_interval"] = (body[0] << 8) | body[1]
        elif marker == 0xDA:  # SOS: start of scan, entropy follows
            ncomp = body[0]
            scan = {}
            for k in range(ncomp):
                off = 1 + 2 * k
                comp_id = body[off]
                tables = body[off + 1]
                scan[comp_id] = {
                    "td": tables >> 4,
                    "ta": tables & 0x0F,
                }
            info["scan"] = scan
            return info, pos + 2 + length
        pos += 2 + length
    raise ValueError("no scan found in JPEG")


def clean_entropy(raw):
    """Strip byte stuffing from entropy data and note restart markers.

    Returns (clean_bytes, restart_positions): restart_positions holds the
    clean-stream offset of the first data byte after each restart marker.
    """
    out = bytearray()
    restarts = []
    i = 0
    n = len(raw)
    while i < n:
        byte = raw[i]
        if byte == 0xFF:
            if i + 1 >= n:
                break
            nxt = raw[i + 1]
            if nxt == 0x00:
                out.append(0xFF)
                i += 2
            elif 0xD0 <= nxt <= 0xD7:
                restarts.append(len(out))
                i += 2
            elif nxt == 0xD9:
                break
            else:
                break
        else:
            out.append(byte)
            i += 1
    return bytes(out), restarts


def decode_block(reader, info, comp_index, predictors):
    """Decode one 8x8 block, returning its dequantised DC coefficient.

    Only the DC value is kept. The AC coefficients are skipped, but the
    skip has to count coefficients, not symbols: a run-length code of 15
    zeros plus a coefficient covers 16 coefficients, and an end-of-block
    code ends the block. A block that uses all 63 AC coefficients with no
    end-of-block must stop exactly at 63 or it eats the next block.
    """
    comp = info["components"][comp_index]
    dc_table, dc_lut = info["dc_tables"][info["scan"][comp["id"]]["td"]]
    ac_table, ac_lut = info["ac_tables"][info["scan"][comp["id"]]["ta"]]
    size = huffman_decode(reader, dc_table, dc_lut)
    diff = extend_value(reader.bits(size), size)
    predictors[comp_index] += diff
    dc = predictors[comp_index] * info["quant"][comp["tq"]]
    covered = 0
    while covered < 63:
        symbol = huffman_decode(reader, ac_table, ac_lut)
        if symbol == 0x00:  # end of block
            break
        run = symbol >> 4
        size = symbol & 0x0F
        if run == 15 and size == 0:  # ZRL: 15 zeros plus a coefficient
            covered += 16
        else:
            covered += run + (1 if size else 0)
            if size:
                reader.bits(size)
    return dc


def decode_dc_grids(data):
    """Decode a JPEG into one DC grid per component.

    Each grid entry is the dequantised DC coefficient of one 8x8 block.
    The block's average pixel value is dc / 16 + 128.
    """
    info, entropy_start = parse_segments(data)
    clean, restarts = clean_entropy(data[entropy_start:])
    reader = BitReader(clean, restarts)
    comps = info["components"]
    h_max = max(c["h"] for c in comps)
    v_max = max(c["v"] for c in comps)
    mcu_x = (info["width"] + 8 * h_max - 1) // (8 * h_max)
    mcu_y = (info["height"] + 8 * v_max - 1) // (8 * v_max)
    grids = []
    for comp in comps:
        grid_w = (info["width"] * comp["h"] + 8 * h_max - 1) // (8 * h_max)
        grid_h = (info["height"] * comp["v"] + 8 * v_max - 1) // (8 * v_max)
        grids.append([[0] * grid_w for _ in range(grid_h)])
    predictors = [0] * len(comps)
    interval = info["restart_interval"]
    done = 0
    for my in range(mcu_y):
        for mx in range(mcu_x):
            for ci, comp in enumerate(comps):
                for by in range(comp["v"]):
                    for bx in range(comp["h"]):
                        gy = my * comp["v"] + by
                        gx = mx * comp["h"] + bx
                        value = decode_block(reader, info, ci, predictors)
                        if gy < len(grids[ci]) and gx < len(grids[ci][0]):
                            grids[ci][gy][gx] = value
            done += 1
            if interval and done % interval == 0:
                reader.align_restart()
                predictors = [0] * len(comps)
    return info, grids


def block_average(dc):
    """Average pixel value of a block from its dequantised DC coefficient.

    The JPEG DCT has a 1/sqrt(2) factor on the zero frequency, so the DC
    coefficient is 8 times the block mean of the level-shifted pixels:
    mean = dc / 8 + 128.
    """
    return dc / 8.0 + 128.0


def ycbcr_to_rgb(y, cb, cr):
    """Convert one block-average YCbCr triple to RGB, clamped to 0..255."""
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.34414 * (cb - 128.0) - 0.71414 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    if r < 0.0:
        r = 0.0
    elif r > 255.0:
        r = 255.0
    if g < 0.0:
        g = 0.0
    elif g > 255.0:
        g = 255.0
    if b < 0.0:
        b = 0.0
    elif b > 255.0:
        b = 255.0
    return r, g, b


def photo_features(info, grids):
    """Measure snow, sky brightness, vegetation and grey cover.

    Returns (snow, bright_top, veg, gray):
      snow       -- share of blocks that are near-white on all channels
      bright_top -- mean brightness of the top quarter of block rows
      veg        -- share of blocks where green clearly dominates
      gray       -- share of blocks that are low-saturation mid-grey
    """
    y_grid = grids[0]
    rows = len(y_grid)
    cols = len(y_grid[0])
    comps = info["components"]
    has_colour = len(grids) >= 3
    snow = 0
    veg = 0
    gray = 0
    top_total = 0.0
    top_count = 0
    top_rows = max(1, rows // 4)
    total = 0
    for by in range(rows):
        for bx in range(cols):
            y = block_average(y_grid[by][bx])
            if has_colour:
                # Chroma blocks cover several luma blocks when subsampled;
                # map each luma block to the chroma block that contains it.
                cb_comp = comps[1]
                cr_comp = comps[2]
                cb_x = bx * cb_comp["h"] // comps[0]["h"]
                cb_y = by * cb_comp["v"] // comps[0]["v"]
                cr_x = bx * cr_comp["h"] // comps[0]["h"]
                cr_y = by * cr_comp["v"] // comps[0]["v"]
                cb = block_average(grids[1][cb_y][cb_x])
                cr = block_average(grids[2][cr_y][cr_x])
                r, g, b = ycbcr_to_rgb(y, cb, cr)
            else:
                r = g = b = y
            bright = (r + g + b) / 3.0
            if r > 200.0 and g > 200.0 and b > 200.0:
                snow += 1
            if g > r + 12.0 and g > b + 12.0 and g > 60.0:
                veg += 1
            spread = max(r, g, b) - min(r, g, b)
            if spread < 20.0 and 60.0 <= bright <= 180.0:
                gray += 1
            if by < top_rows:
                top_total += bright
                top_count += 1
            total += 1
    return (
        snow / total,
        top_total / top_count if top_count else 0.0,
        veg / total,
        gray / total,
    )


def guess_location(snow, bright_top, veg, gray):
    """Apply the validated rule set, in order, and return (lat, lon)."""
    if snow > 0.32 and bright_top > 200.0:
        return GUESS_RUSSIAN_WINTER
    if 0.20 <= veg <= 0.30:
        return GUESS_NORTH_AMERICA
    if gray > 0.50:
        return GUESS_CENTRAL_ASIA
    return GUESS_FALLBACK


def guess_photo(jpeg_bytes):
    """Decode one photo and guess where it was taken.

    Any decode failure falls back to the single best fixed guess so the
    photo still gets an answer instead of breaking the whole trial.
    """
    try:
        info, grids = decode_dc_grids(jpeg_bytes)
        snow, bright_top, veg, gray = photo_features(info, grids)
        return guess_location(snow, bright_top, veg, gray)
    except Exception:
        return GUESS_FALLBACK


def fetch_json(url):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def fetch_bytes(url):
    with urllib.request.urlopen(url) as response:
        return response.read()


def main():
    base = os.environ.get("MAKEFUN_ENV_URL", "http://127.0.0.1:8765")
    base = base.rstrip("/")
    listing = fetch_json(base + "/photos")
    answers = []
    for photo in listing["photos"]:
        url = photo["url"]
        if url.startswith("/"):
            url = base + url
        elif not url.startswith("http"):
            url = base + "/" + url
        lat, lon = guess_photo(fetch_bytes(url))
        answers.append({"id": photo["id"], "lat": lat, "lon": lon})
    payload = json.dumps({"answers": answers}).encode("utf-8")
    request = urllib.request.Request(
        base + "/answers",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
