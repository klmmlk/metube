"""Pure-Python decryptor for CCTV hls_h5e (proprietary NAL-level encryption).

Faithful port of CCTVVideoDownloader's
``src/core/src/crypto/cctv_h5e_decrypt.hpp`` (single-header C++, research
documentation) so metube needs no native build step. Keep the function and
constant names mirroring the hpp -- future upstream fixes to the algorithm
should diff cleanly against this file.

What the encryption is: the h5e TS segments are structurally ordinary
MPEG-TS / H.264, but inside each video NAL a sparse grid of small blocks is
transformed (type 1/5) or TEA-encrypted (type 5 "new mode", keyed from bytes
embedded in the NAL itself). A type-25 signaling NAL switches the stream into
new mode. Decryption therefore:

* walks the TS packets of the video PID, reassembles each PES,
* splits the ES into NAL units and runs the per-type grid transform,
* drops the emulation-prevention 0x03 bytes the encrypted payload hid behind
  (RBSP-indexed grid -- cells straddling an EPB read/write through an
  RBSP->EBSP position map), and
* shrinks the PES back into its TS packets by growing adaptation-field
  stuffing, so packet count and timing are untouched.

Classic (pre-type25) streams instead TEA-decrypt blocks at key@16, start=32,
stride=80.

The transforms are cheap (well under 5% of segment bytes are touched), so a
CPython implementation keeps up with realtime playback easily.
"""

import struct
from typing import Optional

__all__ = [
    'H5eSession', 'decrypt_ts', 'decrypt_ts_inplace', 'detect_video_pid',
    'tea_decrypt_block', 'tea_encrypt_block',
]

_DELTA = 0x9E3779B9


# ===== TEA (type5 / classic) =====

def tea_encrypt_block(block: bytearray, key: bytes) -> None:
    """TEA-16 encrypt one 8-byte block in place (little-endian words).

    Only used by the tests (round-trip vectors); the decryptor never encrypts.
    """
    v0, v1 = struct.unpack_from('<II', block, 0)
    k0, k1, k2, k3 = struct.unpack_from('<IIII', key, 0)
    sum_ = 0
    for _ in range(16):
        sum_ = (sum_ + _DELTA) & 0xFFFFFFFF
        v0 = (v0 + ((((v1 << 4) + k0) ^ (v1 + sum_) ^ ((v1 >> 5) + k1)))) & 0xFFFFFFFF
        v1 = (v1 + ((((v0 << 4) + k2) ^ (v0 + sum_) ^ ((v0 >> 5) + k3)))) & 0xFFFFFFFF
    struct.pack_into('<II', block, 0, v0, v1)


def tea_decrypt_block(block: bytearray, key: bytes) -> None:
    """TEA-16 decrypt one 8-byte block in place (little-endian words)."""
    v0, v1 = struct.unpack_from('<II', block, 0)
    k0, k1, k2, k3 = struct.unpack_from('<IIII', key, 0)
    sum_ = (_DELTA * 16) & 0xFFFFFFFF
    for _ in range(16):
        v1 = (v1 - ((((v0 << 4) + k2) ^ (v0 + sum_) ^ ((v0 >> 5) + k3)))) & 0xFFFFFFFF
        v0 = (v0 - ((((v1 << 4) + k0) ^ (v1 + sum_) ^ ((v1 >> 5) + k1)))) & 0xFFFFFFFF
        sum_ = (sum_ - _DELTA) & 0xFFFFFFFF
    struct.pack_into('<II', block, 0, v0, v1)


def decrypt_classic(nal: bytearray) -> None:
    """Classic layout (no type25): key@16, start=32, stride=80, in place."""
    ln = len(nal)
    if ln < 40:
        return
    key = bytes(nal[16:32])
    j = 0
    while 32 + j * 80 + 8 <= ln:
        o = 32 + j * 80
        blk = bytearray(nal[o:o + 8])
        tea_decrypt_block(blk, key)
        nal[o:o + 8] = blk
        j += 1


# ===== EPB (emulation prevention) helpers =====

def _scan_epbs(nal) -> list:
    """Start positions of every 00 00 03 triple, non-overlapping (hpp loop:
    on a match the scan resumes three bytes later)."""
    epbs = []
    start = 0
    while True:
        p = nal.find(b'\x00\x00\x03', start)
        if p < 0:
            return epbs
        epbs.append(p)
        start = p + 3


def _build_r2e(ln: int, epbs) -> list:
    """RBSP -> EBSP byte-position map: every byte except each EPB's 0x03."""
    r2e = []
    prev = 0
    for e in epbs:
        r2e.extend(range(prev, e))
        r2e.append(e)
        r2e.append(e + 1)
        prev = e + 3
    r2e.extend(range(prev, ln))
    return r2e


def _drop_epb_03(nal: bytearray, epbs) -> int:
    """Delete the 0x03 of each still-present EPB (from the end). New length."""
    nlen = len(nal)
    for e in reversed(epbs):
        if e + 2 < nlen and nal[e] == 0 and nal[e + 1] == 0 and nal[e + 2] == 3:
            del nal[e + 2]
            nlen -= 1
    return nlen


def is_type25_enable(nal, off: int = 0) -> bool:
    return (len(nal) - off >= 4
            and (nal[off] & 0x1F) == 25
            and nal[off + 2] == 0x01
            and nal[off + 3] == 0x09)


# ===== strides =====

_TYPE5_F5_BASE = (160, 192, 224, 256, 288, 320)


def type5_stride_f5(key16: bytes) -> int:
    le = int.from_bytes(key16[0:4], 'little')
    idx = le % 6
    return _TYPE5_F5_BASE[idx] | key16[idx]


def type5_stride_from_nal(nal) -> int:
    """Stride from a type5 NAL: key bytes nal[5..10] (hpp type5_stride_from_nal)."""
    if len(nal) < 11:
        return 0
    return type5_stride_f5(bytes(nal[5:11]))


def type1_stride_f1(nal) -> int:
    """Same BASE/select as F5 but keyed at nal[1:7]."""
    if len(nal) < 7:
        return 0
    return type5_stride_f5(bytes(nal[1:7]))


# ===== type1 G transform =====

def _type1_fbit(W: int) -> int:
    w0 = W & 1
    w8 = (W >> 8) & 1
    w15 = (W >> 15) & 1
    w19 = (W >> 19) & 1
    w25 = (W >> 25) & 1
    w30 = (W >> 30) & 1
    w31 = (W >> 31) & 1
    t = w0 | w8
    return (w31 ^ w15 ^ t
            ^ (w8 & w19)
            ^ (w25 & (w0 ^ w19))
            ^ (w0 & (1 ^ w8) & w30)
            ^ ((1 ^ w0) & w19 & w30)
            ^ (w25 & w30 & (w8 ^ w19))) & 1


_TYPE1_B_STEPS = frozenset((2, 8, 9, 10))


def type1_flip_mask_from_header(hdr) -> int:
    """Bitmask of steps to invert Fbit, derived from the 3 NAL header bytes
    (hpp type1_flip_mask_from_header; multi-header families)."""
    b0, b1, b2 = hdr[0], hdr[1], hdr[2]
    m = 0
    if b0 == 0x01 and b1 == 0xA8:
        if (b2 >> 7) & 1:
            m |= 1 << 0
        if (b2 >> 6) & 1:
            m |= 1 << 1
        if 1 ^ ((b2 >> 5) & 1):
            m |= 1 << 2
        if (b2 >> 4) & 1:
            m |= 1 << 3
        if (b2 >> 3) & 1:
            m |= 1 << 4
        if (b2 >> 1) & 1:
            m |= 1 << 6
        if (b2 >> 0) & 1:
            m |= 1 << 7
        m |= 1 << 9
        m |= 1 << 12
        return m
    if b0 == 0x61:
        if (b2 >> 1) & 1:
            m |= 1 << 0
        if (b2 >> 0) & 1:
            m |= 1 << 1
        if 1 ^ ((b2 >> 5) & 1):
            m |= 1 << 2
        if (b2 >> 3) & 1:
            m |= 1 << 4
        if (b2 >> 2) & 1:
            m |= 1 << 5
        if (b2 >> 1) & 1:
            m |= 1 << 6
        if (b2 >> 0) & 1:
            m |= 1 << 7
        if (b2 >> 3) & 1:
            m |= 1 << 14
        if (b2 >> 2) & 1:
            m |= 1 << 15
        return m
    # Slice-header family: nal_type=1 and b1 high nibble 0x9
    # (41 9a/9b, 01 9e/9f, ... -- vertical / mobile h5e).
    if (b0 & 0x1F) == 1 and (b1 & 0xF0) == 0x90:
        if (b2 >> 7) & 1:
            m |= 1 << 0
        if (b2 >> 6) & 1:
            m |= 1 << 1
        if ((b0 >> 0) & 1) ^ ((b2 >> 5) & 1):
            m |= 1 << 2
        if (b2 >> 4) & 1:
            m |= 1 << 3
        if (b2 >> 3) & 1:
            m |= 1 << 4
        if (b2 >> 2) & 1:
            m |= 1 << 5
        # s6 = b2[1], s7 = b2[0] (GF(2)-fitted on CCTV-16 4K corpus #104).
        if (b2 >> 1) & 1:
            m |= 1 << 6
        if (b2 >> 0) & 1:
            m |= 1 << 7
        if (b0 >> 0) & 1:
            m |= (1 << 9) | (1 << 10) | (1 << 11) | (1 << 12) | (1 << 14)
        if ((b0 >> 0) & 1) ^ ((b0 >> 6) & 1):
            m |= 1 << 13
        if (b1 >> 0) & 1:
            m |= 1 << 15
        return m
    # Unknown family: no flips (same as 61e020 core only).
    return 0


def type1_g_flips(X: int, Y: int, flip_mask: int) -> int:
    W = (X | (Y << 16)) & 0xFFFFFFFF
    P1 = 0
    for s in range(16):
        fv = _type1_fbit(W) ^ ((flip_mask >> s) & 1)
        b = fv ^ (1 if s in _TYPE1_B_STEPS else 0)
        P1 |= b << (15 - s)
        W = ((W << 1) | b) & 0xFFFFFFFF
    return P1


def type1_decrypt_block(blk, hdr) -> bytes:
    """Decrypt one 4-byte grid cell: (X, Y) -> (G(X, Y), X), both LE16."""
    X = blk[0] | (blk[1] << 8)
    Y = blk[2] | (blk[3] << 8)
    P1 = type1_g_flips(X, Y, type1_flip_mask_from_header(hdr))
    return bytes((P1 & 0xFF, (P1 >> 8) & 0xFF, X & 0xFF, (X >> 8) & 0xFF))


# ===== per-NAL grid decryptors =====

def decrypt_type5_new(nal: bytearray, stride: int) -> int:
    """Type5 new-mode: key@5, start=64, RBSP-indexed grid. Returns new length
    (EPB 03 drop can shrink the NAL)."""
    ln = len(nal)
    if ln < 21 or stride < 8:
        return ln
    key = bytes(nal[5:21])
    epbs = _scan_epbs(nal)
    if epbs:
        r2e = _build_r2e(ln, epbs)
        rbsp_len = len(r2e)
    else:
        r2e = None
        rbsp_len = ln
    k = 0
    while True:
        o = 64 + k * stride
        if o + 16 > rbsp_len or o + 8 > rbsp_len:
            break
        if r2e is None:
            src = slice(o, o + 8)
            blk = bytearray(nal[src])
            tea_decrypt_block(blk, key)
            nal[src] = blk
        else:
            blk = bytearray(nal[r2e[o + b]] for b in range(8))
            tea_decrypt_block(blk, key)
            for b in range(8):
                nal[r2e[o + b]] = blk[b]
        k += 1
    if not epbs:
        return ln
    return _drop_epb_03(nal, epbs)


def decrypt_type1_new(nal: bytearray, stride: int = 511,
                      start: int = 64, guard: int = 17) -> int:
    """Type1 grid with header-derived G flips. *guard* bytes must remain at
    each RBSP cell start even though only 4 are rewritten (worker quirk).
    Returns new length after the EPB 03 drop."""
    ln = len(nal)
    if stride < 4 or ln < 3:
        return ln
    hdr = (nal[0], nal[1], nal[2])
    epbs = _scan_epbs(nal)
    if epbs:
        r2e = _build_r2e(ln, epbs)
        rbsp_len = len(r2e)
    else:
        r2e = None
        rbsp_len = ln
    k = 0
    while True:
        o = start + k * stride
        if o + guard > rbsp_len or o + 4 > rbsp_len:
            break
        if r2e is None:
            idx = slice(o, o + 4)
            nal[idx] = type1_decrypt_block(nal[idx], hdr)
        else:
            cell = bytes(nal[r2e[o + b]] for b in range(4))
            out = type1_decrypt_block(cell, hdr)
            for b in range(4):
                nal[r2e[o + b]] = out[b]
        k += 1
    if not epbs:
        return ln
    return _drop_epb_03(nal, epbs)


# ===== session (per-stream mode state) =====

class H5eSession:
    """Mode state across the NALs of one stream. A type25 'enable' NAL flips
    new mode on; before that, type1/5 NALs use the classic layout."""

    def __init__(self):
        self.new_mode = False
        self.type1_start = 64
        self.type1_guard = 17
        # Worker leaves type1 NALs shorter than this untouched (no grid, no
        # EPB drop).
        self.type1_min_len = 129

    def resolve_type5_stride(self, nal) -> int:
        return type5_stride_from_nal(nal)

    def resolve_type1_stride(self, nal) -> int:
        return type1_stride_f1(nal) or 511

    def on_nal(self, nal: bytearray) -> int:
        """Decrypt one NAL in place; returns its new length."""
        if not nal:
            return 0
        ntype = nal[0] & 0x1F
        if ntype == 25:
            if is_type25_enable(nal):
                self.new_mode = True
            return len(nal)
        if not self.new_mode:
            if ntype in (1, 5):
                decrypt_classic(nal)
            return len(nal)
        if ntype == 5:
            stride = self.resolve_type5_stride(nal)
            if stride >= 8:
                return decrypt_type5_new(nal, stride)
            return len(nal)
        if ntype == 1:
            if len(nal) < self.type1_min_len:
                return len(nal)
            stride = self.resolve_type1_stride(nal)
            return decrypt_type1_new(nal, stride,
                                     self.type1_start, self.type1_guard)
        return len(nal)

    def reset(self):
        self.new_mode = False


# ===== MPEG-TS =====

def _expand_af_steal(data: bytearray, pkt_off: int, need: int) -> int:
    """Grow one TS packet's adaptation field by *need* bytes stolen from its
    payload (hpp expand_af_steal). Returns bytes actually stolen."""
    if need == 0 or pkt_off + 188 > len(data):
        return 0
    afc = (data[pkt_off + 3] & 0x30) >> 4
    if afc == 1:
        # payload only -> introduce an AF. af_len = need - 1 (see hpp notes).
        af_len = min(need - 1, 182)
        steal = 1 + af_len
        old_payload = bytes(data[pkt_off + 4:pkt_off + 188])
        data[pkt_off + 3] = (data[pkt_off + 3] & 0xCF) | 0x30  # afc=3
        data[pkt_off + 4] = af_len
        if af_len > 0:
            data[pkt_off + 5] = 0x00  # flags
            data[pkt_off + 6:pkt_off + 5 + af_len] = b'\xFF' * (af_len - 1)
        new_pl = 184 - steal
        data[pkt_off + 5 + af_len:pkt_off + 5 + af_len + new_pl] = \
            old_payload[:new_pl]
        return steal
    if afc in (2, 3):
        af_len = data[pkt_off + 4]
        pi = 5 + af_len
        if pi >= 188:
            return 0
        old_payload_len = 188 - pi
        add = min(need, old_payload_len)
        if add == 0:
            return 0
        if af_len + add > 182:
            add = 182 - af_len
            if add == 0:
                return 0
        new_af_len = af_len + add
        old_payload = bytes(data[pkt_off + pi:pkt_off + 188])
        # extend the AF body with 0xFF stuffing
        data[pkt_off + 5 + af_len:pkt_off + 5 + af_len + add] = b'\xFF' * add
        data[pkt_off + 4] = new_af_len
        new_pl = old_payload_len - add
        data[pkt_off + 5 + new_af_len:pkt_off + 5 + new_af_len + new_pl] = \
            old_payload[:new_pl]
        data[pkt_off + 3] = (data[pkt_off + 3] & 0xCF) | \
            (0x20 if new_pl == 0 else 0x30)
        return add
    return 0


def decrypt_ts_inplace(data: bytearray, session: H5eSession,
                       vpid: int = 0x100) -> int:
    """Decrypt all video NALs in a TS buffer in place.

    Returns the number of NALs passed through session.on_nal. Every video-PES
    span keeps its TS packet count: when decryption shrinks the ES (EPB 03
    drop), the freed bytes become adaptation-field stuffing.
    """
    if len(data) < 188:
        return 0

    pes = bytearray()
    # (packet_off, payload_start_in_packet, payload_len)
    spans = []
    nal_count = 0

    def flush():
        nonlocal spans, nal_count
        if not pes:
            return

        base_skip = 0
        if len(pes) >= 9 and pes[0] == 0 and pes[1] == 0 and pes[2] == 1:
            base_skip = 9 + pes[8]
        if base_skip > len(pes):
            pes.clear()
            spans.clear()
            return

        pes_hdr = bytes(pes[:base_skip])
        es = bytes(pes[base_skip:])

        # Find NAL start codes in the ES
        starts = []  # (pos, sc_len)
        i = 0
        n = len(es)
        while i + 3 < n:
            if es[i] == 0 and es[i + 1] == 0 and es[i + 2] == 0 and es[i + 3] == 1:
                starts.append((i, 4))
                i += 4
            elif es[i] == 0 and es[i + 1] == 0 and es[i + 2] == 1:
                starts.append((i, 3))
                i += 3
            else:
                i += 1

        new_es = bytearray()
        cursor = 0
        for idx, (pos, sc) in enumerate(starts):
            end = starts[idx + 1][0] if idx + 1 < len(starts) else n
            if cursor < pos:
                new_es += es[cursor:pos]
            new_es += es[pos:pos + sc]
            if pos + sc >= end:
                cursor = end
                continue
            nal = bytearray(es[pos + sc:end])
            nlen = session.on_nal(nal)
            nal_count += 1
            new_es += nal[:nlen]
            cursor = end
        if cursor < n:
            new_es += es[cursor:]

        new_pes = pes_hdr + new_es
        capacity = sum(pl for _, _, pl in spans)
        if capacity > len(new_pes):
            remaining = capacity - len(new_pes)
            for pkt_off, _, _ in reversed(spans):
                if remaining <= 0:
                    break
                remaining -= _expand_af_steal(data, pkt_off, remaining)
            # recompute spans after AF changes
            new_spans = []
            for pkt_off, _, _ in spans:
                afc = (data[pkt_off + 3] & 0x30) >> 4
                if afc == 0 or afc == 2:
                    continue
                pi = 4 if afc == 1 else 5 + data[pkt_off + 4]
                if pi >= 188:
                    continue
                new_spans.append((pkt_off, pi, 188 - pi))
            spans = new_spans

        off = 0
        for pkt_off, pi, pl in spans:
            chunk = min(max(len(new_pes) - off, 0), pl)
            if chunk > 0:
                data[pkt_off + pi:pkt_off + pi + chunk] = new_pes[off:off + chunk]
            if chunk < pl:
                data[pkt_off + pi + chunk:pkt_off + pi + pl] = \
                    b'\xFF' * (pl - chunk)
            off += pl

        pes.clear()
        spans.clear()

    off = 0
    total = len(data)
    while off + 188 <= total:
        if data[off] == 0x47:
            pid = ((data[off + 1] & 0x1F) << 8) | data[off + 2]
            if pid == vpid:
                pusi = (data[off + 1] & 0x40) != 0
                afc = (data[off + 3] & 0x30) >> 4
                if afc not in (0, 2):
                    pi = 4 if afc == 1 else 5 + data[off + 4]
                    if pi < 188:
                        if pusi:
                            flush()
                        pes += data[off + pi:off + 188]
                        spans.append((off, pi, 188 - pi))
        off += 188
    flush()
    return nal_count


def decrypt_ts(data, vpid: int = 0x100, session: Optional[H5eSession] = None) -> bytes:
    """Copy-based decrypt with a fresh (or caller-supplied) session."""
    s = session if session is not None else H5eSession()
    buf = bytearray(data)
    decrypt_ts_inplace(buf, s, vpid)
    return bytes(buf)


# ===== PMT-derived video PID =====

_VIDEO_STREAM_TYPES = frozenset((0x1B, 0x21, 0x24, 0x42))  # AVC, HEVC (2), AVS3


def detect_video_pid(data) -> Optional[int]:
    """Scan a TS buffer's PAT/PMT for the H.264/HEVC video PID.

    Returns None when no PMT is found -- callers then fall back to the
    worker's hardcoded 0x100.
    """
    pmt_pids = set()
    for off in range(0, len(data) - 188 + 1, 188):
        if data[off] != 0x47:
            continue
        pid = ((data[off + 1] & 0x1F) << 8) | data[off + 2]
        pusi = (data[off + 1] & 0x40) != 0
        afc = (data[off + 3] & 0x30) >> 4
        if not pusi or afc not in (1, 3):
            continue
        pi = 4 if afc == 1 else 5 + data[off + 4]
        if pi >= 188:
            continue
        payload = data[off + pi:off + 188]
        if not payload:
            continue
        section = payload[1 + payload[0]:]  # skip pointer_field
        if len(section) < 8:
            continue
        if pid == 0:
            # PAT: 4-byte entries after the 8-byte table header
            slen = ((section[1] & 0x0F) << 8) | section[2]
            end = min(3 + slen - 4, len(section))
            p = 8
            while p + 4 <= end:
                # program_number 0 is the network PID, not a program's PMT
                if section[p] or section[p + 1]:
                    pmt_pids.add(((section[p + 2] & 0x1F) << 8) | section[p + 3])
                p += 4
        elif pid in pmt_pids:
            slen = ((section[1] & 0x0F) << 8) | section[2]
            end = min(3 + slen - 4, len(section))
            pil = ((section[10] & 0x0F) << 8) | section[11] if len(section) >= 12 else 0
            p = 12 + pil
            while p + 5 <= end:
                if section[p] in _VIDEO_STREAM_TYPES:
                    return ((section[p + 1] & 0x1F) << 8) | section[p + 2]
                p += 5 + (((section[p + 3] & 0x0F) << 8) | section[p + 4])
    return None
