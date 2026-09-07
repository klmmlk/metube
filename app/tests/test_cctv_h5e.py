"""Unit tests for the hls_h5e decryptor port (app/cctv_h5e.py).

The tests build synthetic encrypted payloads with a mirror of the module's
grid encryptor (TEA round-trip, RBSP-mapped cells), so they verify the port
against itself rather than against official fixtures -- the structural
invariants (exact plaintext recovery, TS packet count, PES header, EPB drop)
are what must hold.
"""

import struct

import pytest

from cctv_h5e import (
    H5eSession,
    _build_r2e,
    _scan_epbs,
    decrypt_classic,
    decrypt_ts,
    decrypt_ts_inplace,
    decrypt_type1_new,
    decrypt_type5_new,
    detect_video_pid,
    is_type25_enable,
    tea_decrypt_block,
    tea_encrypt_block,
    type1_flip_mask_from_header,
    type1_g_flips,
    type1_stride_f1,
    type5_stride_f5,
    type5_stride_from_nal,
)

KEY = bytes(range(16))


def _plain(length, seed=13):
    """Position-unique filler that never contains a 00 byte (so no accidental
    start codes / EPBs appear anywhere but where the test splices them in)."""
    return bytes((((i * 7 + seed) & 0x7F) | 0x80) for i in range(length))


# --- TEA ----------------------------------------------------------------------

def test_tea_roundtrip():
    block = bytearray(b'\x11\x22\x33\x44\x55\x66\x77\x88')
    orig = bytes(block)
    tea_encrypt_block(block, KEY)
    assert bytes(block) != orig
    tea_decrypt_block(block, KEY)
    assert bytes(block) == orig


def test_tea_all_ones_still_scrambles():
    block = bytearray(b'\xff' * 8)
    orig = bytes(block)
    tea_encrypt_block(block, KEY)
    assert bytes(block) != orig
    tea_decrypt_block(block, KEY)
    assert bytes(block) == orig


# --- classic layout ----------------------------------------------------------

def test_decrypt_classic_roundtrip():
    plain = _plain(232)
    nal = bytearray(b'\x21\x00' + plain[:14] + KEY + plain[30:])
    for o in (32, 112, 192):
        blk = bytearray(nal[o:o + 8])
        tea_encrypt_block(blk, KEY)
        nal[o:o + 8] = blk
    decrypt_classic(nal)
    assert bytes(nal) == b'\x21\x00' + plain[:14] + KEY + plain[30:]


def test_decrypt_classic_short_nal_noop():
    nal = bytearray(b'\x21' + b'\x81' * 30)
    orig = bytes(nal)
    decrypt_classic(nal)
    assert bytes(nal) == orig


# --- strides -------------------------------------------------------------------

@pytest.mark.parametrize('key,expected', [
    # stride = base[le32(key[0:4]) % 6] | key[that idx] -- for idx < 4 the
    # OR'd byte sits *inside* the LE word itself (C++ reads the same byte
    # twice), so the word's residue and the byte value are coupled.
    (bytes((0x30, 0, 0, 0, 0, 0)), 160 | 0x30),   # 48 % 6 == 0 -> idx 0
    (bytes((1, 0x21, 0, 0, 0, 0)), 192 | 0x21),   # 0x2101 % 6 == 1
    (bytes((2, 0, 0x21, 0, 0, 0)), 224 | 0x21),   # 0x210002 % 6 == 2
    (bytes((3, 0, 0, 0x21, 0, 0)), 256 | 0x21),   # 0x21000003 % 6 == 3
    (bytes((4, 0, 0, 0, 0x21, 0)), 288 | 0x21),   # outside the LE word
    (bytes((5, 0, 0, 0, 0, 0x21)), 320 | 0x21),
])
def test_type5_stride_formula_all_indices(key, expected):
    assert type5_stride_f5(key) == expected


def test_type5_stride_from_nal_reads_key_at_5():
    key = bytearray(6)
    struct.pack_into('<I', key, 0, 4)         # -> base[4] = 256
    key[4] = 0x7F                             # ... | key[4]
    nal = bytes(5) + bytes(key) + b'\x81' * 10
    assert type5_stride_from_nal(nal) == 256 | 0x7F


def test_type1_stride_keyed_at_1():
    key = bytearray(6)
    struct.pack_into('<I', key, 0, 5)         # -> base[5] = 320
    key[5] = 0x11                             # ... | key[5]
    nal = b'\x41' + bytes(key) + b'\x81' * 10
    assert type1_stride_f1(nal) == 320 | 0x11


# --- type25 / session ----------------------------------------------------------

def _type25_enable_nal():
    return bytearray(b'\x19\x00\x01\x09' + _plain(40))


def test_is_type25_enable():
    assert is_type25_enable(_type25_enable_nal())
    assert not is_type25_enable(bytearray(b'\x19\x00\x01\x0a' + _plain(40)))
    assert not is_type25_enable(bytearray(b'\x01\x00\x01\x09' + _plain(40)))


def test_session_type25_flips_new_mode():
    s = H5eSession()
    assert not s.new_mode
    s.on_nal(_type25_enable_nal())
    assert s.new_mode


def test_session_non_enable_type25_keeps_mode():
    s = H5eSession()
    s.on_nal(bytearray(b'\x19\x00\x01\x05' + _plain(40)))
    assert not s.new_mode


# --- type5 new-mode grid -------------------------------------------------------

def _encrypt_type5_nal(epb_offsets=()):
    """Build a type5 NAL whose RBSP grid is TEA-encrypted, mirroring the
    decryptor's cell indexing (through the RBSP map when EPBs exist).

    The embedded key is 16 zero bytes, so the derived stride is 160
    (LE word 0 -> base[0] | key[0]). *epb_offsets* are payload offsets
    (>= 0, counted after byte 21) at which 00 00 03 is spliced into the
    EBSP -- the CDN's emulation prevention.
    Returns (encrypted_nal, stride, rbsp_plain)."""
    key = bytearray(16)
    nal = bytearray(b'\x25\x00\x00\x00\x00' + key + _plain(512))
    stride = type5_stride_from_nal(nal)
    assert stride == 160
    for p in sorted(epb_offsets, reverse=True):
        nal[21 + p:21 + p] = b'\x00\x00\x03'
    epbs = _scan_epbs(nal)
    r2e = _build_r2e(len(nal), epbs) if epbs else None
    rbsp_len = len(r2e) if r2e else len(nal)
    # the expected decrypt output: the plain NAL with every EPB 03 removed,
    # snapshotted BEFORE the grid cells are encrypted
    rbsp = bytearray(nal)
    for e in reversed(epbs):
        del rbsp[e + 2]
    k = 0
    while True:
        o = 64 + k * stride
        if o + 16 > rbsp_len:
            break
        idx = list(range(o, o + 8)) if r2e is None else r2e[o:o + 8]
        blk = bytearray(nal[i] for i in idx)
        tea_encrypt_block(blk, bytes(key))
        for j, i in enumerate(idx):
            nal[i] = blk[j]
        k += 1
    return nal, stride, bytes(rbsp)


def test_decrypt_type5_new_roundtrip_no_epb():
    nal, stride, rbsp = _encrypt_type5_nal()
    nlen = decrypt_type5_new(nal, stride)
    assert nlen == len(nal) == len(rbsp)
    assert bytes(nal) == rbsp


def test_decrypt_type5_new_epb_grid_uses_rbsp_indexing():
    # An EPB before the first cell shifts every EBSP position: the grid must
    # index the RBSP, and the dropped 03s shrink the NAL back to the RBSP.
    nal, stride, rbsp = _encrypt_type5_nal(epb_offsets=(40, 300))
    orig_len = len(nal)
    nlen = decrypt_type5_new(nal, stride)
    assert nlen == len(rbsp) == orig_len - 2
    assert bytes(nal[:nlen]) == rbsp


def test_decrypt_type5_new_short_nal_untouched():
    nal = bytearray(b'\x25' + b'\x11' * 15)   # 16 bytes < 21
    orig = bytes(nal)
    assert decrypt_type5_new(nal, 176) == 16
    assert bytes(nal) == orig


# --- type1 ---------------------------------------------------------------------

def test_type1_flip_mask_families():
    # 01 a8 00: bit 2 (inverted), 9, 12
    assert type1_flip_mask_from_header((0x01, 0xA8, 0x00)) == \
        (1 << 2) | (1 << 9) | (1 << 12)
    # 01 a8 ff: b2 bits 7..0 all set (bit2 needs b2[5]==0 so it drops out)
    assert type1_flip_mask_from_header((0x01, 0xA8, 0xFF)) == \
        (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4) | (1 << 6) | (1 << 7) | \
        (1 << 9) | (1 << 12)
    # 61 e0 20: only b2[5] set -> no flip at all
    assert type1_flip_mask_from_header((0x61, 0xE0, 0x20)) == 0
    # unknown family
    assert type1_flip_mask_from_header((0x09, 0x00, 0x00)) == 0


def test_type1_g_flips_deterministic_and_mask_sensitive():
    assert type1_g_flips(0x1234, 0x5678, 0) == type1_g_flips(0x1234, 0x5678, 0)
    assert type1_g_flips(0x1234, 0x5678, 0) != type1_g_flips(0x1234, 0x5678, 1)


def test_decrypt_type1_new_grid_layout():
    # slice-header family header 41 9a 00, stride 511 -> cells at 64 and 575
    nal = bytearray(b'\x41\x9a\x00' + _plain(600))
    orig = bytes(nal)
    nlen = decrypt_type1_new(nal, 511)
    assert nlen == len(nal)                      # no EPB -> same length
    # cell tail carries X back (bytes 2:4 == original bytes 0:2)
    assert bytes(nal[66:68]) == orig[64:66]
    assert bytes(nal[577:579]) == orig[575:577]
    # everything off the grid is untouched
    assert bytes(nal[3:64]) == orig[3:64]
    assert bytes(nal[68:575]) == orig[68:575]
    assert bytes(nal[579:]) == orig[579:]


def test_decrypt_type1_new_epb_drop_shrinks():
    nal = bytearray(b'\x41\x9a\x00' + _plain(300))
    nal[100:100] = b'\x00\x00\x03'
    orig_len = len(nal)
    nlen = decrypt_type1_new(nal, 511)
    assert nlen == orig_len - 1


def test_decrypt_type1_new_min_length_guard():
    # session-level guard: short type1 NALs are left untouched in new mode
    s = H5eSession()
    s.on_nal(_type25_enable_nal())
    nal = bytearray(b'\x41\x9a\x00' + _plain(100))     # 103 < 129
    orig = bytes(nal)
    assert s.on_nal(nal) == len(nal)
    assert bytes(nal) == orig


# --- EPB helpers ---------------------------------------------------------------

def test_scan_epbs_non_overlapping():
    assert _scan_epbs(b'\x00\x00\x03\x00\x00\x03\x81\x82') == [0, 3]
    assert _scan_epbs(b'\x00\x00\x01\x00\x00\x02') == []


def test_build_r2e_skips_each_03():
    data = b'\x81\x00\x00\x03\x82\x00\x00\x03\x83'
    r2e = _build_r2e(len(data), _scan_epbs(data))
    assert bytes(data[i] for i in r2e) == b'\x81\x00\x00\x82\x00\x00\x83'


# --- MPEG-TS -------------------------------------------------------------------

def _ts_packet(pid, payload, pusi=False):
    hdr3 = 0x10 | (0x40 if pusi else 0)
    assert len(payload) <= 184
    pkt = bytearray(188)
    pkt[0] = 0x47
    pkt[1] = ((pid >> 8) & 0x1F) | (0x40 if pusi else 0)
    pkt[2] = pid & 0xFF
    pkt[3] = hdr3
    pkt[4:4 + len(payload)] = payload
    return bytes(pkt)


def _ts_packet_af(pid, payload, pusi=False):
    """TS packet whose payload ends exactly at the packet boundary, the tail
    filled by adaptation-field stuffing (afc=3) -- conformant TS. The decryptor
    treats every payload byte as PES data, so a final short packet MUST be
    stuffed, never zero-padded."""
    assert 0 < len(payload) <= 183
    pkt = bytearray(188)
    pkt[0] = 0x47
    pkt[1] = ((pid >> 8) & 0x1F) | (0x40 if pusi else 0)
    pkt[2] = pid & 0xFF
    pkt[3] = 0x30 | (0x40 if pusi else 0)
    af_len = 183 - len(payload)
    pkt[4] = af_len
    if af_len:
        pkt[5] = 0x00                      # flags
        pkt[6:5 + af_len] = b'\xff' * (af_len - 1)
    pkt[5 + af_len:] = payload
    return bytes(pkt)


def _pes_packets(pid, nal_chunks, stream_id=0xE0):
    """PES (9-byte header, no optional fields) chunked into TS packets."""
    es = b''.join(b'\x00\x00\x00\x01' + nal for nal in nal_chunks)
    pes = bytes((0, 0, 1, stream_id, 0, 0, 0x80, 0, 0)) + es
    pkts = []
    while pes:
        chunk, pes = pes[:184], pes[184:]
        if not pes and len(chunk) <= 183:
            pkts.append(_ts_packet_af(pid, chunk, pusi=not pkts))
        else:
            pkts.append(_ts_packet(pid, chunk, pusi=not pkts))
    return pkts


def _parse_pes(data, pid):
    """Reassemble PES payloads from *data* for *pid* (test oracle)."""
    pes = bytearray()
    for off in range(0, len(data) - 187, 188):
        assert data[off] == 0x47, f'lost TS sync at {off}'
        if ((data[off + 1] & 0x1F) << 8) | data[off + 2] != pid:
            continue
        afc = (data[off + 3] & 0x30) >> 4
        pi = 4 if afc == 1 else 5 + data[off + 4]
        if pi < 188:
            pes += data[off + pi:off + 188]
    return pes


def test_decrypt_ts_new_mode_roundtrip():
    nal25 = bytes(_type25_enable_nal())
    nal5, stride, rbsp = _encrypt_type5_nal()
    pkts = _pes_packets(0x100, [nal25, bytes(nal5)])
    other = _ts_packet(0x101, b'\xEE' * 100)
    ts = pkts[0] + other + b''.join(pkts[1:])
    data = bytearray(ts)

    session = H5eSession()
    count = decrypt_ts_inplace(data, session)
    assert count == 2
    assert session.new_mode
    assert len(data) == len(ts)
    # non-video packet untouched
    assert bytes(data[188:376]) == other
    # PES header intact; ES = intact type25 NAL + decrypted type5 RBSP
    pes = _parse_pes(bytes(data), 0x100)
    assert pes[:9] == bytes((0, 0, 1, 0xE0, 0, 0, 0x80, 0, 0))
    es = bytes(pes[9:])
    assert es.startswith(b'\x00\x00\x00\x01\x19\x00\x01\x09')
    i = es.find(b'\x00\x00\x00\x01\x25')
    assert i >= 0
    assert es[i + 4:i + 4 + len(rbsp)] == rbsp


def test_decrypt_ts_classic_mode():
    plain = _plain(232)
    expected_plain = b'\x21\x00' + plain[:14] + KEY + plain[30:]
    nal = bytearray(expected_plain)
    for o in (32, 112, 192):
        blk = bytearray(nal[o:o + 8])
        tea_encrypt_block(blk, KEY)
        nal[o:o + 8] = blk
    data = bytearray(b''.join(_pes_packets(0x100, [bytes(nal)])))
    session = H5eSession()
    assert decrypt_ts_inplace(data, session) == 1
    assert not session.new_mode
    pes = _parse_pes(bytes(data), 0x100)
    assert bytes(pes[9 + 4:]) == expected_plain


def test_decrypt_ts_epb_shrink_keeps_packet_count():
    nal25 = bytes(_type25_enable_nal())
    nal5, stride, rbsp = _encrypt_type5_nal(epb_offsets=(70, 300))
    pkts = _pes_packets(0x100, [nal25, bytes(nal5)])
    ts = b''.join(pkts)
    data = bytearray(ts)
    decrypt_ts_inplace(data, H5eSession())
    assert len(data) == len(ts)
    for off in range(0, len(data), 188):
        assert data[off] == 0x47
    # ES shrank by exactly the two dropped EPB bytes...
    pes = _parse_pes(bytes(data), 0x100)
    assert len(pes) == len(_parse_pes(ts, 0x100)) - 2
    # ...and the freed TS capacity became adaptation-field stuffing
    assert b'\xff\xff\xff' in bytes(data)
    # decrypted type5 NAL == its RBSP
    es = bytes(pes[9:])
    i = es.find(b'\x00\x00\x00\x01\x25')
    assert es[i + 4:i + 4 + len(rbsp)] == rbsp


def test_decrypt_ts_skips_other_pids():
    ts = b''.join(_pes_packets(0x233, [bytes(_type25_enable_nal())]))
    data = bytearray(ts)
    assert decrypt_ts_inplace(data, H5eSession(), vpid=0x100) == 0
    assert bytes(data) == ts


def test_decrypt_ts_copy_wrapper():
    ts = b''.join(_pes_packets(0x100, [bytes(_type25_enable_nal())]))
    out = decrypt_ts(ts)
    assert isinstance(out, bytes)
    assert len(out) == len(ts)


def test_decrypt_ts_reuses_session_across_segments():
    # The proxy hands each segment to the same session: type25 arrives with
    # segment 1, and segment 2 (no type25) must still decrypt in new mode.
    session = H5eSession()
    nal25 = bytes(_type25_enable_nal())
    seg1 = bytearray(b''.join(_pes_packets(0x100, [nal25])))
    decrypt_ts_inplace(seg1, session)
    assert session.new_mode

    nal5, stride, rbsp = _encrypt_type5_nal()
    seg2 = bytearray(b''.join(_pes_packets(0x100, [bytes(nal5)])))
    decrypt_ts_inplace(seg2, session)
    es = bytes(_parse_pes(bytes(seg2), 0x100)[9:])
    assert es[4:4 + len(rbsp)] == rbsp


# --- PMT video-PID detection ---------------------------------------------------

def _psi_packet(pid, section):
    return _ts_packet(pid, b'\x00' + section, pusi=True)   # pointer_field = 0


def _pat(pmt_pid=0x100):
    entries = struct.pack('>HH', 1, 0xE000 | pmt_pid)
    body = struct.pack('>H', 1) + b'\xc1\x00\x00' + entries
    slen = len(body) + 4
    return b'\x00' + struct.pack('>H', 0xB000 | slen) + body + b'\0\0\0\0'


def _pmt(video_pid=0x100, stype=0x1B):
    entry = bytes((stype,)) + struct.pack('>HH', 0xE000 | video_pid, 0xF000)
    body = struct.pack('>H', 1) + b'\xc1\x00\x00' + \
        struct.pack('>HH', 0xE000 | video_pid, 0xF000) + entry
    slen = len(body) + 4
    return b'\x02' + struct.pack('>H', 0xB000 | slen) + body + b'\0\0\0\0'


def test_detect_video_pid_from_pmt():
    ts = _psi_packet(0x0000, _pat(pmt_pid=0x0FFF)) + \
        _psi_packet(0x0FFF, _pmt(video_pid=0x233))
    assert detect_video_pid(ts) == 0x233


def test_detect_video_pid_hevc_stream_type():
    ts = _psi_packet(0x0000, _pat(pmt_pid=0x0FFF)) + \
        _psi_packet(0x0FFF, _pmt(video_pid=0x101, stype=0x24))
    assert detect_video_pid(ts) == 0x101


def test_detect_video_pid_without_psi_returns_none():
    assert detect_video_pid(_ts_packet(0x100, b'\x81' * 10)) is None
