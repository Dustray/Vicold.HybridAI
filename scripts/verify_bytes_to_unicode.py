import json

# This is the actual GPT-2 byte-to-unicode mapping used by HuggingFace
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

b2u = bytes_to_unicode()
# What does byte 0x20 map to?
print('0x20 ->', repr(b2u[0x20]), 'cp', hex(ord(b2u[0x20])))
# What do bytes 0xe4 0xa0 correspond to in unicode? decode utf8
print('bytes e4a0 ->', b'\xe4\xa0'.decode('utf-8'), 'cp', hex(ord(b'\xe4\xa0'.decode('utf-8'))))

# The merge string raw bytes \xc4\xa0 is utf8 for U+0120? Let's check.
print('c4a0 decode', b'\xc4\xa0'.decode('utf-8'), hex(ord(b'\xc4\xa0'.decode('utf-8'))))
# So raw JSON uses U+0120. Python json rendered it as 臓 because the terminal font maps it? Wait ord returned 0x81d3 not 0x120.
# Maybe Python's json decoder converted it wrong due to surrogate? Actually \xc4\xa0 valid utf8 -> U+0120. Let's decode manually.
manual = b'\xc4\xa0'.decode('utf-8')
print('manual', repr(manual), hex(ord(manual)))
