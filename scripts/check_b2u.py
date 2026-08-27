def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

b2u = bytes_to_unicode()
print("space byte 0x20 ->", repr(b2u[0x20]))
print("cap A byte 0x41 ->", repr(b2u[0x41]))
print("byte 0x00 ->", repr(b2u[0x00]))
print("byte 0x0A ->", repr(b2u[0x0A]))
print("byte 0xC4 ->", repr(b2u[0xC4]))
