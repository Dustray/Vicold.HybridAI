import unicodedata

for cp in [ord('H'), ord('e'), ord('l'), ord('o'), ord(','), ord(' '), ord('h'), ord('w'), ord('a'), ord('r'), ord('u'), ord('?'), ord('年'), ord('C'), ord('a'), ord('f'), ord('é')]:
    cat = unicodedata.category(chr(cp))
    print(f"U+{cp:04X} '{chr(cp)}' category={cat} letter={cat.startswith('L')} number={cat.startswith('N')} mark={cat.startswith('M')}")
