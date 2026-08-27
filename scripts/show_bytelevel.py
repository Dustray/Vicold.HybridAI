from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
b2u = bytes_to_unicode()
print("len", len(b2u))
print("space byte", repr(b2u[0x20]))
print("newline byte", repr(b2u[0x0A]))
print("0xC4 byte", repr(b2u[0xC4]))
print("first few:")
for b, u in sorted(b2u.items())[:10]:
    print(b, repr(u))
