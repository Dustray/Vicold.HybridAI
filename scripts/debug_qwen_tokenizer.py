import json
import re
from collections import OrderedDict

class QwenTokenizerDebug:
    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model = data['model']
        self.vocab = model['vocab']  # dict token->id
        self.merges = model['merges']
        # Build merge ranks
        self.mr = {}
        for rank, m in enumerate(self.merges):
            a, b = m.split(' ')
            self.mr[(a, b)] = rank
        # Byte-level unicode mapping
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    @staticmethod
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def bpe(self, token):
        word = tuple(token)
        pairs = self.get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.mr.get(pair, float('inf')))
            if bigram not in self.mr:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = self.get_pairs(word)
        return word

    @staticmethod
    def get_pairs(word):
        pairs = set()
        prev = word[0]
        for w in word[1:]:
            pairs.add((prev, w))
            prev = w
        return pairs

    def encode(self, text):
        # Simple manual pre-tokenization: split on whitespace, attach leading space to token
        tokens = []
        for match in re.finditer(r'\S+', text):
            start = match.start()
            if start > 0 and text[start-1] == ' ':
                # count preceding spaces
                j = start - 1
                while j >= 0 and text[j] == ' ':
                    j -= 1
                space_count = start - 1 - j
                tokens.append(' ' * space_count + match.group())
            else:
                tokens.append(match.group())
        print('pre_tokens:', tokens)
        bpe_tokens = []
        for token in tokens:
            # encode each byte to unicode
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            # BPE
            bpe_result = self.bpe(token_translated)
            for bpe_token in bpe_result:
                if bpe_token in self.vocab:
                    bpe_tokens.append(self.vocab[bpe_token])
                else:
                    # fallback char by char
                    for c in bpe_token:
                        if c in self.vocab:
                            bpe_tokens.append(self.vocab[c])
                        else:
                            bpe_tokens.append(self.vocab.get('<|endoftext|>', 0))
        return bpe_tokens

tok = QwenTokenizerDebug('E:/models/Qwen3.8-27B-FP8/tokenizer.json')
print('encode Hello:', tok.encode('Hello, how are you?'))
