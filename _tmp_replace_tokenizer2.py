import pathlib

p = pathlib.Path(r'd:\Vicold\Vicold.HybridAI\demo\qwen_infer.cc')
lines = p.read_text(encoding='utf-8').splitlines()
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if line == '// 占位 tokenizer：当前项目没有 tokenizer 实现。':
        new_lines.extend([
            '// Tokenizer 实例，由 main 初始化后传入。',
            'const hybridai::tokenizer::QwenTokenizer* g_tokenizer = nullptr;',
            '',
            'std::vector<int64_t> tokenize(const std::string& text) {',
            '    if (g_tokenizer != nullptr && g_tokenizer->is_loaded()) {',
            '        return g_tokenizer->encode(text, true);',
            '    }',
            '    std::cerr << "[WARN] Tokenizer not loaded. "',
            '              << "Using placeholder token ids." << std::endl;',
            '    return {151644, 872, 198, 151645};',
            '}',
            '',
            'std::string decode_ids(const std::vector<int64_t>& ids) {',
            '    if (g_tokenizer != nullptr && g_tokenizer->is_loaded()) {',
            '        return g_tokenizer->decode(ids, false);',
            '    }',
            '    std::string out = "<decoded> ";',
            '    for (int64_t id : ids) {',
            '        out += std::to_string(id) + " ";',
            '    }',
            '    out += "</decoded>";',
            '    return out;',
            '}',
            '',
            '// 选择可用设备',
        ])
        while i < len(lines) and lines[i] != '// 选择可用设备':
            i += 1
        i += 1
        continue
    new_lines.append(line)
    i += 1

s2 = '\n'.join(new_lines)
s2 = s2.replace('placeholder_tokenize(prompt)', 'tokenize(prompt)')
s2 = s2.replace('placeholder_decode(output_ids)', 'decode_ids(output_ids)')
p.write_text(s2, encoding='utf-8')
print('replaced')
