import pathlib

path = pathlib.Path(r'd:\Vicold\Vicold.HybridAI\demo\qwen_infer.cc')
s = path.read_text(encoding='utf-8')

old = '''// 占位 tokenizer：当前项目没有 tokenizer 实现。
// 真实实现需要加载 tokenizer.json / merges.txt / vocab.json 并把文本映射
// 到 ids。这里直接返回一个固定的 id 序列，并打印明确提示。
std::vector<int64_t> placeholder_tokenize(const std::string& text) {
    (void)text;
    std::cerr << "[WARN] No tokenizer is implemented yet. "
              << "Using placeholder token ids [151644, 872, 198, 151645]."
              << std::endl;
    // 151644/151645 大致对应 <|im_start|>/与۩，872 是 "hello" 类 token
    return {151644, 872, 198, 151645};
}

std::string placeholder_decode(const std::vector<int64_t>& ids) {
    std::string out = "<decoded> ";
    for (int64_t id : ids) {
        out += std::to_string(id) + " ";
    }
    out += "</decoded>";
    std::cerr << "[WARN] No tokenizer decoder implemented yet. "
              << "Returning raw ids." << std::endl;
    return out;
}

// 选择可用设备
'''

new = '''// Tokenizer 实例，由 main 初始化后传入。
const hybridai::tokenizer::QwenTokenizer* g_tokenizer = nullptr;

std::vector<int64_t> tokenize(const std::string& text) {
    if (g_tokenizer != nullptr && g_tokenizer->is_loaded()) {
        return g_tokenizer->encode(text, true);
    }
    std::cerr << "[WARN] Tokenizer not loaded. "
              << "Using placeholder token ids." << std::endl;
    return {151644, 872, 198, 151645};
}

std::string decode_ids(const std::vector<int64_t>& ids) {
    if (g_tokenizer != nullptr && g_tokenizer->is_loaded()) {
        return g_tokenizer->decode(ids, false);
    }
    std::string out = "<decoded> ";
    for (int64_t id : ids) {
        out += std::to_string(id) + " ";
    }
    out += "</decoded>";
    return out;
}

// 选择可用设备
'''

if old not in s:
    print('old block not found')
else:
    s = s.replace(old, new)
    s = s.replace('placeholder_tokenize(prompt)', 'tokenize(prompt)')
    s = s.replace('placeholder_decode(output_ids)', 'decode_ids(output_ids)')
    path.write_text(s, encoding='utf-8')
    print('replaced')
