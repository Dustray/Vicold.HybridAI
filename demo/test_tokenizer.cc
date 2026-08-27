#include "tokenizer/qwen_tokenizer.h"

#include <chrono>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: test_tokenizer <model_dir>" << std::endl;
        return 1;
    }
    hybridai::tokenizer::QwenTokenizer tokenizer;
    auto status = tokenizer.load(argv[1]);
    if (!status.ok()) {
        std::cerr << "Load failed: " << status.message() << std::endl;
        return 1;
    }
    std::cout << "Loaded." << std::endl;

    std::vector<std::string> prompts = {
        "Hello, how are you?",
        "hello world",
        "  hello   world  ",
        "2025年",
        "<|im_start|>user\nHello\n<|im_end|>\n",
        "'s 't 're 've 'm 'll 'd",
    };

    for (const auto& prompt : prompts) {
        std::cout << "Prompt: " << prompt << std::endl;
        auto t0 = std::chrono::steady_clock::now();
        auto ids = tokenizer.encode(prompt, true);
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "  ids: [";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << ids[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "  time: "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms" << std::endl;
        auto decoded = tokenizer.decode(ids, false);
        std::cout << "  decoded: " << decoded << std::endl;
    }
    return 0;
}
