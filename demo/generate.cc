// run: HIP_ALLOC_INITIALIZE=0 HIP_VISIBLE_DEVICES=2  ./demo/build/generate /public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master/ "Hello, how are you?" hip 1024

#include "hybrid.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
              << " <model_dir> [prompt] [backend] [max_new_tokens]"
                 " [--enable-mtp] [--speculative-mtp]\n";
        return 1;
    }
    hybridai::GeneratorOptions model;
    model.model_dir = argv[1];
    model.backend = argc > 3 ? argv[3] : "hip";
    bool enable_mtp = false;
    bool speculative_mtp = false;
    for (int i = 5; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--enable-mtp") {
            enable_mtp = true;
        } else if (option == "--speculative-mtp") {
            enable_mtp = true;
            speculative_mtp = true;
        } else {
            std::cerr << "Unknown option: " << option << '\n';
            return 1;
        }
    }
    model.enable_mtp = enable_mtp;
    hybridai::Generator generator(model);
    auto status = generator.load_model();
    if (!status.ok()) {
        std::cerr << "load_model failed: " << status.message() << '\n';
        return 1;
    }
    hybridai::GenerationOptions request;
    request.prompt = argc > 2 ? argv[2] : "Hello, how are you?";
    if (argc > 4) request.max_new_tokens = std::stoi(argv[4]);
    request.enable_speculative_mtp = speculative_mtp;
    hybridai::GenerationResult result;
    status = generator.generate(request, &result);
    if (!status.ok()) {
        std::cerr << "generate failed: " << status.message() << '\n';
        return 1;
    }
    std::cout << "token_ids:";
    for (auto id : result.token_ids) std::cout << ' ' << id;
    std::cout << "\ntext:\n" << result.text << '\n';
    std::cout << "prompt_tokens=" << result.prompt_tokens
              << " generated_tokens=" << result.token_ids.size()
              << " decode_tps=" << result.decode_tokens_per_second
              << " mtp_proposed=" << result.mtp_proposed_tokens
              << " mtp_accepted=" << result.mtp_accepted_tokens
              << " mtp_fallback=" << result.mtp_fallback_steps << '\n';
    return 0;
}