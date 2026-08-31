#include "hybrid.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_dir> [prompt] [backend] [max_new_tokens]\n";
        return 1;
    }
    hybridai::GeneratorOptions model;
    model.model_dir = argv[1];
    model.backend = argc > 3 ? argv[3] : "hip";
    hybridai::Generator generator(model);
    auto status = generator.load_model();
    if (!status.ok()) {
        std::cerr << "load_model failed: " << status.message() << '\n';
        return 1;
    }
    hybridai::GenerationOptions request;
    request.prompt = argc > 2 ? argv[2] : "Hello, how are you?";
    if (argc > 4) request.max_new_tokens = std::stoi(argv[4]);
    hybridai::GenerationResult result;
    status = generator.generate(request, &result);
    if (!status.ok()) {
        std::cerr << "generate failed: " << status.message() << '\n';
        return 1;
    }
    std::cout << "token_ids:";
    for (auto id : result.token_ids) std::cout << ' ' << id;
    std::cout << "\ntext:\n" << result.text << '\n';
    return 0;
}