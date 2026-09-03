#include "hybrid.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_json_string(const std::string& value) {
    std::cout << '"';
    for (const char ch : value) {
        switch (ch) {
        case '\\': std::cout << "\\\\"; break;
        case '"': std::cout << "\\\""; break;
        case '\n': std::cout << "\\n"; break;
        case '\r': std::cout << "\\r"; break;
        case '\t': std::cout << "\\t"; break;
        default: std::cout << ch; break;
        }
    }
    std::cout << '"';
}

void usage(const char* program) {
    std::cerr << "Usage: " << program
              << " <model_dir> [prompt] [backend] [max_new_tokens]"
              << " [warmup_runs] [measure_runs]\n";
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr &&
           (std::string(value) == "1" || std::string(value) == "true" ||
            std::string(value) == "TRUE" || std::string(value) == "on" ||
            std::string(value) == "ON");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    hybridai::GeneratorOptions generator_options;
    generator_options.model_dir = argv[1];
    generator_options.backend = argc > 3 ? argv[3] : "hip";
    generator_options.enable_mtp = env_flag_enabled("HYBRIDAI_ENABLE_MTP");

    hybridai::GenerationOptions request;
    request.enable_speculative_mtp =
        env_flag_enabled("HYBRIDAI_SPECULATIVE_MTP");
    request.prompt = argc > 2 ? argv[2] : "Hello, how are you? And who are you?";
    try {
        if (argc > 4) request.max_new_tokens = std::stoi(argv[4]);
    } catch (const std::exception& error) {
        std::cerr << "invalid max_new_tokens: " << error.what() << '\n';
        return 2;
    }
    int warmup_runs = 1;
    int measure_runs = 5;
    try {
        if (argc > 5) warmup_runs = std::stoi(argv[5]);
        if (argc > 6) measure_runs = std::stoi(argv[6]);
    } catch (const std::exception& error) {
        std::cerr << "invalid warmup/runs: " << error.what() << '\n';
        return 2;
    }
    if (request.max_new_tokens <= 0 || warmup_runs < 0 || measure_runs <= 0) {
        usage(argv[0]);
        return 2;
    }

    hybridai::Generator generator(generator_options);
    const auto load_start = std::chrono::steady_clock::now();
    auto status = generator.load_model();
    const double load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - load_start).count();
    if (!status.ok()) {
        std::cerr << "load_model failed: " << status.message() << '\n';
        return 1;
    }

    for (int i = 0; i < warmup_runs; ++i) {
        hybridai::GenerationResult ignored;
        status = generator.generate(request, &ignored);
        if (!status.ok()) {
            std::cerr << "warmup failed: " << status.message() << '\n';
            return 1;
        }
    }

    std::cout << std::setprecision(12);
    for (int run = 1; run <= measure_runs; ++run) {
        hybridai::GenerationResult result;
        status = generator.generate(request, &result);
        if (!status.ok()) {
            std::cerr << "generate failed: " << status.message() << '\n';
            return 1;
        }
        const double output_tps = result.elapsed_seconds > 0.0
            ? static_cast<double>(result.token_ids.size()) /
                  result.elapsed_seconds
            : 0.0;
        std::cout << "{\"schema_version\":1"
                  << ",\"run\":" << run
                  << ",\"model_dir\":";
        print_json_string(generator_options.model_dir);
        std::cout << ",\"backend\":";
        print_json_string(generator_options.backend);
        std::cout << ",\"prompt\":";
        print_json_string(request.prompt);
        std::cout << ",\"prompt_tokens\":" << result.prompt_tokens
                  << ",\"generated_tokens\":" << result.token_ids.size()
                  << ",\"max_new_tokens\":" << request.max_new_tokens
                  << ",\"load_seconds\":" << load_seconds
                  << ",\"elapsed_seconds\":" << result.elapsed_seconds
                  << ",\"ttft_seconds\":"
                  << result.time_to_first_token_seconds
                  << ",\"decode_seconds\":" << result.decode_seconds
                  << ",\"decode_tokens_per_second\":"
                  << result.decode_tokens_per_second
                  << ",\"output_tokens_per_second\":" << output_tps
                  << ",\"mtp_proposed_tokens\":"
                  << result.mtp_proposed_tokens
                  << ",\"mtp_accepted_tokens\":"
                  << result.mtp_accepted_tokens
                  << ",\"mtp_rejected_tokens\":"
                  << result.mtp_rejected_tokens
                  << ",\"mtp_correction_tokens\":"
                  << result.mtp_correction_tokens
                  << ",\"speculative_replay_tokens\":"
                  << result.speculative_replay_tokens
                  << ",\"speculative_mtp_recovery_tokens\":"
                  << result.speculative_mtp_recovery_tokens
                  << ",\"speculative_max_proposal_width\":"
                  << result.speculative_max_proposal_width
                  << ",\"speculative_mtp_cache_clone_count\":"
                  << result.speculative_mtp_cache_clone_count
                  << ",\"speculative_target_cache_clone_count\":"
                  << result.speculative_target_cache_clone_count
                  << ",\"mtp_fallback_steps\":"
                  << result.mtp_fallback_steps
                  << ",\"mtp_acceptance_rate\":"
                  << result.mtp_acceptance_rate
                  << ",\"speculative_proposal_seconds\":"
                  << result.speculative_proposal_seconds
                  << ",\"speculative_verification_seconds\":"
                  << result.speculative_verification_seconds
                  << ",\"speculative_replay_seconds\":"
                  << result.speculative_replay_seconds
                  << ",\"speculative_fallback_seconds\":"
                  << result.speculative_fallback_seconds
                  << ",\"speculative_mtp_cache_clone_seconds\":"
                  << result.speculative_mtp_cache_clone_seconds
                  << ",\"speculative_target_cache_clone_seconds\":"
                  << result.speculative_target_cache_clone_seconds
                  << ",\"speculative_argmax_seconds\":"
                  << result.speculative_argmax_seconds
                  << ",\"speculative_rounds\":"
                  << result.speculative_rounds
                  << ",\"token_ids\":[";
        for (size_t i = 0; i < result.token_ids.size(); ++i) {
            if (i != 0) std::cout << ',';
            std::cout << result.token_ids[i];
        }
        std::cout
                  << "],\"decode_step_seconds\":[";
        for (size_t i = 0; i < result.decode_step_seconds.size(); ++i) {
            if (i != 0) std::cout << ',';
            std::cout << result.decode_step_seconds[i];
        }
        std::cout << "]}\n";
    }
    return 0;
}
