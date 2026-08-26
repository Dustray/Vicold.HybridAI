#pragma once

#include <cstdint>
#include <string>

namespace hybridai {
namespace platform {

// Returns the number of host CPU cores (hardware concurrency).
int32_t cpu_count() noexcept;

// Sleep the current thread for at least `ms` milliseconds.
void sleep_ms(int32_t ms) noexcept;

// Yield the current thread time slice.
void thread_yield() noexcept;

// Get a process-wide monotonic timestamp in microseconds.
int64_t monotonic_us() noexcept;

// Returns the absolute path of the current executable's directory.
std::string executable_dir();

// Join path components in a platform-correct way.
std::string join_path(const std::string& a, const std::string& b);

// Returns true if the given file path exists.
bool file_exists(const std::string& path);

// Get the size of a file in bytes, or -1 if it does not exist.
int64_t file_size(const std::string& path);

// Set thread name (for debugging / profiling). Limited length on some OSes.
void set_thread_name(const std::string& name);

} // namespace platform
} // namespace hybridai
