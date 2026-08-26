#include "core/platform.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace hybridai {
namespace platform {

int32_t cpu_count() noexcept {
    const int32_t count = static_cast<int32_t>(std::thread::hardware_concurrency());
    return count > 0 ? count : 1;
}

void sleep_ms(int32_t ms) noexcept {
    if (ms <= 0) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

void thread_yield() noexcept {
    std::this_thread::yield();
}

int64_t monotonic_us() noexcept {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
}

std::string executable_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    DWORD len = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (len == 0) return "";
    std::filesystem::path p(std::wstring(buf, len));
    return p.parent_path().string();
#else
    std::error_code ec;
    std::filesystem::path p = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) return "";
    return p.parent_path().string();
#endif
}

std::string join_path(const std::string& a, const std::string& b) {
    std::filesystem::path pa(a);
    std::filesystem::path pb(b);
    return (pa / pb).string();
}

bool file_exists(const std::string& path) {
    return std::filesystem::exists(path);
}

int64_t file_size(const std::string& path) {
    std::error_code ec;
    auto size = std::filesystem::file_size(path, ec);
    if (ec) return -1;
    return static_cast<int64_t>(size);
}

void set_thread_name(const std::string& name) {
    std::string truncated = name.substr(0, 15);
#ifdef _WIN32
    // Thread description is available on Windows 10 1607+; best-effort.
    using SetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PCWSTR);
    HMODULE kernel = GetModuleHandleW(L"KernelBase.dll");
    if (kernel == nullptr) kernel = GetModuleHandleW(L"kernel32.dll");
    if (kernel != nullptr) {
        auto fn = reinterpret_cast<SetThreadDescriptionFn>(
            GetProcAddress(kernel, "SetThreadDescription"));
        if (fn != nullptr) {
            int len = MultiByteToWideChar(CP_UTF8, 0, truncated.c_str(), -1,
                                          nullptr, 0);
            if (len > 0) {
                std::wstring wname(len - 1, L'\0');
                MultiByteToWideChar(CP_UTF8, 0, truncated.c_str(), -1,
                                    wname.data(), len);
                fn(GetCurrentThread(), wname.c_str());
            }
        }
    }
#else
    pthread_setname_np(pthread_self(), truncated.c_str());
#endif
}

} // namespace platform
} // namespace hybridai
