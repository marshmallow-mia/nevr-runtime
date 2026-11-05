#include "runtime/patch/render_config_hook.h"

#include <winsock2.h>
#include <windows.h>

#include <dxgi.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "nevr/pc/nevr_common.h"
#include "runtime/hook/patching.h"
#include "runtime/patch/gpu_score.h"

namespace nevr_render_config_hook {
namespace {

using nevr_render_config::Collector;
using nevr_render_config::Externals;
using nevr_render_config::MemoryReader;
using nevr_render_config::SettingsObserver;

// ApplyRuntimeGfxSettings @ 0x140c31870 (echovr.exe, read back): the first 27 bytes, up to and including
// `SUB RSP,0x60`. Nothing in them is position dependent.
constexpr std::array<std::uint8_t, 27> kApplyPrologue = {0x48, 0x89, 0x5c, 0x24, 0x10, 0x48, 0x89, 0x74, 0x24,
                                                          0x18, 0x48, 0x89, 0x7c, 0x24, 0x20, 0x55, 0x41, 0x56,
                                                          0x41, 0x57, 0x48, 0x8b, 0xec, 0x48, 0x83, 0xec, 0x60};
// The GPU auto-selector @ 0x140c31e00 (read back): the first 26 bytes, up to and including `MOV RBX,RCX`.
constexpr std::array<std::uint8_t, 26> kSelectorPrologue = {0x48, 0x89, 0x5c, 0x24, 0x08, 0x48, 0x89, 0x6c, 0x24,
                                                             0x10, 0x48, 0x89, 0x74, 0x24, 0x18, 0x57, 0x48, 0x83,
                                                             0xec, 0x20, 0x48, 0x8b, 0xea, 0x48, 0x8b, 0xd9};

// Both return a 64-bit value. The game's functions return nothing the callers read (the callers at 0x14010920e and
// 0x1401091e9 ignore RAX), but passing RAX through unchanged makes the observers exact whatever the original leaves
// there.
using ApplyFn = std::uint64_t (*)(void* settings, void* game, int logFlag);
using SelectorFn = std::uint64_t (*)(void* settings, void* game);

// Reads this process's own memory. ReadProcessMemory on the current process answers FALSE for an unmapped or
// protected page instead of faulting, and the answer and the copy are one system call: a VirtualQuery check followed by
// a memcpy leaves a window in which the page can go away. The reads are a handful of fixed game globals once a second
// and one 40-byte settings struct per Apply call, so the system call costs nothing that matters.
class ProcessReader final : public MemoryReader {
 public:
  bool Read(std::uintptr_t address, void* out, std::size_t length) const override {
    SIZE_T got = 0;
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(address), out, length, &got) != 0 &&
           got == length;
  }
};

const ProcessReader g_processReader;

// ---- The two values that are not game memory. Both run once, at the first line. ----

std::string QueryGpuDriver(const std::string& gpuName) {
  // DXGI's IDXGIAdapter::CheckInterfaceSupport(IDXGIDevice) answers the user-mode driver version as a LARGE_INTEGER,
  // four 16-bit parts. The game has no NVAPI import, so this is the one source for it that needs nothing new in the
  // process: dxgi.dll is loaded by the game already and is looked up, never loaded here.
  HMODULE dxgi = GetModuleHandleW(L"dxgi.dll");
  if (dxgi == nullptr) return "unknown";
  using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
  const auto create =
      reinterpret_cast<CreateFactoryFn>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
  if (create == nullptr) return "unknown";
  IDXGIFactory1* factory = nullptr;
  if (FAILED(create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) || factory == nullptr) {
    return "unknown";
  }
  std::string driver = "unknown";
  for (UINT index = 0;; ++index) {
    IDXGIAdapter* adapter = nullptr;
    if (factory->EnumAdapters(index, &adapter) != S_OK || adapter == nullptr) break;
    DXGI_ADAPTER_DESC desc = {};
    if (SUCCEEDED(adapter->GetDesc(&desc))) {
      char narrow[256] = {};
      WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, narrow, sizeof(narrow) - 1, nullptr, nullptr);
      if (nevr_render_config::CleanText(narrow) == gpuName) {
        LARGE_INTEGER umd = {};
        if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd))) {
          char text[48];
          std::snprintf(text, sizeof(text), "%u.%u.%u.%u", static_cast<unsigned>((umd.HighPart >> 16) & 0xFFFF),
                        static_cast<unsigned>(umd.HighPart & 0xFFFF), static_cast<unsigned>((umd.LowPart >> 16) & 0xFFFF),
                        static_cast<unsigned>(umd.LowPart & 0xFFFF));
          driver = text;
        }
        adapter->Release();
        break;
      }
    }
    adapter->Release();
  }
  factory->Release();
  return driver;
}

std::string QueryRuntimeVersion() {
  // The game loads LibOVRRT64_1.dll itself and resolves ovr_* by name; the version string comes the same way.
  HMODULE runtime = GetModuleHandleW(L"LibOVRRT64_1.dll");
  if (runtime == nullptr) return "unknown";
  using VersionFn = const char*(__cdecl*)();
  const auto version =
      reinterpret_cast<VersionFn>(reinterpret_cast<void*>(GetProcAddress(runtime, "ovr_GetVersionString")));
  if (version == nullptr) return "unknown";
  const char* text = version();
  if (text == nullptr) return "unknown";
  const std::string clean = nevr_render_config::CleanText(text);
  return clean.empty() ? std::string("unknown") : clean;
}

// ---- The GPU score, when the game's own rating failed (#1419). The pure rule is patch/gpu_score.h. A Windows run with a
// working driver keeps the game's own score and never gets here. ----

// NVAPI function IDs, asked of nvapi_QueryInterface. GetGpuCoreCount is the call the game makes for the cores in its device
// init (echovr.exe 0x14058f98c..0x14058fa04); the others set up the GPU handle it takes.
constexpr std::uint32_t kNvApiInitialize = 0x0150E828U;
constexpr std::uint32_t kNvApiEnumPhysicalGpus = 0xE5AC921FU;
constexpr std::uint32_t kNvApiGetGpuCoreCount = 0xC7026A87U;
constexpr std::size_t kNvApiMaxPhysicalGpus = 64;

using NvStatus = std::int32_t;
using NvQueryInterfaceFn = void* (*)(std::uint32_t id);
using NvInitializeFn = NvStatus (*)();
using NvEnumPhysicalGpusFn = NvStatus (*)(void** handles, std::uint32_t* count);
using NvGetGpuCoreCountFn = NvStatus (*)(void* handle, std::uint32_t* count);

// NVAPI's core count of the first GPU. Under Proton dxvk-nvapi refuses it (`nvapi_get_gpu_core_count_failed` on an RTX
// 2080 Ti); the first GPU is the one the game's device is on on a one-NVIDIA machine.
void QueryNvapiCores(nevr_gpu_score::NvidiaAnswers* out) {
  HMODULE nvapi = GetModuleHandleW(L"nvapi64.dll");
  if (nvapi == nullptr) {
    out->nvapiFailure = "nvapi_absent";
    return;
  }
  const auto query = reinterpret_cast<NvQueryInterfaceFn>(
      reinterpret_cast<void*>(GetProcAddress(nvapi, "nvapi_QueryInterface")));
  if (query == nullptr) {
    out->nvapiFailure = "nvapi_no_query_interface";
    return;
  }
  const auto initialize = reinterpret_cast<NvInitializeFn>(query(kNvApiInitialize));
  const auto enumGpus = reinterpret_cast<NvEnumPhysicalGpusFn>(query(kNvApiEnumPhysicalGpus));
  const auto coreCount = reinterpret_cast<NvGetGpuCoreCountFn>(query(kNvApiGetGpuCoreCount));
  if (initialize == nullptr || enumGpus == nullptr || coreCount == nullptr) {
    out->nvapiFailure = "nvapi_function_missing";
    return;
  }
  if (initialize() != 0) {
    out->nvapiFailure = "nvapi_initialize_failed";
    return;
  }
  void* handles[kNvApiMaxPhysicalGpus] = {};
  std::uint32_t count = 0;
  if (enumGpus(handles, &count) != 0 || count == 0) {
    out->nvapiFailure = "nvapi_enum_gpus_failed";
    return;
  }
  std::uint32_t cores = 0;
  if (coreCount(handles[0], &cores) != 0 || cores == 0) {
    out->nvapiFailure = "nvapi_get_gpu_core_count_failed";
    return;
  }
  out->nvapiCoresOk = true;
  out->nvapiCores = cores;
}

// NVML (under Proton, Wine's nvml.dll bridges to the host driver): the first device's CUDA compute capability and its
// maximum graphics clock.
using NvmlInitFn = int (*)();
using NvmlShutdownFn = int (*)();
using NvmlGetHandleByIndexFn = int (*)(unsigned index, void** device);
using NvmlGetCudaComputeCapabilityFn = int (*)(void* device, int* major, int* minor);
using NvmlGetMaxClockInfoFn = int (*)(void* device, int clockType, unsigned* mhz);
constexpr int kNvmlClockGraphics = 0;

void QueryNvml(nevr_gpu_score::NvidiaAnswers* out) {
  HMODULE nvml = LoadLibraryW(L"nvml.dll");
  if (nvml == nullptr) {
    out->nvmlFailure = "nvml_absent";
    return;
  }
  const auto init = reinterpret_cast<NvmlInitFn>(reinterpret_cast<void*>(GetProcAddress(nvml, "nvmlInit_v2")));
  const auto shutdown = reinterpret_cast<NvmlShutdownFn>(reinterpret_cast<void*>(GetProcAddress(nvml, "nvmlShutdown")));
  const auto handle = reinterpret_cast<NvmlGetHandleByIndexFn>(
      reinterpret_cast<void*>(GetProcAddress(nvml, "nvmlDeviceGetHandleByIndex_v2")));
  const auto capability = reinterpret_cast<NvmlGetCudaComputeCapabilityFn>(
      reinterpret_cast<void*>(GetProcAddress(nvml, "nvmlDeviceGetCudaComputeCapability")));
  const auto maxClock = reinterpret_cast<NvmlGetMaxClockInfoFn>(
      reinterpret_cast<void*>(GetProcAddress(nvml, "nvmlDeviceGetMaxClockInfo")));
  if (init == nullptr || handle == nullptr || capability == nullptr || maxClock == nullptr) {
    out->nvmlFailure = "nvml_function_missing";
  } else if (init() != 0) {
    out->nvmlFailure = "nvml_init_failed";
  } else {
    void* device = nullptr;
    unsigned mhz = 0;
    if (handle(0, &device) != 0 || device == nullptr) {
      out->nvmlFailure = "nvml_no_device";
    } else if (capability(device, &out->cudaMajor, &out->cudaMinor) != 0) {
      out->nvmlFailure = "nvml_cuda_capability_failed";
    } else if (maxClock(device, kNvmlClockGraphics, &mhz) != 0 || mhz == 0) {
      out->nvmlFailure = "nvml_max_clock_failed";
    } else {
      out->nvmlOk = true;
      out->maxClockMhz = mhz;
    }
    if (shutdown != nullptr) shutdown();
  }
  FreeLibrary(nvml);
}

// The Linux host's DRM class directory, through the Z: drive Wine maps to the host's root. It exists only under Wine.
constexpr const char* kHostDrmRoot = "Z:\\sys\\class\\drm";

// The Vulkan structures the queries read, as the Vulkan 1.1 headers lay them out (the tree has no Vulkan headers).
constexpr std::int32_t kVkStructureTypeApplicationInfo = 0;
constexpr std::int32_t kVkStructureTypeInstanceCreateInfo = 1;
constexpr std::int32_t kVkStructureTypePhysicalDeviceProperties2 = 1000059001;
constexpr std::int32_t kVkStructureTypeShaderSmBuiltinsPropertiesNv = 1000154001;
constexpr std::int32_t kVkStructureTypeShaderCorePropertiesAmd = 1000185000;
constexpr std::int32_t kVkStructureTypeShaderCoreProperties2Amd = 1000227000;
constexpr std::uint32_t kVkApiVersion11 = (1U << 22) | (1U << 12);
constexpr std::uint32_t kVkMaxPhysicalDevices = 16;

struct VkApplicationInfoLite {
  std::int32_t sType;
  const void* pNext;
  const char* pApplicationName;
  std::uint32_t applicationVersion;
  const char* pEngineName;
  std::uint32_t engineVersion;
  std::uint32_t apiVersion;
};
struct VkInstanceCreateInfoLite {
  std::int32_t sType;
  const void* pNext;
  std::uint32_t flags;
  const VkApplicationInfoLite* pApplicationInfo;
  std::uint32_t enabledLayerCount;
  const char* const* ppEnabledLayerNames;
  std::uint32_t enabledExtensionCount;
  const char* const* ppEnabledExtensionNames;
};
// VkPhysicalDeviceProperties2: VkPhysicalDeviceProperties is 824 bytes (vendorID at +8, deviceID at +12).
struct VkPhysicalDeviceProperties2Lite {
  std::int32_t sType;
  void* pNext;
  std::uint32_t apiVersion;
  std::uint32_t driverVersion;
  std::uint32_t vendorID;
  std::uint32_t deviceID;
  unsigned char rest[824 - 16];
};
static_assert(sizeof(VkPhysicalDeviceProperties2Lite) == 840, "VkPhysicalDeviceProperties2 is 840 bytes on x64");
struct VkShaderSmBuiltinsPropertiesNvLite {
  std::int32_t sType;
  void* pNext;
  std::uint32_t shaderSMCount;
  std::uint32_t shaderWarpsPerSM;
};
// VkPhysicalDeviceShaderCorePropertiesAMD: 14 uint32 after the header; the first three are the CU layout.
struct VkShaderCorePropertiesAmdLite {
  std::int32_t sType;
  void* pNext;
  std::uint32_t shaderEngineCount;
  std::uint32_t shaderArraysPerEngineCount;
  std::uint32_t computeUnitsPerShaderArray;
  std::uint32_t rest[11];
};
struct VkShaderCoreProperties2AmdLite {
  std::int32_t sType;
  void* pNext;
  std::uint32_t shaderCoreFeatures;
  std::uint32_t activeComputeUnitCount;
};
struct VkExtensionPropertiesLite {
  char extensionName[256];
  std::uint32_t specVersion;
};

using VkGetInstanceProcAddrFn = void* (*)(void* instance, const char* name);
using VkCreateInstanceFn = std::int32_t (*)(const VkInstanceCreateInfoLite* info, const void* allocator, void** instance);
using VkDestroyInstanceFn = void (*)(void* instance, const void* allocator);
using VkEnumeratePhysicalDevicesFn = std::int32_t (*)(void* instance, std::uint32_t* count, void** devices);
using VkGetPhysicalDeviceProperties2Fn = void (*)(void* device, VkPhysicalDeviceProperties2Lite* properties);
using VkEnumerateDeviceExtensionPropertiesFn = std::int32_t (*)(void* device, const char* layer, std::uint32_t* count,
                                                                VkExtensionPropertiesLite* properties);

// The PCI vendor and device id of the DXGI adapter whose description is the game's GPU name: the adapter the game rated
// (it matches its AGS device by this name, 0x14058eb90..0x14058ebb2).
bool AdapterIdsByName(const std::string& gpuName, std::uint32_t* vendorId, std::uint32_t* deviceId) {
  HMODULE dxgi = GetModuleHandleW(L"dxgi.dll");
  if (dxgi == nullptr) return false;
  using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
  const auto create =
      reinterpret_cast<CreateFactoryFn>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
  if (create == nullptr) return false;
  IDXGIFactory1* factory = nullptr;
  if (FAILED(create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) || factory == nullptr) return false;
  bool found = false;
  for (UINT index = 0; !found; ++index) {
    IDXGIAdapter* adapter = nullptr;
    if (factory->EnumAdapters(index, &adapter) != S_OK || adapter == nullptr) break;
    DXGI_ADAPTER_DESC desc = {};
    if (SUCCEEDED(adapter->GetDesc(&desc))) {
      char narrow[256] = {};
      WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, narrow, sizeof(narrow) - 1, nullptr, nullptr);
      if (nevr_render_config::CleanText(narrow) == gpuName) {
        *vendorId = desc.VendorId;
        *deviceId = desc.DeviceId;
        found = true;
      }
    }
    adapter->Release();
  }
  factory->Release();
  return found;
}

// What Vulkan says of one device: NVIDIA's SM count (VK_NV_shader_sm_builtins) and AMD's compute units.
struct VulkanCores {
  bool found = false;
  std::uint32_t nvidiaSms = 0;
  std::uint32_t amdComputeUnits = 0;
  const char* failure = "";
};

// The core properties of the Vulkan device with this PCI vendor and device id; the step that failed in `failure`.
VulkanCores QueryVulkanCores(std::uint32_t vendorId, std::uint32_t deviceId) {
  VulkanCores out;
  HMODULE vulkan = LoadLibraryW(L"vulkan-1.dll");
  if (vulkan == nullptr) {
    out.failure = "vulkan_absent";
    return out;
  }
  const auto getProc = reinterpret_cast<VkGetInstanceProcAddrFn>(
      reinterpret_cast<void*>(GetProcAddress(vulkan, "vkGetInstanceProcAddr")));
  const auto createInstance =
      getProc != nullptr ? reinterpret_cast<VkCreateInstanceFn>(getProc(nullptr, "vkCreateInstance")) : nullptr;
  void* instance = nullptr;
  if (createInstance == nullptr) {
    out.failure = "vulkan_no_create_instance";
  } else {
    const VkApplicationInfoLite app = {kVkStructureTypeApplicationInfo, nullptr, "nevr-runtime", 1, "nevr-runtime", 1,
                                       kVkApiVersion11};
    const VkInstanceCreateInfoLite info = {kVkStructureTypeInstanceCreateInfo, nullptr, 0, &app, 0, nullptr, 0, nullptr};
    if (createInstance(&info, nullptr, &instance) != 0 || instance == nullptr) {
      instance = nullptr;
      out.failure = "vulkan_create_instance_failed";
    }
  }
  if (instance != nullptr) {
    const auto destroy = reinterpret_cast<VkDestroyInstanceFn>(getProc(instance, "vkDestroyInstance"));
    const auto enumerate = reinterpret_cast<VkEnumeratePhysicalDevicesFn>(getProc(instance, "vkEnumeratePhysicalDevices"));
    const auto properties2 =
        reinterpret_cast<VkGetPhysicalDeviceProperties2Fn>(getProc(instance, "vkGetPhysicalDeviceProperties2"));
    const auto extensions = reinterpret_cast<VkEnumerateDeviceExtensionPropertiesFn>(
        getProc(instance, "vkEnumerateDeviceExtensionProperties"));
    void* devices[kVkMaxPhysicalDevices] = {};
    std::uint32_t count = kVkMaxPhysicalDevices;
    if (enumerate == nullptr || properties2 == nullptr || extensions == nullptr) {
      out.failure = "vulkan_function_missing";
    } else if (enumerate(instance, &count, devices) < 0) {
      out.failure = "vulkan_enumerate_failed";
    } else {
      out.failure = "vulkan_no_matching_device";
      for (std::uint32_t i = 0; i < count && i < kVkMaxPhysicalDevices; ++i) {
        VkPhysicalDeviceProperties2Lite props = {};
        props.sType = kVkStructureTypePhysicalDeviceProperties2;
        properties2(devices[i], &props);
        if (props.vendorID != vendorId || props.deviceID != deviceId) continue;
        // Only extensions the device has go into the chain.
        std::uint32_t extensionCount = 0;
        extensions(devices[i], nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionPropertiesLite> list(extensionCount);
        if (extensionCount != 0) extensions(devices[i], nullptr, &extensionCount, list.data());
        const auto has = [&list](const char* name) {
          for (const VkExtensionPropertiesLite& extension : list) {
            if (std::strcmp(extension.extensionName, name) == 0) return true;
          }
          return false;
        };
        VkShaderSmBuiltinsPropertiesNvLite sm = {};
        sm.sType = kVkStructureTypeShaderSmBuiltinsPropertiesNv;
        VkShaderCorePropertiesAmdLite core = {};
        core.sType = kVkStructureTypeShaderCorePropertiesAmd;
        VkShaderCoreProperties2AmdLite core2 = {};
        core2.sType = kVkStructureTypeShaderCoreProperties2Amd;
        void* chain = nullptr;
        if (has("VK_NV_shader_sm_builtins")) {
          sm.pNext = chain;
          chain = &sm;
        }
        if (has("VK_AMD_shader_core_properties")) {
          core.pNext = chain;
          chain = &core;
        }
        if (has("VK_AMD_shader_core_properties2")) {
          core2.pNext = chain;
          chain = &core2;
        }
        props.pNext = chain;
        properties2(devices[i], &props);
        out.found = true;
        out.failure = "";
        out.nvidiaSms = sm.shaderSMCount;
        out.amdComputeUnits =
            nevr_gpu_score::AmdComputeUnits(core2.activeComputeUnitCount, core.shaderEngineCount,
                                            core.shaderArraysPerEngineCount, core.computeUnitsPerShaderArray);
        break;
      }
    }
    if (destroy != nullptr) destroy(instance, nullptr);
  }
  FreeLibrary(vulkan);
  return out;
}

// NVIDIA: NVAPI's cores, else Vulkan's SMs for the game's adapter x NVML's cores per SM; the clock from NVML
// (patch/gpu_score.h, MeasureNvidia).
void QueryNvidia(const std::string& gpuName, nevr_gpu_score::NvidiaMeasure* out) {
  nevr_gpu_score::NvidiaAnswers answers;
  QueryNvapiCores(&answers);
  if (!answers.nvapiCoresOk) {
    std::uint32_t vendorId = 0;
    std::uint32_t deviceId = 0;
    if (!AdapterIdsByName(gpuName, &vendorId, &deviceId)) {
      answers.vulkanFailure = "dxgi_adapter_not_found";
    } else {
      const VulkanCores vulkan = QueryVulkanCores(vendorId, deviceId);
      answers.vulkanSmsOk = vulkan.found && vulkan.nvidiaSms != 0;
      answers.vulkanSms = vulkan.nvidiaSms;
      answers.vulkanFailure = vulkan.found && vulkan.nvidiaSms == 0 ? "vulkan_no_sm_builtins" : vulkan.failure;
    }
  }
  QueryNvml(&answers);
  *out = nevr_gpu_score::MeasureNvidia(answers);
}

using NvidiaQueryFn = void (*)(const std::string& gpuName, nevr_gpu_score::NvidiaMeasure* out);

// AMD: compute units from Vulkan and the Linux host's top engine clock (patch/gpu_score.h, AmdScore). Each value that
// answers sets its `have` flag; the first step that does not answer names itself in `failure`.
void QueryAmd(const std::string& gpuName, nevr_gpu_score::AmdMeasure* out) {
  std::uint32_t vendorId = 0;
  std::uint32_t deviceId = 0;
  if (!AdapterIdsByName(gpuName, &vendorId, &deviceId)) {
    out->failure = "dxgi_adapter_not_found";
    return;
  }
  const VulkanCores vulkan = QueryVulkanCores(vendorId, deviceId);
  if (!vulkan.found) {
    out->failure = vulkan.failure;
    return;
  }
  if (vulkan.amdComputeUnits == 0) {
    out->failure = "vulkan_no_shader_core_properties";
    return;
  }
  out->haveComputeUnits = true;
  out->computeUnits = vulkan.amdComputeUnits;
  nevr_gpu_score::ReadHostSclk(kHostDrmRoot, deviceId, out);
}

using AmdQueryFn = void (*)(const std::string& gpuName, nevr_gpu_score::AmdMeasure* out);

// ---- State. Written by Install and the hooks, read by Poll. ----

ApplyFn g_originalApply = nullptr;
SelectorFn g_originalSelector = nullptr;
std::atomic<const MemoryReader*> g_reader{&g_processReader};
Externals g_externals = {&QueryGpuDriver, &QueryRuntimeVersion};
SettingsObserver g_observer;
std::atomic<std::uintptr_t> g_bootReturn{0};
std::atomic<std::uintptr_t> g_gameBase{0};
std::atomic<std::uintptr_t> g_lastApplyReturn{0};
std::atomic<std::uint64_t> g_swallowed{0};  // exceptions the hooks caught: they cannot log on the game's path
std::unique_ptr<Collector> g_collector;
std::atomic<bool> g_active{false};
std::atomic<bool> g_serverSkipLogged{false};
std::atomic<bool> g_checkFailureLogged{false};
std::atomic<bool> g_traceLogged{false};
std::atomic<bool> g_scoreLogged{false};
NvidiaQueryFn g_nvidiaQuery = &QueryNvidia;
AmdQueryFn g_amdQuery = &QueryAmd;

// noinline: __builtin_return_address(0) must name this function's caller, the game's call site.
__attribute__((noinline)) std::uint64_t ApplyHook(void* settings, void* game, int logFlag) {
  const auto returnAddress = reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));
  const std::uint64_t result = g_originalApply(settings, game, logFlag);
  try {
    g_lastApplyReturn.store(returnAddress, std::memory_order_relaxed);
    g_observer.OnApply(*g_reader.load(std::memory_order_acquire), reinterpret_cast<std::uintptr_t>(settings),
                       returnAddress, g_bootReturn.load(std::memory_order_relaxed));
  } catch (const std::exception&) {
    g_swallowed.fetch_add(1, std::memory_order_relaxed);
  }
  return result;
}

// Writes the score the selector is about to read (#1419), where the game's getter reads it (fcn.1400df7a0 is
// MOVSD XMM0,[0x14209d000]). The game's setter (fcn.1400dfeb0) is not called: its bytes are not checked here.
bool WriteGameScore(std::uintptr_t address, double value) {
  SIZE_T written = 0;
  return WriteProcessMemory(GetCurrentProcess(), reinterpret_cast<LPVOID>(address), &value, sizeof(value), &written) != 0 &&
         written == sizeof(value);
}

// The game's GPU name (kGpuNameVa), cleaned as the render_config line cleans it; "" when it cannot be read.
std::string ReadGameGpuName(const MemoryReader& reader, std::uintptr_t base) {
  char name[nevr_render_config::kGpuNameBytes + 1] = {};
  if (!reader.Read(nevr_render_config::Resolve(base, nevr_render_config::kGpuNameVa), name,
                   nevr_render_config::kGpuNameBytes)) {
    return "";
  }
  return nevr_render_config::CleanText(name);
}

// Runs in the selector hook before the original selector reads the score. The game's own rating is kept when it is not
// zero; a zero is filled for NVIDIA with both NVAPI values and for AMD with both AMD values (patch/gpu_score.h). The
// decision is logged once a run.
void FillGpuScore() {
  const std::uintptr_t base = g_gameBase.load(std::memory_order_relaxed);
  const MemoryReader& reader = *g_reader.load(std::memory_order_acquire);
  std::uint32_t vendor = 0;
  double gameScore = 0.0;
  const bool readOk = reader.Read(nevr_render_config::Resolve(base, nevr_render_config::kGpuVendorVa), &vendor, sizeof(vendor)) &&
                      reader.Read(nevr_render_config::Resolve(base, nevr_render_config::kGpuScoreVa), &gameScore, sizeof(gameScore));

  nevr_gpu_score::Decision decision;
  if (!readOk) {
    decision.reason = "game_unreadable";
  } else {
    nevr_gpu_score::NvidiaMeasure nvidia;
    nevr_gpu_score::AmdMeasure amd;
    if (gameScore == 0.0 && vendor == nevr_gpu_score::kVendorNvidia) g_nvidiaQuery(ReadGameGpuName(reader, base), &nvidia);
    if (gameScore == 0.0 && vendor == nevr_gpu_score::kVendorAmd) g_amdQuery(ReadGameGpuName(reader, base), &amd);
    decision = nevr_gpu_score::DecideScore(vendor, gameScore, nvidia, amd);
    if (decision.outcome == nevr_gpu_score::Outcome::Filled &&
        !WriteGameScore(nevr_render_config::Resolve(base, nevr_render_config::kGpuScoreVa), decision.score)) {
      decision = nevr_gpu_score::Decision();
      decision.reason = "write_failed";
      decision.score = gameScore;
    }
  }

  if (!g_scoreLogged.exchange(true)) {
    Log(decision.outcome == nevr_gpu_score::Outcome::Unrated ? EchoVR::LogLevel::Warning : EchoVR::LogLevel::Info,
        "[NEVR.RENDER] gpu_score %s vendor=%u game_score=%.17g score=%.17g source=%s reason=%s",
        nevr_gpu_score::OutcomeName(decision.outcome), static_cast<unsigned>(vendor), gameScore, decision.score,
        decision.source, decision.reason[0] != '\0' ? decision.reason : "-");
  }
}

std::uint64_t SelectorHook(void* settings, void* game) {
  try {
    FillGpuScore();
  } catch (const std::exception&) {
    g_swallowed.fetch_add(1, std::memory_order_relaxed);
  }
  const std::uint64_t result = g_originalSelector(settings, game);
  try {
    g_observer.OnSelector(reinterpret_cast<std::uintptr_t>(settings));
  } catch (const std::exception&) {
    g_swallowed.fetch_add(1, std::memory_order_relaxed);
  }
  return result;
}

std::string HexBytes(const void* address, std::size_t count) {
  if (address == nullptr) return "unreadable";
  std::string out;
  char byte[4];
  for (std::size_t i = 0; i < count; ++i) {
    std::snprintf(byte, sizeof(byte), "%02x", static_cast<const std::uint8_t*>(address)[i]);
    out += byte;
  }
  return out;
}

bool PrologueMatches(const char* name, std::uint64_t va, void* target, const std::uint8_t* expected,
                     std::size_t length) {
  if (nevr::ValidatePrologue(target, expected, length)) return true;
  Log(EchoVR::LogLevel::Warning,
      "[NEVR.RENDER] hook off name=%s va=0x%llx reason=prologue_mismatch expected=%s actual=%s "
      "(the settings fields of the render_config line read unavailable)",
      name, static_cast<unsigned long long>(va), HexBytes(expected, 8).c_str(), HexBytes(target, 8).c_str());
  return false;
}

}  // namespace

bool Install(std::uintptr_t gameBase, bool isServer) {
  if (isServer) {
    if (!g_serverSkipLogged.exchange(true)) {
      Log(EchoVR::LogLevel::Info, "[NEVR.RENDER] render_config skipped reason=server_mode (a server renders nothing)");
    }
    return false;
  }
  if (g_active.load(std::memory_order_acquire)) return true;
  if (gameBase == 0) {
    Log(EchoVR::LogLevel::Warning, "[NEVR.RENDER] render_config off reason=no_game_base");
    return false;
  }

  g_bootReturn.store(nevr_render_config::Resolve(gameBase, nevr_render_config::kBootApplyReturnVa),
                     std::memory_order_relaxed);
  g_gameBase.store(gameBase, std::memory_order_relaxed);
  g_collector = std::make_unique<Collector>(*g_reader.load(std::memory_order_acquire), gameBase, g_observer, g_externals);
  // The check works with or without the hooks: without them the settings fields read unavailable.
  g_active.store(true, std::memory_order_release);

  void* const apply = nevr::ResolveVA_Checked(gameBase, nevr_render_config::kApplyVa);
  void* const selector = nevr::ResolveVA_Checked(gameBase, nevr_render_config::kSelectorVa);
  const bool applyOk = PrologueMatches("ApplyRuntimeGfxSettings", nevr_render_config::kApplyVa, apply,
                                       kApplyPrologue.data(), kApplyPrologue.size());
  const bool selectorOk = PrologueMatches("GfxSettingsAutoSelect", nevr_render_config::kSelectorVa, selector,
                                          kSelectorPrologue.data(), kSelectorPrologue.size());
  if (!applyOk || !selectorOk) return false;

  g_originalApply = reinterpret_cast<ApplyFn>(apply);
  g_originalSelector = reinterpret_cast<SelectorFn>(selector);
  const BOOL applyHooked = PatchDetour(&g_originalApply, reinterpret_cast<PVOID>(&ApplyHook), "ApplyRuntimeGfxSettings");
  const BOOL selectorHooked =
      PatchDetour(&g_originalSelector, reinterpret_cast<PVOID>(&SelectorHook), "GfxSettingsAutoSelect");
  Log(applyHooked && selectorHooked ? EchoVR::LogLevel::Info : EchoVR::LogLevel::Warning,
      "[NEVR.RENDER] render_config hooks apply=%s selector=%s (settings source and values on the render_config line)",
      applyHooked ? "installed" : "failed", selectorHooked ? "installed" : "failed");
  return applyHooked && selectorHooked;
}

void Poll(std::uint64_t nowUs) {
  if (!g_active.load(std::memory_order_acquire)) return;
  try {
    const std::string line = g_collector->Poll(nowUs / 1000);
    if (!line.empty()) {
      Log(EchoVR::LogLevel::Info, "%s", line.c_str());
      // Once, with the first line: what the observers saw, so a `settings_source=unknown` or `menu_or_json` explains itself
      // (observers installed after the game's boot Apply call, or a boot return address that is not the one the game uses).
      if (!g_traceLogged.exchange(true)) {
        const SettingsObserver::View seen = g_observer.Snapshot();
        Log(EchoVR::LogLevel::Info,
            "[NEVR.RENDER] render_config settings_trace apply_calls=%llu read_failures=%llu last_apply_return=0x%llx "
            "boot_return=0x%llx",
            static_cast<unsigned long long>(seen.applyCalls), static_cast<unsigned long long>(seen.readFailures),
            static_cast<unsigned long long>(g_lastApplyReturn.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(g_bootReturn.load(std::memory_order_relaxed)));
      }
    }
  } catch (const std::exception& error) {
    // Out of memory while building the line: the frame goes on, and the failure is said once.
    if (!g_checkFailureLogged.exchange(true)) {
      Log(EchoVR::LogLevel::Warning, "[NEVR.RENDER] render_config check failed reason=exception what=%s", error.what());
    }
  }
}

#ifdef NEVR_RENDER_CONFIG_TESTING
void SetReaderForTest(const MemoryReader* reader) {
  g_reader.store(reader != nullptr ? reader : &g_processReader, std::memory_order_release);
}
void SetExternalsForTest(Externals externals) { g_externals = std::move(externals); }
void SetNvidiaQueryForTest(void (*query)(const std::string& gpuName, nevr_gpu_score::NvidiaMeasure* out)) {
  g_nvidiaQuery = query != nullptr ? query : &QueryNvidia;
}
void SetAmdQueryForTest(void (*query)(const std::string& gpuName, nevr_gpu_score::AmdMeasure* out)) {
  g_amdQuery = query != nullptr ? query : &QueryAmd;
}
void SetBootReturnForTest(std::uintptr_t address) { g_bootReturn.store(address, std::memory_order_relaxed); }
std::uintptr_t LastApplyReturnForTest() { return g_lastApplyReturn.load(std::memory_order_relaxed); }
SettingsObserver::View ObservedForTest() { return g_observer.Snapshot(); }
bool ActiveForTest() { return g_active.load(std::memory_order_acquire); }

void ResetForTest() {
  g_active.store(false, std::memory_order_release);
  if (g_originalApply != nullptr) Hooking::Detach(reinterpret_cast<PVOID*>(&g_originalApply), reinterpret_cast<PVOID>(&ApplyHook));
  if (g_originalSelector != nullptr) {
    Hooking::Detach(reinterpret_cast<PVOID*>(&g_originalSelector), reinterpret_cast<PVOID>(&SelectorHook));
  }
  g_originalApply = nullptr;
  g_originalSelector = nullptr;
  g_collector.reset();
  g_reader.store(&g_processReader, std::memory_order_release);
  g_externals = {&QueryGpuDriver, &QueryRuntimeVersion};
  g_bootReturn.store(0);
  g_gameBase.store(0);
  g_lastApplyReturn.store(0);
  g_swallowed.store(0);
  g_serverSkipLogged.store(false);
  g_checkFailureLogged.store(false);
  g_traceLogged.store(false);
  g_scoreLogged.store(false);
  g_nvidiaQuery = &QueryNvidia;
  g_amdQuery = &QueryAmd;
  g_observer.Reset();
}
#endif

}  // namespace nevr_render_config_hook
