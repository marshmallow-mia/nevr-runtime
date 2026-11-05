#include "runtime/patch/gpu_score.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace nevr_gpu_score {

double NvidiaScore(std::uint32_t cores, std::uint32_t clockKhz) {
  const std::uint64_t product = static_cast<std::uint64_t>(cores) * clockKhz * 2;
  return static_cast<double>(product) * 1e-9;
}

namespace {

// AGS: 64 lanes per compute unit, 2 operations per FMA.
constexpr std::uint64_t kAmdOpsPerComputeUnitClock = 64 * 2;
constexpr int kMaxDrmCards = 16;

bool ReadText(const std::string& path, std::string* out) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return false;
  std::ostringstream text;
  text << file.rdbuf();
  *out = text.str();
  return true;
}

bool Plausible(double score) { return std::isfinite(score) && score > 0.0 && score <= kMaxPlausibleScore; }

}  // namespace

std::uint32_t CudaCoresPerSm(int major, int minor) {
  if (major < 3) return 0;
  if (major == 3) return 192;
  if (major == 6 && minor == 0) return 64;
  if (major == 7) return 64;
  if (major == 8 && minor == 0) return 64;
  return 128;
}

std::uint32_t BaseClockKhzFromMaxMhz(std::uint32_t maxClockMhz) { return maxClockMhz * 620U; }

NvidiaMeasure MeasureNvidia(const NvidiaAnswers& answers) {
  NvidiaMeasure m;
  if (answers.nvapiCoresOk && answers.nvapiCores != 0) {
    m.haveCores = true;
    m.cores = answers.nvapiCores;
  } else if (!answers.vulkanSmsOk || answers.vulkanSms == 0) {
    m.failure = answers.vulkanFailure[0] != '\0' ? answers.vulkanFailure : "vulkan_no_sm_count";
  } else if (!answers.nvmlOk) {
    m.failure = answers.nvmlFailure[0] != '\0' ? answers.nvmlFailure : "nvml_unavailable";
  } else if (CudaCoresPerSm(answers.cudaMajor, answers.cudaMinor) == 0) {
    m.failure = "nvml_cuda_capability_unknown";
  } else {
    m.haveCores = true;
    m.cores = answers.vulkanSms * CudaCoresPerSm(answers.cudaMajor, answers.cudaMinor);
    m.coresFromVulkan = true;
  }
  if (!m.haveCores) return m;
  if (!answers.nvmlOk || answers.maxClockMhz == 0) {
    m.failure = answers.nvmlFailure[0] != '\0' ? answers.nvmlFailure : "nvml_max_clock_unavailable";
    return m;
  }
  m.haveClockKhz = true;
  m.clockKhz = BaseClockKhzFromMaxMhz(answers.maxClockMhz);
  return m;
}

double AmdScore(std::uint32_t computeUnits, std::uint32_t clockMhz) {
  const std::uint64_t opsPerMicrosecond = static_cast<std::uint64_t>(computeUnits) * kAmdOpsPerComputeUnitClock * clockMhz;
  const float teraFlops = static_cast<float>(static_cast<double>(opsPerMicrosecond) / 1e6);
  return static_cast<double>(teraFlops);
}

std::uint32_t AmdComputeUnits(std::uint32_t activeComputeUnits, std::uint32_t shaderEngines,
                              std::uint32_t arraysPerEngine, std::uint32_t computeUnitsPerArray) {
  if (activeComputeUnits != 0) return activeComputeUnits;
  return shaderEngines * arraysPerEngine * computeUnitsPerArray;
}

std::uint32_t TopSclkMhz(std::string_view ppDpmSclk) {
  std::uint32_t top = 0;
  std::size_t start = 0;
  while (start < ppDpmSclk.size()) {
    std::size_t end = ppDpmSclk.find('\n', start);
    if (end == std::string_view::npos) end = ppDpmSclk.size();
    const std::string line(ppDpmSclk.substr(start, end - start));
    start = end + 1;
    unsigned level = 0;
    unsigned mhz = 0;
    char unit[4] = {};
    if (std::sscanf(line.c_str(), "%u: %u%3c", &level, &mhz, unit) != 3) continue;
    if ((unit[0] != 'M' && unit[0] != 'm') || (unit[1] != 'h' && unit[1] != 'H') || (unit[2] != 'z' && unit[2] != 'Z')) continue;
    if (mhz > top) top = mhz;
  }
  return top;
}

void ReadHostSclk(const std::string& drmRoot, std::uint32_t deviceId, AmdMeasure* out) {
  for (int card = 0; card < kMaxDrmCards; ++card) {
    const std::string device = drmRoot + "\\card" + std::to_string(card) + "\\device";
    std::string idText;
    if (!ReadText(device + "\\device", &idText)) continue;
    if (std::strtoul(idText.c_str(), nullptr, 16) != deviceId) continue;
    std::string sclk;
    const std::uint32_t mhz = ReadText(device + "\\pp_dpm_sclk", &sclk) ? TopSclkMhz(sclk) : 0;
    if (mhz == 0) {
      out->failure = "host_sclk_unreadable";
      return;
    }
    out->haveClockMhz = true;
    out->clockMhz = mhz;
    return;
  }
  out->failure = "host_sclk_card_not_found";
}

Decision DecideScore(std::uint32_t vendor, double gameScore, const NvidiaMeasure& nvidia, const AmdMeasure& amd) {
  Decision decision;
  if (gameScore != 0.0) {
    decision.outcome = Outcome::Kept;
    decision.score = gameScore;
    decision.source = "game";
    return decision;
  }
  if (vendor == kVendorAmd) {
    if (!amd.haveComputeUnits) {
      decision.reason = amd.failure[0] != '\0' ? amd.failure : "amd_compute_units_unavailable";
      return decision;
    }
    if (!amd.haveClockMhz) {
      decision.reason = amd.failure[0] != '\0' ? amd.failure : "amd_clock_unavailable";
      return decision;
    }
    const double score = AmdScore(amd.computeUnits, amd.clockMhz);
    if (!Plausible(score)) {
      decision.reason = "score_implausible";
      return decision;
    }
    decision.outcome = Outcome::Filled;
    decision.score = score;
    decision.source = "amd_vulkan_sysfs";
    return decision;
  }
  if (vendor != kVendorNvidia) {
    decision.reason = "vendor_unsupported";
    return decision;
  }
  if (!nvidia.haveCores) {
    decision.reason = nvidia.failure[0] != '\0' ? nvidia.failure : "nvidia_cores_unavailable";
    return decision;
  }
  if (!nvidia.haveClockKhz) {
    decision.reason = nvidia.failure[0] != '\0' ? nvidia.failure : "nvidia_clock_unavailable";
    return decision;
  }
  const double score = NvidiaScore(nvidia.cores, nvidia.clockKhz);
  if (!Plausible(score)) {
    decision.reason = "score_implausible";
    return decision;
  }
  decision.outcome = Outcome::Filled;
  decision.score = score;
  decision.source = nvidia.coresFromVulkan ? "nvidia_vulkan_nvml" : "nvidia_nvapi_nvml";
  return decision;
}

const char* OutcomeName(Outcome outcome) {
  switch (outcome) {
    case Outcome::Kept: return "kept";
    case Outcome::Filled: return "filled";
    case Outcome::Unrated: return "unrated";
  }
  return "unrated";
}

}  // namespace nevr_gpu_score
