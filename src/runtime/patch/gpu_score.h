#pragma once

// #1419: the GPU score the game's auto-selector reads, filled by the runtime when the game's own rating failed.
//
// The game writes the score during D3D12 device init: NVIDIA cores x base clock (kHz) x 2 x 1e-9 through NVAPI, AMD's AGS
// teraFlops (a float, widened). Under Proton the NVIDIA base-clock query is refused (dxvk-nvapi answers only the current
// clock) and the game's AGS reaches no driver, so the score stays 0 and the selector picks quality level 1 on any GPU. This
// file is the half without the process: the formulas, the decision, and the reader of the Linux host's engine-clock table
// (plain file reads under a root the caller names). The NVAPI and Vulkan queries and the write into the game are in
// patch/render_config_hook.cpp; unit tests feed this file directly (src/runtime/tests/test_gpu_score.cpp).

#include <cstdint>
#include <string>
#include <string_view>

namespace nevr_gpu_score {

/// The vendor codes the game stores and the selector compares (fcn.140c31e00: 1 against 8.0 and 4.5, 2 against 10.0 and
/// 6.0). Device init sets 2 for a DXGI adapter of PCI vendor 0x1002 or 0x1022 (0x14058eabd..0x14058ead8).
inline constexpr std::uint32_t kVendorNvidia = 1;
inline constexpr std::uint32_t kVendorAmd = 2;
/// No real adapter rates above this in the game's units (TFLOPS); a larger value is a misread, not a score.
inline constexpr double kMaxPlausibleScore = 1000.0;

/// What the NVIDIA query found. `failure` names the first step that did not answer, "" when none failed.
struct NvidiaMeasure {
  bool haveCores = false;
  std::uint32_t cores = 0;
  bool haveClockKhz = false;
  std::uint32_t clockKhz = 0;
  const char* failure = "";
};

/// What the AMD query found: compute units from Vulkan, the top engine clock from the host's amdgpu clock table. `failure`
/// names the first step that did not answer, "" when none failed.
struct AmdMeasure {
  bool haveComputeUnits = false;
  std::uint32_t computeUnits = 0;
  bool haveClockMhz = false;
  std::uint32_t clockMhz = 0;
  const char* failure = "";
};

enum class Outcome { Kept, Filled, Unrated };

struct Decision {
  Outcome outcome = Outcome::Unrated;
  double score = 0.0;               // the score the selector should see
  const char* source = "none";      // "game" (kept), "nvidia_nvapi" or "amd_vulkan_sysfs" (filled), "none" (unrated)
  const char* reason = "";          // why unrated; "" otherwise
};

/// The game's NVIDIA score: cores x clock (kHz) x 2 x 1e-9, in the same integer and then double steps the game takes
/// (imul, add to double the product, mulsd by 1e-9).
double NvidiaScore(std::uint32_t cores, std::uint32_t clockKhz);

/// The game's AMD score, AGS's teraFlops: compute units x 64 lanes x 2 (an FMA is two operations) x the top engine clock
/// (MHz) / 1e6, held as a float the way AGS reports it and widened to double the way the game reads it (CVTPS2PD at
/// 0x14058ec7a). The game's own RX 570 Series constant, 5.095, is this formula for 32 CUs at 1244 MHz.
double AmdScore(std::uint32_t computeUnits, std::uint32_t clockMhz);

/// The compute units of an AMD GPU as Vulkan reports them: VK_AMD_shader_core_properties2's active count when it is
/// non-zero, else VK_AMD_shader_core_properties' layout (shader engines x arrays per engine x CUs per array).
std::uint32_t AmdComputeUnits(std::uint32_t activeComputeUnits, std::uint32_t shaderEngines,
                              std::uint32_t arraysPerEngine, std::uint32_t computeUnitsPerArray);

/// The highest clock in amdgpu's pp_dpm_sclk text, one "<level>: <mhz>Mhz" line per level; 0 when no line is one.
std::uint32_t TopSclkMhz(std::string_view ppDpmSclk);

/// Finds, under `drmRoot` (the host's /sys/class/drm, through Wine's Z: drive), the card<N> whose device\device file holds
/// `deviceId` and sets `out`'s clock to the top level of its device\pp_dpm_sclk. Cards 0 to 15 are searched. On no
/// matching card `out->failure` is "host_sclk_card_not_found"; on a matching card with no clock level,
/// "host_sclk_unreadable".
void ReadHostSclk(const std::string& drmRoot, std::uint32_t deviceId, AmdMeasure* out);

/// The score the selector should see. A non-zero game score is kept as it is, whatever the vendor (Windows is untouched).
/// A zero score is filled for NVIDIA with both NVAPI values and for AMD with both AMD values, each with its own vendor's
/// formula, and only when the result is a finite, positive, plausible number; otherwise it stays zero and the reason says
/// which step failed.
Decision DecideScore(std::uint32_t vendor, double gameScore, const NvidiaMeasure& nvidia,
                     const AmdMeasure& amd = AmdMeasure());

const char* OutcomeName(Outcome outcome);

}  // namespace nevr_gpu_score
