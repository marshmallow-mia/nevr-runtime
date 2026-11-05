// #1419: the GPU score rule (patch/gpu_score.h). Pure: no game image, no Windows call.

#include <gtest/gtest.h>

#include <windows.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "runtime/patch/gpu_score.h"

namespace {

using nevr_gpu_score::AmdMeasure;
using nevr_gpu_score::DecideScore;
using nevr_gpu_score::NvidiaMeasure;
using nevr_gpu_score::Outcome;

NvidiaMeasure Measured(std::uint32_t cores, std::uint32_t clockKhz) {
  NvidiaMeasure m;
  m.haveCores = true;
  m.cores = cores;
  m.haveClockKhz = true;
  m.clockKhz = clockKhz;
  return m;
}

AmdMeasure AmdMeasured(std::uint32_t computeUnits, std::uint32_t clockMhz) {
  AmdMeasure m;
  m.haveComputeUnits = true;
  m.computeUnits = computeUnits;
  m.haveClockMhz = true;
  m.clockMhz = clockMhz;
  return m;
}

// The game's arithmetic: 3584 cores at 1,500,000 kHz: 3584 x 1500000 x 2 = 10752000000, x 1e-9 = 10.752.
TEST(GpuScore, NvidiaScoreIsCoresTimesKhzTimesTwoTimesOneE9) {
  EXPECT_DOUBLE_EQ(nevr_gpu_score::NvidiaScore(3584, 1500000), 10.752);
  EXPECT_DOUBLE_EQ(nevr_gpu_score::NvidiaScore(1, 1), 2e-9);
}

TEST(GpuScore, AnAbsurdProductIsRefusedNotStored) {
  // 3584 cores at 1,000,000,000 kHz (a clock of 1 THz): 3584 x 10^9 x 2 x 1e-9 = 7168, over the 1000 ceiling.
  const auto d = DecideScore(nevr_gpu_score::kVendorNvidia, 0.0, Measured(3584, 1000000000U));
  EXPECT_EQ(d.outcome, Outcome::Unrated);
  EXPECT_STREQ(d.reason, "score_implausible");
}

TEST(GpuScore, ANonZeroGameScoreIsKeptWhateverTheVendorOrTheQuery) {
  const auto d = DecideScore(nevr_gpu_score::kVendorNvidia, 6.0, Measured(3584, 1500000));
  EXPECT_EQ(d.outcome, Outcome::Kept);
  EXPECT_DOUBLE_EQ(d.score, 6.0);
  EXPECT_STREQ(d.source, "game");
  EXPECT_STREQ(d.reason, "");

  const auto amd = DecideScore(2, 10.0, NvidiaMeasure());
  EXPECT_EQ(amd.outcome, Outcome::Kept);
  EXPECT_DOUBLE_EQ(amd.score, 10.0);
}

TEST(GpuScore, ANvidiaZeroScoreWithBothValuesIsFilledWithTheFormula) {
  const auto d = DecideScore(nevr_gpu_score::kVendorNvidia, 0.0, Measured(3584, 1500000));
  EXPECT_EQ(d.outcome, Outcome::Filled);
  EXPECT_DOUBLE_EQ(d.score, 10.752);
  EXPECT_STREQ(d.source, "nvidia_nvapi");
  EXPECT_STREQ(d.reason, "");
}

TEST(GpuScore, AZeroScoreOfAnotherVendorStaysUnratedWithTheReason) {
  // The selector rates only vendors 1 and 2; any other code keeps its zero, whatever was measured.
  const auto d = DecideScore(3, 0.0, Measured(3584, 1500000), AmdMeasured(72, 2250));
  EXPECT_EQ(d.outcome, Outcome::Unrated);
  EXPECT_DOUBLE_EQ(d.score, 0.0);
  EXPECT_STREQ(d.reason, "vendor_unsupported");
}

TEST(GpuScore, AMissingNvapiValueNamesTheStepThatFailed) {
  NvidiaMeasure noCores;
  noCores.haveClockKhz = true;
  noCores.clockKhz = 1500000;
  noCores.failure = "nvapi_get_gpu_core_count_failed";
  const auto a = DecideScore(nevr_gpu_score::kVendorNvidia, 0.0, noCores);
  EXPECT_EQ(a.outcome, Outcome::Unrated);
  EXPECT_STREQ(a.reason, "nvapi_get_gpu_core_count_failed");

  NvidiaMeasure noClock;
  noClock.haveCores = true;
  noClock.cores = 3584;
  const auto b = DecideScore(nevr_gpu_score::kVendorNvidia, 0.0, noClock);
  EXPECT_EQ(b.outcome, Outcome::Unrated);
  EXPECT_STREQ(b.reason, "nvapi_clock_unavailable");
}

TEST(GpuScore, AZeroOrAbsurdNvidiaResultStaysUnrated) {
  const auto zero = DecideScore(nevr_gpu_score::kVendorNvidia, 0.0, Measured(0, 1500000));
  EXPECT_EQ(zero.outcome, Outcome::Unrated);
  EXPECT_STREQ(zero.reason, "score_implausible");

  const auto absurd = DecideScore(nevr_gpu_score::kVendorNvidia, 0.0, Measured(3584, 1000000000U));
  EXPECT_EQ(absurd.outcome, Outcome::Unrated);
  EXPECT_STREQ(absurd.reason, "score_implausible");
}

TEST(GpuScore, ANanGameScoreIsKeptAsItIs) {
  // A NaN is not zero, so the game's own value is never overwritten by the runtime.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto d = DecideScore(nevr_gpu_score::kVendorNvidia, nan, Measured(3584, 1500000));
  EXPECT_EQ(d.outcome, Outcome::Kept);
  EXPECT_TRUE(std::isnan(d.score));
}

// ---- AMD (vendor 2): AGS's teraFlops, which the game widens from a float (cvtps2pd at 0x14058ec7a) ----

// AGS rates CUs x 64 lanes x 2 (an FMA is two operations) x the top engine clock in MHz, / 1e6 = TFLOPS, as a float.
// 72 CUs at 2250 MHz: 72 x 64 x 2 x 2250 = 20736000, / 1e6 = 20.736. 36 CUs at 1340 MHz: 6174720 -> 6.17472.
TEST(GpuScore, AmdScoreIsAgsTeraFlopsAsTheGameReadsIt) {
  EXPECT_DOUBLE_EQ(nevr_gpu_score::AmdScore(72, 2250), static_cast<double>(20.736F));
  EXPECT_DOUBLE_EQ(nevr_gpu_score::AmdScore(36, 1340), static_cast<double>(6.17472F));
  EXPECT_DOUBLE_EQ(nevr_gpu_score::AmdScore(1, 1), static_cast<double>(0.000128F));
}

// The game's own constant for an RX 570 Series that AGS rates 0 is 5.095 (0x141c39e80, stored at 0x14058ecb3). An RX 570
// has 32 CUs and a top engine clock of 1244 MHz: 32 x 64 x 2 x 1244 / 1e6 = 5.095424. The game's units are this formula's.
TEST(GpuScore, AmdScoreMatchesTheGamesOwnRx570Constant) {
  EXPECT_NEAR(nevr_gpu_score::AmdScore(32, 1244), 5.095, 0.0005);
}

TEST(GpuScore, AnAmdZeroScoreWithBothValuesIsFilledWithTheFormula) {
  const auto d = DecideScore(nevr_gpu_score::kVendorAmd, 0.0, NvidiaMeasure(), AmdMeasured(72, 2250));
  EXPECT_EQ(d.outcome, Outcome::Filled);
  EXPECT_DOUBLE_EQ(d.score, static_cast<double>(20.736F));
  EXPECT_STREQ(d.source, "amd_vulkan_sysfs");
  EXPECT_STREQ(d.reason, "");
}

TEST(GpuScore, ANonZeroAmdGameScoreIsKeptAndTheMeasureIgnored) {
  // Windows: AGS answered, the game's value stands.
  const auto d = DecideScore(nevr_gpu_score::kVendorAmd, 12.5, NvidiaMeasure(), AmdMeasured(72, 2250));
  EXPECT_EQ(d.outcome, Outcome::Kept);
  EXPECT_DOUBLE_EQ(d.score, 12.5);
  EXPECT_STREQ(d.source, "game");
}

TEST(GpuScore, AnAmdMeasureDoesNotRateNvidiaNorTheReverse) {
  const auto nvidia = DecideScore(nevr_gpu_score::kVendorNvidia, 0.0, NvidiaMeasure(), AmdMeasured(72, 2250));
  EXPECT_EQ(nvidia.outcome, Outcome::Unrated);
  EXPECT_STREQ(nvidia.reason, "nvapi_cores_unavailable");
  const auto amd = DecideScore(nevr_gpu_score::kVendorAmd, 0.0, Measured(3584, 1500000), AmdMeasure());
  EXPECT_EQ(amd.outcome, Outcome::Unrated);
  EXPECT_STREQ(amd.reason, "amd_compute_units_unavailable");
}

TEST(GpuScore, AMissingAmdValueNamesTheStepThatFailed) {
  AmdMeasure noUnits;
  noUnits.failure = "vulkan_absent";
  const auto a = DecideScore(nevr_gpu_score::kVendorAmd, 0.0, NvidiaMeasure(), noUnits);
  EXPECT_EQ(a.outcome, Outcome::Unrated);
  EXPECT_DOUBLE_EQ(a.score, 0.0);
  EXPECT_STREQ(a.reason, "vulkan_absent");

  AmdMeasure noClock;
  noClock.haveComputeUnits = true;
  noClock.computeUnits = 72;
  const auto b = DecideScore(nevr_gpu_score::kVendorAmd, 0.0, NvidiaMeasure(), noClock);
  EXPECT_EQ(b.outcome, Outcome::Unrated);
  EXPECT_STREQ(b.reason, "amd_clock_unavailable");

  noClock.failure = "host_sclk_card_not_found";
  const auto c = DecideScore(nevr_gpu_score::kVendorAmd, 0.0, NvidiaMeasure(), noClock);
  EXPECT_STREQ(c.reason, "host_sclk_card_not_found");
}

TEST(GpuScore, AZeroOrAbsurdAmdResultStaysUnrated) {
  const auto zero = DecideScore(nevr_gpu_score::kVendorAmd, 0.0, NvidiaMeasure(), AmdMeasured(0, 2250));
  EXPECT_EQ(zero.outcome, Outcome::Unrated);
  EXPECT_STREQ(zero.reason, "score_implausible");

  // 72 CUs at 1,000,000 MHz (1 THz): 72 x 128 x 10^6 / 1e6 = 9216 TFLOPS, over the 1000 ceiling.
  const auto absurd = DecideScore(nevr_gpu_score::kVendorAmd, 0.0, NvidiaMeasure(), AmdMeasured(72, 1000000U));
  EXPECT_EQ(absurd.outcome, Outcome::Unrated);
  EXPECT_DOUBLE_EQ(absurd.score, 0.0);
  EXPECT_STREQ(absurd.reason, "score_implausible");
}

// VK_AMD_shader_core_properties2 gives the active CU count (harvested parts have fewer than the layout); without it the
// count is the layout: shader engines x arrays per engine x CUs per array.
TEST(GpuScore, AmdComputeUnitsPreferTheActiveCount) {
  EXPECT_EQ(nevr_gpu_score::AmdComputeUnits(60, 4, 2, 10), 60U);
  EXPECT_EQ(nevr_gpu_score::AmdComputeUnits(0, 4, 2, 10), 80U);
  EXPECT_EQ(nevr_gpu_score::AmdComputeUnits(0, 0, 0, 0), 0U);
}

// amdgpu's pp_dpm_sclk: one "<level>: <mhz>Mhz" line per engine clock level, the current one marked with '*'.
TEST(GpuScore, TopSclkIsTheHighestLevelLine) {
  EXPECT_EQ(nevr_gpu_score::TopSclkMhz("0: 500Mhz \n1: 1800Mhz *\n2: 2250Mhz \n"), 2250U);
  EXPECT_EQ(nevr_gpu_score::TopSclkMhz("0: 2575Mhz\n1: 300Mhz *\n"), 2575U);
  EXPECT_EQ(nevr_gpu_score::TopSclkMhz("0: 700MHz\n"), 700U);
}

TEST(GpuScore, TopSclkIgnoresLinesThatAreNotClockLevels) {
  EXPECT_EQ(nevr_gpu_score::TopSclkMhz(""), 0U);
  EXPECT_EQ(nevr_gpu_score::TopSclkMhz("0: 2250\n"), 0U);            // no unit
  EXPECT_EQ(nevr_gpu_score::TopSclkMhz("OD_SCLK: 9999Mhz\n"), 0U);   // not a level number
  EXPECT_EQ(nevr_gpu_score::TopSclkMhz("garbage\n1: 900Mhz\n"), 900U);
  EXPECT_EQ(nevr_gpu_score::TopSclkMhz("0: 99999kHz\n1: 900Mhz\n"), 900U);  // another unit
}

// A fake host /sys/class/drm in a temporary directory: card<N>\device\device holds the PCI device id, pp_dpm_sclk the levels.
class HostSclk : public ::testing::Test {
 protected:
  void SetUp() override {
    char temp[MAX_PATH] = {};
    ASSERT_GT(GetTempPathA(sizeof(temp), temp), 0U);
    root_ = std::string(temp) + "nevr_gpu_score_" + std::to_string(GetCurrentProcessId()) + "_" +
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
    ASSERT_TRUE(CreateDirectoryA(root_.c_str(), nullptr) || GetLastError() == ERROR_ALREADY_EXISTS);
  }
  void TearDown() override {
    for (const std::string& file : files_) DeleteFileA(file.c_str());
    for (auto it = dirs_.rbegin(); it != dirs_.rend(); ++it) RemoveDirectoryA(it->c_str());
    RemoveDirectoryA(root_.c_str());
  }
  void Card(int index, const char* deviceFile, const char* sclkFile) {
    const std::string card = root_ + "\\card" + std::to_string(index);
    const std::string device = card + "\\device";
    CreateDirectoryA(card.c_str(), nullptr);
    CreateDirectoryA(device.c_str(), nullptr);
    dirs_.push_back(card);
    dirs_.push_back(device);
    if (deviceFile != nullptr) Write(device + "\\device", deviceFile);
    if (sclkFile != nullptr) Write(device + "\\pp_dpm_sclk", sclkFile);
  }
  void Write(const std::string& path, const char* text) {
    FILE* f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr) << path;
    std::fputs(text, f);
    std::fclose(f);
    files_.push_back(path);
  }
  std::string root_;
  std::vector<std::string> dirs_;
  std::vector<std::string> files_;
};

TEST_F(HostSclk, TheCardWithTheDevicesIdGivesItsTopClock) {
  Card(0, "0x164e\n", "0: 400Mhz\n1: 2200Mhz *\n");  // an integrated GPU first
  Card(1, "0x73bf\n", "0: 500Mhz *\n1: 2250Mhz\n");
  AmdMeasure m;
  nevr_gpu_score::ReadHostSclk(root_, 0x73bf, &m);
  EXPECT_TRUE(m.haveClockMhz);
  EXPECT_EQ(m.clockMhz, 2250U);
  EXPECT_STREQ(m.failure, "");
}

TEST_F(HostSclk, NoCardWithTheDevicesIdIsNamed) {
  Card(0, "0x164e\n", "0: 2200Mhz\n");
  AmdMeasure m;
  nevr_gpu_score::ReadHostSclk(root_, 0x73bf, &m);
  EXPECT_FALSE(m.haveClockMhz);
  EXPECT_STREQ(m.failure, "host_sclk_card_not_found");
}

TEST_F(HostSclk, AMatchingCardWithoutAClockTableIsNamed) {
  Card(2, "0x73bf\n", nullptr);
  AmdMeasure m;
  nevr_gpu_score::ReadHostSclk(root_, 0x73bf, &m);
  EXPECT_FALSE(m.haveClockMhz);
  EXPECT_STREQ(m.failure, "host_sclk_unreadable");
}

TEST_F(HostSclk, AnAbsentRootIsNamed) {
  AmdMeasure m;
  nevr_gpu_score::ReadHostSclk(root_ + "\\absent", 0x73bf, &m);
  EXPECT_FALSE(m.haveClockMhz);
  EXPECT_STREQ(m.failure, "host_sclk_card_not_found");
}

}  // namespace
