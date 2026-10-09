// DllLoadHook's callbacks for a DLL already loaded when they register: during nEVR's boot they wait for
// FireHeldCallbacks() (the game's log isn't usable before the log filter; an injector like Revive's has
// dxgi/d3d12 loaded before nEVR starts), afterwards they fire right away.
#include <gtest/gtest.h>

#include "runtime/hook/dll_load_hook.h"

namespace {

int g_fired = 0;
HMODULE g_module = nullptr;

void Count(const char*, HMODULE module) {
  g_fired++;
  g_module = module;
}

}  // namespace

TEST(DllLoadHook, HoldsALoadedDllsCallbackDuringBootUntilFired) {
  g_fired = 0;
  DllLoadHook::HoldEarlyCallbacks();
  DllLoadHook::OnLoad("KERNEL32.dll", Count);
  EXPECT_EQ(g_fired, 0);

  DllLoadHook::FireHeldCallbacks();
  EXPECT_EQ(g_fired, 1);
  EXPECT_EQ(g_module, GetModuleHandleA("kernel32.dll"));

  // Held callbacks fire once.
  DllLoadHook::FireHeldCallbacks();
  EXPECT_EQ(g_fired, 1);
  DllLoadHook::Shutdown();
}

TEST(DllLoadHook, AfterBootALoadedDllsCallbackFiresRightAway) {
  g_fired = 0;
  DllLoadHook::HoldEarlyCallbacks();
  DllLoadHook::FireHeldCallbacks();
  DllLoadHook::OnLoad("kernel32.dll", Count);
  EXPECT_EQ(g_fired, 1);
  DllLoadHook::Shutdown();
}

TEST(DllLoadHook, ADllNotLoadedYetIsNotFiredByTheHeldOnes) {
  g_fired = 0;
  DllLoadHook::HoldEarlyCallbacks();
  DllLoadHook::OnLoad("nevr-not-a-real-module.dll", Count);
  DllLoadHook::FireHeldCallbacks();
  EXPECT_EQ(g_fired, 0);
  DllLoadHook::Shutdown();
}
