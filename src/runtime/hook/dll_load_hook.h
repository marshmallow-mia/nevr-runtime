/* SYNTHESIS -- custom tool code, not from binary */
#pragma once

#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#endif

namespace DllLoadHook {

/* Callback signature: called after a DLL loads successfully.
 * dll_name: lowercase filename (e.g., "pnsdemo.dll")
 * module: the loaded HMODULE
 * Return: ignored. */
typedef void (*PatchCallback)(const char* dll_name, HMODULE module);

void Install();
void Shutdown();

/* Register a callback to fire when a DLL matching `dll_name` loads.
 * dll_name is matched case-insensitively against the filename only (not path).
 * The callback fires once per load. Multiple callbacks per DLL are supported. */
void OnLoad(const char* dll_name, PatchCallback callback);

/* While nEVR boots, a callback for a DLL that is already loaded when it registers is held
 * instead of fired: an injector such as Revive's loads dxgi/d3d12 before nEVR starts, and the
 * callbacks log, which before the log filter goes through the game's CLog, not set up yet this
 * early (an access violation inside echovr.exe). Install() starts holding them;
 * FireHeldCallbacks(), once the log filter is in, fires what was held and stops holding. */
void HoldEarlyCallbacks();
void FireHeldCallbacks();

/* Trigger callbacks for a DLL that was loaded through a non-LoadLibrary path
 * (e.g., the game's CSysDLL_Load). Only fires each callback once per DLL. */
void FireCallbacksForModule(const char* lower_name, HMODULE module);

} // namespace DllLoadHook
