#include <iostream>
#include "gapprof/gapprof.hpp"

/*
extern C required as C++ encodes type information in function names
this library is loaded at runtime with the dynamic linker,
resolving symbols that are defined in binary by the C ABI
*/
extern "C" {
    /*
    The __attribute__((constructor)) tells the OS dynamic linker (ld.so) to execute
    this function exactly once when the shared library is loaded into the target process's
    memory space, before the target application's main() function even begins
    */
    __attribute__((constructor)) void EarlyInitialize(void) {
        std::cout << "[GapProf] LD_PRELOAD hooked. Booting telemetry early..." << std::endl;
        gapprof::Profiler::get().initialize();
    }

    /*
    This allows manual initialization due to the behavior of some Python libraries.
    The `is_initialized` flag inside Profiler::initialize() prevents double-initialization
    */
    __attribute__((visibility("default"))) int InitializeInjection(void) {
        std::cout << "[GapProf] Library injected. Initializing telemetry..." << std::endl;
        gapprof::Profiler::get().initialize();
        return 1;
    }

    /*
    Runs when the shared library is unloaded, before static destruction of
    global objects across every library in the process. Avoid static destruction
    order fiasco: https://stackoverflow.com/questions/469597/destruction-order-of-static-objects-in-c
    */
    __attribute__((destructor)) void EarlyShutdown(void) {
        gapprof::Profiler::get().shutdown();
    }
}