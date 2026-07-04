#include <pybind11/pybind11.h>
#include "compute_sanitizer.h"

#include <iostream>
#include <cstdlib>

static bool pyaccelprof_initialized = false;

void accelprof_init() {
    pyaccelprof_initialized = true;
    enable_compute_sanitizer(false);
}

void accelprof_start() {
    if (!pyaccelprof_initialized) {
        std::cerr << "error: PyCUProf is not initialized. Please call accelprof_init() first." << std::endl;
        std::exit(1);
    }
    std::cout << "[PYCUPROF INFO] Start CUProf profiling" << std::endl;
    std::cout.flush();
    enable_compute_sanitizer(true);
}

void accelprof_stop() {
    if (!pyaccelprof_initialized) {
        std::cerr << "error: PyCUProf is not initialized. Please call accelprof_init() first." << std::endl;
        std::exit(1);
    }
    std::cout << "[PYCUPROF INFO] Stop CUProf profiling" << std::endl;
    std::cout.flush();
    enable_compute_sanitizer(false);
}

PYBIND11_MODULE(pyaccelprof, m) {
    m.doc() = "Python interface of AccelProf";

    // Call accelprof_init automatically during import
    accelprof_init();

    // m.def("init", &accelprof_init, "Initialize CUProf (must be called at the very beginning)");
    m.def("start", &accelprof_start, "Start AccelProf, start profiling");
    m.def("stop", &accelprof_stop, "Stop AccelProf, stop profiling");
}
