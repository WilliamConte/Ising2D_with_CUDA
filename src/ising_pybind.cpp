#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "IsingModel2d.h"

namespace py = pybind11;

PYBIND11_MODULE(ising2d, m) {
    m.doc() = "CUDA-accelerated 2D Ising Model simulation";

    // 1. Bind the Mode Enum
    py::enum_<Mode>(m, "Mode")
        .value("SERIAL", Mode::serial)
        .value("OPENMP", Mode::openMP)
        .value("CUDA_GLOBAL", Mode::cuda_global)
        .value("CUDA_SHARED", Mode::cuda_shared)
        .export_values();

    // 2. Bind the IsingModel2d Class
    py::class_<IsingModel2d>(m, "IsingModel")
        .def(py::init<int, double, double, double, unsigned int>(),
             py::arg("L"), py::arg("T"), py::arg("J") = 1.0, py::arg("h") = 0.0, py::arg("seed") = 42)
        
        // Main methods
        .def("update", &IsingModel2d::update, py::arg("mode"), py::arg("steps"))
        .def("copy_to_host", &IsingModel2d::copy_to_host)
        .def("copy_to_device", &IsingModel2d::copy_to_device)
        .def("energy", &IsingModel2d::energy, py::arg("mode"))
        .def("magnetization", &IsingModel2d::magnetization, py::arg("mode"))
        .def("device_synchronize", &IsingModel2d::device_synchronize)
        
        // Setters for benchmarking
        .def("set_cuda_block_size", &IsingModel2d::set_cuda_block_size)
        .def("set_num_threads", &IsingModel2d::set_num_threads)

        // The "Zero-Copy" Lattice Access
        // This creates a NumPy array that points to the SAME memory in RAM as your std::vector
        .def_property_readonly("lattice", [](IsingModel2d &self) {
            return py::array_t<int>(
                self.get_shape(),   // {L, L}
                self.get_strides(), // {L*4, 4}
                self.get_data_ptr() // Pointer to lattice.data()
            );
        });
}