#include <CL/sycl.hpp>


#include <iostream>

class SampleSelector : public sycl::device_selector {
    public:
    int operator()(const sycl::device &Device) const override {
        using namespace sycl::info;
        auto DeviceName = Device.get_info<device::name>();

        return DeviceName.find("FPGA") != std::string::npos; 
    }
};

int main() {
    SampleSelector selector;
    try {
        sycl::queue q(selector);
        auto dev = q.get_device();
        std::cout << dev.get_info<sycl::info::device::name>() << std::endl;
    } catch (...) {
        std::cerr << "Err!";
    }
}