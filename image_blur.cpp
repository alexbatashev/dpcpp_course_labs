#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <CL/sycl.hpp>
#include <functional>

int main() {
    int w, h, c;

    std::string filename = "image.png";
    unsigned char* image = stbi_load(filename.c_str(), &w, &h, &c, 3);

    size_t width = w, height = h, comp = c;

    if (image == nullptr) {
        std::terminate();
    }


    sycl::buffer<sycl::float3, 2> imgBuf{sycl::range<2>{width, height}};
    sycl::buffer<sycl::float3, 2> outBuf{sycl::range<2>{width, height}};
    { 
	    auto imageData = imgBuf.get_access<sycl::access::mode::discard_write>();
	    for (size_t i = 0; i < width*height; i++) {
	      float r, g, b;
	      r = static_cast<float>(image[3*i]) / 255.f;
	      g = static_cast<float>(image[3*i + 1]) / 255.f;
	      b = static_cast<float>(image[3*i + 2]) / 255.f;
	      imageData.get_pointer()[i] = sycl::float3{r, g, b};
	    }
    }

    stbi_image_free(image);

    std::cout << width << std::endl;
    std::cout << height << std::endl;

    auto kernelCG = [&](sycl::range<2> range, sycl::id<2> offset, sycl::handler& cgh) {
        auto inpAcc = imgBuf.get_access<sycl::access::mode::read>(cgh); 
        auto outAcc = outBuf.get_access<sycl::access::mode::discard_write>(cgh, range, offset);
	cgh.parallel_for<class ImageBlur>(range, offset, [=](sycl::item<2> item) {
	    constexpr int KerSize = 3;
	    const auto id = item.get_id();
	    sycl::float3 sum{};
	    for (int x = -KerSize / 2; x < KerSize / 2; x++) {
	      for (int y = -KerSize / 2; y < KerSize / 2; y++) {
		size_t realX = sycl::clamp(static_cast<int>(id[0]) + x, 0, static_cast<int>(item.get_range(0) - 1));
		size_t realY = sycl::clamp(static_cast<int>(id[1]) + y, 0, static_cast<int>(item.get_range(1) - 1));
		sum += inpAcc[sycl::id<2>{realX, realY}];
	      }
	    }     
	    outAcc[id] = sum / (KerSize * KerSize);
	});
    };

    try {
      sycl::queue q1{sycl::cpu_selector{}};
      sycl::queue q2{sycl::host_selector{}};

      auto cgh1 = std::bind(kernelCG, sycl::range<2>{width/2, height}, sycl::id<2>{0, 0}, std::placeholders::_1);
      auto cgh2 = std::bind(kernelCG, sycl::range<2>{width/2, height}, sycl::id<2>{width/2, 0}, std::placeholders::_1);
      q1.submit(cgh1);
      q1.submit(cgh2);

      q1.wait_and_throw();
      q2.wait_and_throw();
    } catch (sycl::exception e) {
      std::cerr << "Exception caught" << std::endl;
      std::cerr << e.what() << std::endl; 
    }

    return 0;
}
