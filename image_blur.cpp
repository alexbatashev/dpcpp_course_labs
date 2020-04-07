#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <CL/sycl.hpp>
#include <functional>

constexpr int radius = 3;

int main() {
  int w, h, c;

  std::string filename = "image.png";
  unsigned char *image = stbi_load(filename.c_str(), &w, &h, &c, 4);

  size_t width = w, height = h, comp = c;

  if (image == nullptr) {
    std::terminate();
  }

  //sycl::buffer<sycl::float3, 2> imgBuf{sycl::range<2>{width, height}};
  sycl::range<2> imgRange{height, width};
  sycl::image<2> imgBuf{image, sycl::image_channel_order::rgba,
   sycl::image_channel_type::unorm_int8, 
   imgRange};
  sycl::buffer<sycl::float4, 2> outBuf{imgRange};
  /*{
    auto imageData = imgBuf.get_access<sycl::access::mode::discard_write>();
    for (size_t i = 0; i < width * height; i++) {
      float r, g, b;
      r = static_cast<float>(image[3 * i]) / 255.f;
      g = static_cast<float>(image[3 * i + 1]) / 255.f;
      b = static_cast<float>(image[3 * i + 2]) / 255.f;
      imageData.get_pointer()[i] = sycl::float3{r, g, b};
    }
  }*/

  stbi_image_free(image);

  std::cout << width << std::endl;
  std::cout << height << std::endl;

  

  auto kernelCG = [&](sycl::range<2> range, sycl::id<2> offset,
                      sycl::handler &cgh) {
    auto inpAcc = imgBuf.get_access<sycl::float4, sycl::access::mode::read>(cgh);
    auto outAcc = outBuf.get_access<sycl::access::mode::discard_write>(
        cgh, range, offset);
    sycl::sampler s{sycl::coordinate_normalization_mode::unnormalized, sycl::addressing_mode::clamp, sycl::filtering_mode::nearest};
    cgh.parallel_for<class ImageBlur>(range, offset, [=](sycl::item<2> item) {
      const auto id = item.get_id();
      sycl::float4 sum{};
      for (int x = -radius; x < radius; x++) {
        for (int y = -radius; y < radius; y++) {
          /*size_t realX = sycl::clamp(static_cast<int>(id[0]) + x, 0,
                                     static_cast<int>(item.get_range(0) - 1));
          size_t realY = sycl::clamp(static_cast<int>(id[1]) + y, 0,
                                     static_cast<int>(item.get_range(1) - 1));
          sum += inpAcc[sycl::id<2>{realX, realY}];*/
          int realX = static_cast<int>(id[1]) + x;
          int realY = static_cast<int>(id[0]) + y;

          sum += inpAcc.read(sycl::int2{realX, realY}, s);
        }
      }
      outAcc[id - item.get_offset()] = sum / (4 * radius * radius);
    });
  };

  try {
    //sycl::queue q1{sycl::cpu_selector{}};
    sycl::queue q2{sycl::host_selector{}};

    //auto cgh1 = std::bind(kernelCG, sycl::range<2>{width / 2, height},
    //                      sycl::id<2>{0, 0}, std::placeholders::_1);
    auto cgh2 = std::bind(kernelCG, sycl::range<2>{height/2, width},
                          sycl::id<2>{0, 0}, std::placeholders::_1);
    //q1.submit(cgh1);
    q2.submit(cgh2);

    //q1.wait_and_throw();
    q2.wait_and_throw();
  } catch (sycl::exception e) {
    std::cerr << "Exception caught" << std::endl;
    std::cerr << e.what() << std::endl;
  }

  typedef unsigned char uc4 __attribute__((ext_vector_type(4)));
  auto rawOutData = new unsigned char[width * height];
  {
    auto outAcc = outBuf.get_access<sycl::access::mode::read>();
    for (size_t x = 0; x < width; x++) {
      for (size_t y = 0; y < height; y++) {
        unsigned char r, g, b;
        auto pixel = outAcc[sycl::id<2>{x, y}];

        r = static_cast<unsigned char>(pixel.r() * 255);
        g = static_cast<unsigned char>(pixel.g() * 255);
        b = static_cast<unsigned char>(pixel.b() * 255);

        uc4 out{r, g, b, 255};

        rawOutData[y * width + x] = r;
        std::cout << static_cast<int>(r) << std::endl;
        //rawOutData[y * width * 4 + x + 1] = g;
        //rawOutData[y * width * 4 + x + 2] = b;
        //rawOutData[y * width * 4 + x + 3] = 255;
      }
    }
  }

  stbi_write_png("blurred.png", w, h, 1, rawOutData, 0);

  return 0;
}
