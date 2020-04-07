#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <CL/sycl.hpp>

#include <functional>

class Sample;

int main(int argc, char* argv[]) {

  int inputWidth, inputHeight, inputChannels;
  constexpr int numChannels = 4;

  void* rawImage = stbi_load(argv[1], &inputWidth, &inputHeight, &inputChannels,
                        numChannels);

  if (!rawImage) {
    return -1;
  }

  sycl::range<2> imgRange{static_cast<size_t>(inputHeight), static_cast<size_t>(inputWidth)};

  sycl::image<2> inpImg{rawImage, sycl::image_channel_order::rgba, sycl::image_channel_type::unorm_int8, imgRange};
  sycl::buffer<sycl::float4, 2> outpImage{imgRange};

  sycl::id<2> offset{0, 0};


  //q.submit([&](sycl::handler& cgh) {
  auto KernelCGH = [&](sycl::range<2> range, sycl::id<2> offset, sycl::handler& cgh) {
      auto inpAcc = inpImg.get_access<sycl::float4, sycl::access::mode::read>(cgh);
      auto outAcc = outpImage.get_access<sycl::access::mode::write>(cgh, range, offset);

      sycl::sampler smpl(sycl::coordinate_normalization_mode::unnormalized,
                   sycl::addressing_mode::clamp_to_edge, sycl::filtering_mode::nearest);

        constexpr int radius = 8;

      cgh.parallel_for<Sample>(range, offset, [=](sycl::item<2> item) {
          auto id = item.get_id();
          auto offset = item.get_offset();

          sycl::float4 sum{};
          for (int x = -radius; x < radius; x++) {
              for (int y = -radius; y < radius; y++) {
                  int realX = static_cast<int>(id[1]) + x;
                  int realY = static_cast<int>(id[0]) + y;
                  sum += inpAcc.read(sycl::int2{realX, realY}, smpl); 
              }
          }

          outAcc[id - offset] = (sum / ((2 * radius + 1) * (2 * radius + 1)));
      });
  };

  try {
    sycl::queue q1{sycl::cpu_selector()};
    sycl::queue q2{sycl::cpu_selector()};
    q1.submit(std::bind(KernelCGH, sycl::range<2>(inputHeight / 2, inputWidth), sycl::id<2>{0, 0}, std::placeholders::_1));
    q2.submit(std::bind(KernelCGH, sycl::range<2>(inputHeight / 2, inputWidth), sycl::id<2>(inputHeight / 2, 0), std::placeholders::_1));

    q1.wait_and_throw();
    q2.wait_and_throw();
  } catch (sycl::exception e) {
      std::cerr << e.what();
  }


  auto rawOut = new unsigned char[inputWidth * inputHeight * numChannels];

  {
    auto acc = outpImage.get_access<sycl::access::mode::read>();
    for (size_t x = 0; x < inputHeight; x++) {
        for (size_t y = 0; y < inputWidth; y++) {
            unsigned char r, g, b;
            auto pixel = sycl::clamp(acc[sycl::id<2>{x, y}] * 255.f, 0.f, 255.f);
            r = static_cast<unsigned char>(pixel.r());
            g = static_cast<unsigned char>(pixel.g());
            b = static_cast<unsigned char>(pixel.b());

            rawOut[x * inputHeight * numChannels + numChannels * y] = r;
            rawOut[x * inputHeight * numChannels + numChannels * y + 1] = g;
            rawOut[x * inputHeight * numChannels + numChannels * y + 2] = b;
            rawOut[x * inputHeight * numChannels + numChannels * y + 3] = 255;
        }
    }
  }


  stbi_write_png("blurred.png", inputWidth, inputHeight, numChannels,
                 rawOut, 0);

  return 0;
}