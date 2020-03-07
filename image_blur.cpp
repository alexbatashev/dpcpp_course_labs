#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <CL/sycl.hpp>

template <typename T, int KerSize>
class ImageBlurKernel {
private:
  using InpAccT = sycl::accessor<T, 2, sycl::access::mode::read>; 
  using OutAccT = sycl::accessor<T, 2, sycl::access::mode::discard_write>; 
  const size_t mImgWidth, mImgHeight;
  InpAccT mInpImg;
  OutAccT mOutImg;
public:
  ImageBlurKernel(size_t w, size_t h, InpAccT inpAcc, OutAccT outAcc) : 
      mImgWidth(w), mImgHeight(h), mInpImg(inpAcc), mOutImg(outAcc) {}

  void operator()(sycl::item<2> item) {
    const auto id = item.get_id();
    T sum{};
    for (int x = -KerSize / 2; x < KerSize / 2; x++) {
      for (int y = -KerSize / 2; y < KerSize / 2; y++) {
        size_t realX = sycl::clamp(static_cast<int>(id[0]) + x, 0, static_cast<int>(mImgWidth - 1));
        size_t realY = sycl::clamp(static_cast<int>(id[1]) + y, 0, static_cast<int>(mImgHeight - 1));
        sum += mInpImg[sycl::id<2>{realX, realY}];
      }
    }     
    mOutImg[id - item.get_offset()] = sum / (KerSize * KerSize);
  }
};

int main() {
    int w, h, c;

    std::string filename = "image.png";
    unsigned char* image = stbi_load(filename.c_str(), &w, &h, &c, 3);

    size_t width = w, height = h, comp = c;

    if (image == nullptr) {
        std::terminate();
    }

    sycl::float3* imageData = new sycl::float3[width*height];

    for (size_t i = 0; i < width*height; i++) {
      float r, g, b;
      r = static_cast<float>(image[3*i]) / 255.f;
      g = static_cast<float>(image[3*i + 1]) / 255.f;
      b = static_cast<float>(image[3*i + 2]) / 255.f;
      imageData[i] = sycl::float3{r, g, b};
    }

    stbi_image_free(image);

    std::cout << width << std::endl;
    std::cout << height << std::endl;

    sycl::buffer imgBuf{imageData, sycl::range<2>{width, height}};
    sycl::buffer<sycl::float3, 2> outBuf{sycl::range<2>{width, height}};

    sycl::buffer sub1{outBuf, sycl::id<2>{0, 0}, sycl::range<2>{width/ 2, height}};
    sycl::buffer sub2{outBuf, sycl::id<2>{width / 2, 0}, sycl::range<2>{width / 2, height}};

    try {
      sycl::queue q1{sycl::cpu_selector{}};
      sycl::queue q2{sycl::host_selector{}};

      q1.submit([&](sycl::handler &cgh) {
        auto inpAcc = imgBuf.get_access<sycl::access::mode::read>(cgh);
        auto outAcc = sub1.get_access<sycl::access::mode::discard_write>(cgh);
        ImageBlurKernel<sycl::float3, 2> kernel(width, height, inpAcc, outAcc);

        cgh.parallel_for<decltype(kernel)>(sycl::range<2>{width/2, height}, kernel);
      });

      q2.submit([&](sycl::handler &cgh) {
        auto inpAcc = imgBuf.get_access<sycl::access::mode::read>(cgh);
        auto outAcc = sub2.get_access<sycl::access::mode::discard_write>(cgh);
        ImageBlurKernel<sycl::float3, 2> kernel(width, height, inpAcc, outAcc);

        cgh.parallel_for<decltype(kernel)>(sycl::range<2>{width/2, height}, sycl::id<2>{width/2, 0}, kernel);
      });

      q1.wait_and_throw();
      q2.wait_and_throw();
    } catch (sycl::exception e) {
      std::cerr << e.what() << std::endl; 
    }

    delete [] imageData;

    return 0;
}
