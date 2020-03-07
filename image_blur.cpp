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

  void operator()(sycl::id<2> id) {
    T sum{};
    for (int x = -KerSize / 2; x < KerSize / 2; x++) {
      for (int y = -KerSize / 2; y < KerSize / 2; y++) {
        auto realX = sycl::clamp(id[0] + x, 0UL, mImgWidth);
        auto realY = sycl::clamp(id[1] + y, 0UL, mImgHeight);
        sum += mInpImg[sycl::id<2>{realX, realY}];
      }
    }     
    mOutImg[id] = sum / (KerSize * KerSize);
  }
};

int main() {
    int w, h, c;

    std::string filename = "image.png";
    unsigned char* image = stbi_load(filename.c_str(), &w, &h, &c, STBI_rgb);

    size_t width = w, height = h, comp = c;

    if (image == nullptr) {
        std::terminate();
    }

    sycl::buffer imgBuf{reinterpret_cast<sycl::float3*>(image), sycl::range<2>{width, height}};
    sycl::buffer<sycl::float3, 2> outBuf{sycl::range<2>{width, height}};

    sycl::buffer sub1{outBuf, sycl::id<2>{0, 0}, sycl::range<2>{width/ 2, height}};
    sycl::buffer sub2{outBuf, sycl::id<2>{width / 2, 0}, sycl::range<2>{width / 2, height}};

    try {
      sycl::queue q1{sycl::cpu_selector{}};
      sycl::queue q2{sycl::host_selector{}};

      /*q1.submit([&](sycl::handler &cgh) {
        auto inpAcc = imgBuf.get_access<sycl::access::mode::read>(cgh);
        auto outAcc = sub1.get_access<sycl::access::mode::discard_write>(cgh);
        ImageBlurKernel<sycl::float3, 2> kernel(width, height, inpAcc, outAcc);

        cgh.parallel_for<decltype(kernel)>(sycl::range<2>{width/2, height}, kernel);
      });*/

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


    return 0;
}