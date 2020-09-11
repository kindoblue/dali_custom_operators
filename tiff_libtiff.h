//
// Created by ice on 27-09-20.
//

#ifndef CUSTOMOP_TIFF_LIBTIFF_H
#define CUSTOMOP_TIFF_LIBTIFF_H


#include <tiffio.h>
#include <utility>
#include <memory>
#include <dali/core/span.h>
#include <dali/core/tensor_shape.h>

namespace other_ns {

    using Shape = dali::TensorShape<3>;

    class TiffImage_Libtiff {
    public:
        TiffImage_Libtiff(const uint8_t *encoded_buffer, size_t length);
        bool CanDecode() const;
        void Decode();
        Shape GetShape() const;

        std::shared_ptr<uint8_t> GetImage() const;

    protected:
        std::pair<std::shared_ptr<uint8_t>, Shape>
        DecodeImpl(const uint8_t *encoded_buffer, size_t length) const;

        Shape PeekShapeImpl(const uint8_t *encoded_buffer, size_t length) const;

    private:
        dali::span<const uint8_t> buf_;
        size_t buf_pos_;
        std::unique_ptr<TIFF, void (*)(TIFF *)> tif_ = {nullptr, &TIFFClose};

        dali::TensorShape<3> shape_ = {0, 0, 0};
        bool is_tiled_ = false;
        uint16_t bit_depth_ = 8;
        uint16_t orientation_ = ORIENTATION_TOPLEFT;
        uint32_t rows_per_strip_ = 0xFFFFFFFF;
        uint16_t compression_ = COMPRESSION_NONE;
        std::shared_ptr<uint8_t> decoded_image_ = nullptr;
        const uint8_t *encoded_image_;
        const size_t length_;
        bool decoded_ = false;
    };

}  // namespace other_ns


#endif //CUSTOMOP_TIFF_LIBTIFF_H
