//
// Created by ice on 27-09-20.
//

#include "tiff_libtiff.h"

#include <tiffio.h>
#include <cstring>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include "dali/core/span.h"

#define LIBTIFF_CALL_SUCCESS 1
#define LIBTIFF_CALL(call)                                \
  do {                                                    \
    int retcode = (call);                                 \
    DALI_ENFORCE(LIBTIFF_CALL_SUCCESS == retcode,         \
      "libtiff call failed with code "                    \
      + std::to_string(retcode) + ": " #call);            \
  } while (0)

namespace other_ns {
    namespace detail {

        class BufDecoderHelper {
        private:
            dali::span<const uint8_t> &buf_;
            size_t &buf_pos_;

        public:
            BufDecoderHelper(dali::span<const uint8_t> &buf, size_t &buf_pos)
                    : buf_(buf), buf_pos_(buf_pos) {}

            static tmsize_t read(thandle_t handle, void *buffer, tmsize_t n) {
                auto *helper = reinterpret_cast<BufDecoderHelper *>(handle);
                auto &buf = helper->buf_;
                const tmsize_t size = buf.size();
                tmsize_t pos = helper->buf_pos_;
                if (n > (size - pos)) {
                    n = size - pos;
                }
                memcpy(buffer, buf.data() + pos, n);
                helper->buf_pos_ += n;
                return n;
            }

            static tmsize_t write(thandle_t /*handle*/, void * /*buffer*/, tmsize_t /*n*/) {
                // Not used for decoding.
                return 0;
            }

            static toff_t seek(thandle_t handle, toff_t offset, int whence) {
                auto *helper = reinterpret_cast<BufDecoderHelper *>(handle);
                auto &buf = helper->buf_;
                const toff_t size = buf.size();
                toff_t new_pos = helper->buf_pos_;
                switch (whence) {
                    case SEEK_SET:
                        new_pos = offset;
                        break;
                    case SEEK_CUR:
                        new_pos += offset;
                        break;
                    case SEEK_END:
                        new_pos = size + offset;
                        break;
                }
                new_pos = std::min(new_pos, size);
                helper->buf_pos_ = static_cast<size_t>(new_pos);
                return new_pos;
            }

            static int map(thandle_t handle, void **base, toff_t *size) {
                auto *helper = reinterpret_cast<BufDecoderHelper *>(handle);
                auto &buf = helper->buf_;
                *base = const_cast<uint8_t*>(buf.data());
                *size = buf.size();
                return 0;
            }

            static toff_t size(thandle_t handle) {
                auto *helper = reinterpret_cast<BufDecoderHelper *>(handle);
                return helper->buf_.size();
            }

            static int close(thandle_t handle) {
                auto *helper = reinterpret_cast<BufDecoderHelper *>(handle);
                delete helper;
                return 0;
            }
        };

    }  // namespace detail

    TiffImage_Libtiff::TiffImage_Libtiff(const uint8_t *encoded_buffer,
                                         size_t length)
            : length_(length),
              encoded_image_(encoded_buffer),
              buf_({encoded_buffer, static_cast<ptrdiff_t>(length)}),
              buf_pos_(0) {
        tif_.reset(
                TIFFClientOpen("", "r",
                               reinterpret_cast<thandle_t>(
                                       new detail::BufDecoderHelper(buf_, buf_pos_)),
                               &detail::BufDecoderHelper::read,
                               &detail::BufDecoderHelper::write,
                               &detail::BufDecoderHelper::seek,
                               &detail::BufDecoderHelper::close,
                               &detail::BufDecoderHelper::size,
                               &detail::BufDecoderHelper::map,
                        /*unmap=*/0));
        DALI_ENFORCE(tif_, "Cannot open TIFF file.");

        LIBTIFF_CALL(
                TIFFGetField(tif_.get(), TIFFTAG_IMAGELENGTH, &shape_[1]));
        LIBTIFF_CALL(
                TIFFGetField(tif_.get(), TIFFTAG_IMAGEWIDTH, &shape_[2]));

        unsigned int dircount = 0;
        do { dircount++; } while (TIFFReadDirectory(tif_.get()));
        shape_[0] = dircount;

        is_tiled_ = static_cast<bool>(
                TIFFIsTiled(tif_.get()));
        LIBTIFF_CALL(
                TIFFGetFieldDefaulted(tif_.get(), TIFFTAG_BITSPERSAMPLE, &bit_depth_));
        DALI_ENFORCE(bit_depth_ <= 64,
                     "Unexpected bit depth: " + std::to_string(bit_depth_));
        LIBTIFF_CALL(
                TIFFGetFieldDefaulted(tif_.get(), TIFFTAG_ORIENTATION, &orientation_));
        LIBTIFF_CALL(
                TIFFGetFieldDefaulted(tif_.get(), TIFFTAG_ROWSPERSTRIP, &rows_per_strip_));
        LIBTIFF_CALL(
                TIFFGetFieldDefaulted(tif_.get(), TIFFTAG_COMPRESSION, &compression_));
    }

    Shape TiffImage_Libtiff::PeekShapeImpl(const uint8_t *encoded_buffer, size_t length) const {
        DALI_ENFORCE(encoded_buffer != nullptr);
        assert(encoded_buffer == buf_.data());
        return shape_;
    }

    void TiffImage_Libtiff::Decode() {
        DALI_ENFORCE(!decoded_, "Called decode for already decoded image");
        auto decoded = DecodeImpl(encoded_image_, length_);
        decoded_image_ = decoded.first;
        shape_ = decoded.second;
        decoded_ = true;
    }

    std::pair<std::shared_ptr<uint8_t>, Shape>
    TiffImage_Libtiff::DecodeImpl(const uint8 *encoded_buffer, size_t length) const {

        if (!CanDecode()) {
            DALI_FAIL("TIFF file should be not tiled, 8bit, topleft");
        }

        // allocate memory for the output tensor
        const size_t decoded_size = volume(shape_);
        std::shared_ptr<uint8_t> decoded_img_ptr{
                new uint8_t[decoded_size],
                [](uint8_t* ptr){ delete [] ptr; }
        };

        uint8_t * tensor_out = decoded_img_ptr.get();

        unsigned int row_stride = shape_[2];  // stride for row

        // decode the tiff buffer
        for (int dirnum = 0; dirnum < shape_[0]; dirnum++) {
            LIBTIFF_CALL(TIFFSetDirectory(tif_.get(), dirnum));
            for (int row = 0; row < shape_[1]; row++)
            {
                LIBTIFF_CALL(TIFFReadScanline(tif_.get(), tensor_out, row));
                tensor_out += row_stride;
            }
        }

        return {decoded_img_ptr, shape_};
    }

    Shape TiffImage_Libtiff::GetShape() const {
        DALI_ENFORCE(decoded_, "Image not decoded. Run Decode()");
        return shape_;
    }

    std::shared_ptr<uint8_t> TiffImage_Libtiff::GetImage() const {
        DALI_ENFORCE(decoded_, "Image not decoded. Run Decode()");
        return decoded_image_;
    }

    bool TiffImage_Libtiff::CanDecode() const {
        return !is_tiled_
               && bit_depth_ == 8
               && orientation_ == ORIENTATION_TOPLEFT;
    }

}  // namespace other_ns