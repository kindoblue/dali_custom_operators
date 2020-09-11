//
// Created by ice on 26-09-20.
//

#include "tiff_decoder.h"
#include "tiff_libtiff.h"

#include <dali/pipeline/data/types.h>
#include <tiff.h>

namespace other_ns {

    void TiffDecoder::RunImpl(dali::SampleWorkspace &ws) {
        const auto &input = ws.Input<dali::CPUBackend>(0);
        auto &output = ws.Output<dali::CPUBackend>(0);
        auto file_name = input.GetSourceInfo();

        // Verify input
        DALI_ENFORCE(input.ndim() == 1,
                     "Input must be 1D encoded image.");
        DALI_ENFORCE(dali::IsType<uint8>(input.type()),
                     "Input must be stored as uint8 data.");

        // Decode the image
        std::unique_ptr<TiffImage_Libtiff> img;
        try {
            img = std::make_unique<TiffImage_Libtiff>(input.data<uint8>(), input.size());
            img->Decode();
        } catch (std::exception &e) {
            DALI_FAIL(e.what() + ". File: " + file_name);
        }

        // Return the image
        const auto decoded = img->GetImage();
        const auto shape = img->GetShape();
        output.Resize(shape);
        output.SetLayout("CHW");
        auto *out_data = output.mutable_data<unsigned char>();
        std::memcpy(out_data, decoded.get(), volume(shape));  // TODO pass out_data to Decode to directly write there and avoid the memcopy
    }

    DALI_REGISTER_OPERATOR(TiffDecoder, TiffDecoder, dali::CPU);

    DALI_SCHEMA(TiffDecoder)
       .DocStr(R"code(Decodes TIFF multichannel images)code")
       .NumInput(1)
       .NumOutput(1);

}  // namespace other_ns
