#ifndef FILE_READER_H_
#define FILE_READER_H_

#include <vector>

#include <dali/pipeline/operator/operator.h>
#include <dali/operators/reader/reader_op.h>

#include "file_file_loader.h"

namespace other_ns {

class SegFileReader : public dali::DataReader<dali::CPUBackend, ImageMaskWrapper> {
public:
    explicit SegFileReader(const dali::OpSpec& spec)
            : DataReader<dali::CPUBackend, ImageMaskWrapper>(spec) {
        bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
        loader_ = dali::InitLoader<FileMaskLoader>(spec, std::vector<std::pair<std::string, std::string>>(),
                                                   shuffle_after_epoch);
    }

    void RunImpl(dali::SampleWorkspace &ws) override {
        const int idx = ws.data_idx();

        const auto& image_mask = GetSample(idx);

        // copy from raw_data -> outputs directly
        auto &image_output = ws.Output<dali::CPUBackend>(0);
        auto &mask_output = ws.Output<dali::CPUBackend>(1);

        dali::Index image_size = image_mask.image.size();
        dali::Index mask_size = image_mask.mask.size();

        image_output.Resize({image_size});
        mask_output.Resize({mask_size});

        image_output.mutable_data<uint8_t>();
        mask_output.mutable_data<uint8_t>();

        std::memcpy(image_output.raw_mutable_data(),
                    image_mask.image.raw_data(),
                    image_size);

        std::memcpy(mask_output.raw_mutable_data(),
                    image_mask.mask.raw_data(),
                    mask_size);

        image_output.SetSourceInfo(image_mask.image.GetSourceInfo());
        mask_output.SetSourceInfo(image_mask.mask.GetSourceInfo());

    }

protected:
    USE_READER_OPERATOR_MEMBERS(dali::CPUBackend, ImageMaskWrapper);
};

}  // namespace other_ns

#endif  // FILE_READER_H_
