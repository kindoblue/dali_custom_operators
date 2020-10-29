//
// Created by ice on 26-09-20.
//

#ifndef CUSTOMOP_TIFFDECODER_H
#define CUSTOMOP_TIFFDECODER_H

#include <dali/pipeline/data/backend.h>
#include "dali/pipeline/operator/operator.h"

namespace other_ns {

    class TiffDecoder : public dali::Operator<dali::CPUBackend> {
    public:
        explicit inline TiffDecoder(const dali::OpSpec &spec) :
                Operator<dali::CPUBackend>(spec)
        {}

        inline ~TiffDecoder() override = default;
        DISABLE_COPY_MOVE_ASSIGN(TiffDecoder);

    protected:
        bool SetupImpl(std::vector<dali::OutputDesc> &output_desc, const dali::HostWorkspace &ws) override {
            return false;
        }

        void RunImpl(dali::SampleWorkspace &ws) override;

        int c_;
    };

}  // namespace other_ns


#endif //CUSTOMOP_TIFFDECODER_H
