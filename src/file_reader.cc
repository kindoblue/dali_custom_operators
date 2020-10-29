#include "file_reader.h"

namespace other_ns {

    DALI_REGISTER_OPERATOR(SegFileReader, SegFileReader, dali::CPU);

    DALI_SCHEMA(SegFileReader)
                    .DocStr("Read (Image, Mask) pairs from a directory")
                    .NumInput(0)
                    .NumOutput(2)  // (Images, Masks)
                    .AddArg("file_root",
                            R"code(Path to a directory containing data files.
``FileReader`` supports flat directory structure. ``file_root`` directory should contain
directories with images in them. To obtain labels ``FileReader`` sorts directories in
``file_root`` in alphabetical order and takes an index in this order as a class label.)code",
                            dali::DALI_STRING)
                    .AddArg("file_list",
                            R"code(Path to a text file containing rows of ``filename label`` pairs, where the filenames are
relative to ``file_root``.)code",
                            dali::DALI_STRING)
                    .AddOptionalArg("shuffle_after_epoch",
                                    R"code(If true, reader shuffles whole dataset after each epoch. It is exclusive with
``stick_to_shard`` and ``random_shuffle``.)code",
                                    false)
                    .AddParent("LoaderBase");

}  // namespace other_ns
