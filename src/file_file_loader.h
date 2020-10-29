#ifndef DALI_OPERATORS_READER_LOADER_FILE_LABEL_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_FILE_LABEL_LOADER_H_

#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>

#include <dali/core/common.h>
#include <dali/util/file.h>
#include <dali/operators/reader/loader/loader.h>

namespace other_ns {

namespace filesystem {

std::vector<std::pair<std::string, std::string>> traverse_directories(const std::string& path);

}  // namespace filesystem

struct ImageMaskWrapper {
  dali::Tensor<dali::CPUBackend> image;
  dali::Tensor<dali::CPUBackend> mask;
};

class FileMaskLoader : public dali::Loader<dali::CPUBackend, ImageMaskWrapper> {
 public:
  explicit inline FileMaskLoader(
    const dali::OpSpec& spec,
    std::vector<std::pair<std::string, std::string>> image_mask_pairs = std::vector<std::pair<std::string, std::string>>(),
    bool shuffle_after_epoch = false)
    : Loader<dali::CPUBackend, ImageMaskWrapper>(spec),
      file_root_(spec.GetArgument<std::string>("file_root")),
      file_list_(spec.GetArgument<std::string>("file_list")),
      image_mask_pairs_(std::move(image_mask_pairs)),
      shuffle_after_epoch_(shuffle_after_epoch),
      current_index_(0),
      current_epoch_(0) {
      /*
      * Those options are mutually exclusive as `shuffle_after_epoch` will make every shard looks differently
      * after each epoch so coexistence with `stick_to_shard` doesn't make any sense
      * Still when `shuffle_after_epoch` we will set `stick_to_shard` internally in the FileMaskLoader so all
      * DALI instances will do shuffling after each epoch
      */
      if (shuffle_after_epoch_ || stick_to_shard_)
        DALI_ENFORCE(
          !shuffle_after_epoch_ || !stick_to_shard_,
          "shuffle_after_epoch and stick_to_shard cannot be both true");
      if (shuffle_after_epoch_ || shuffle_)
        DALI_ENFORCE(
          !shuffle_after_epoch_ || !shuffle_,
          "shuffle_after_epoch and random_shuffle cannot be both true");
      /*
       * Imply `stick_to_shard` from  `shuffle_after_epoch`
       */
      if (shuffle_after_epoch_) {
        stick_to_shard_ = true;
      }
    if (!dont_use_mmap_) {
      mmap_reserver = dali::FileStream::FileStreamMappinReserver(static_cast<unsigned int>(initial_buffer_fill_));
    }
    copy_read_data_ = dont_use_mmap_ || !mmap_reserver.CanShareMappedData();
  }

  void PrepareEmpty(ImageMaskWrapper &image_mask) override;
  void ReadSample(ImageMaskWrapper &image_mask) override;

 protected:
  dali::Index SizeImpl() override;

  void PrepareMetadataImpl() override {
    if (image_mask_pairs_.empty()) {
        // load (path, mask) pairs from list
        std::ifstream s(file_list_);
        DALI_ENFORCE(s.is_open(), "Cannot open: " + file_list_);

        std::string image_file;
        std::string mask_file;
        while (s >> image_file >> mask_file) {
            auto p = std::make_pair(image_file, mask_file);
            image_mask_pairs_.push_back(p);
        }
        DALI_ENFORCE(s.eof(), "Wrong format of file_list: " + file_list_);
    }
    DALI_ENFORCE(Size() > 0, "No files found.");

    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(kDaliDataloaderSeed);
      std::shuffle(image_mask_pairs_.begin(), image_mask_pairs_.end(), g);
    }
    Reset(true);
  }

  void Reset(bool wrap_to_shard) override {
    if (wrap_to_shard) {
      current_index_ = dali::start_index(shard_id_, num_shards_, Size());
    } else {
      current_index_ = 0;
    }

    current_epoch_++;

    if (shuffle_after_epoch_) {
      std::mt19937 g(kDaliDataloaderSeed + current_epoch_);
      std::shuffle(image_mask_pairs_.begin(), image_mask_pairs_.end(), g);
    }
  }

  using Loader<dali::CPUBackend, ImageMaskWrapper>::shard_id_;
  using Loader<dali::CPUBackend, ImageMaskWrapper>::num_shards_;

  std::string file_root_, file_list_;
  std::vector<std::pair<std::string, std::string>> image_mask_pairs_;
  bool shuffle_after_epoch_;
  dali::Index current_index_;
  int current_epoch_;
  dali::FileStream::FileStreamMappinReserver mmap_reserver;
};

}  // namespace

#endif  // DALI_OPERATORS_READER_LOADER_FILE_LABEL_LOADER_H_
