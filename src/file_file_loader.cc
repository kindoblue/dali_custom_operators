#include <dirent.h>
#include <errno.h>
#include <memory>

#include <dali/core/common.h>
#include <dali/util/file.h>
#include <dali/operators/reader/loader/utils.h>

#include "file_file_loader.h"

namespace other_ns {

void FileMaskLoader::PrepareEmpty(ImageMaskWrapper &image_mask) {
  PrepareEmptyTensor(image_mask.image);
  PrepareEmptyTensor(image_mask.mask);
}

void FileMaskLoader::ReadSample(ImageMaskWrapper &image_mask) {
  auto image_pair = image_mask_pairs_[current_index_++];

  // handle wrap-around
  MoveToNextShard(current_index_);

  dali::DALIMeta imageMeta;
  imageMeta.SetSourceInfo(image_pair.first);
  imageMeta.SetSkipSample(false);

  // if image is cached, skip loading
  if (ShouldSkipImage(image_pair.first)) {
    imageMeta.SetSkipSample(true);
    image_mask.image.Reset();
    image_mask.image.SetMeta(imageMeta);
    image_mask.image.set_type(dali::TypeInfo::Create<uint8_t>());
    image_mask.image.Resize({0});
    return;
  }

  dali::DALIMeta maskMeta;
  maskMeta.SetSourceInfo(image_pair.second);
  maskMeta.SetSkipSample(false);

  // if image is cached, skip loading
  if (ShouldSkipImage(image_pair.second)) {
      maskMeta.SetSkipSample(true);
      image_mask.mask.Reset();
      image_mask.mask.SetMeta(maskMeta);
      image_mask.mask.set_type(dali::TypeInfo::Create<uint8_t>());
      image_mask.mask.Resize({0});
      return;
  }

  auto current_image = dali::FileStream::Open(file_root_ + "/" + image_pair.first,
                                        read_ahead_, !copy_read_data_);
  auto current_mask = dali::FileStream::Open(file_root_ + "/" + image_pair.second,
                                        read_ahead_, !copy_read_data_);
  dali::Index image_size = current_image->Size();
  dali::Index mask_size = current_mask->Size();

  if (copy_read_data_) {

    if (image_mask.image.shares_data()) {
      image_mask.image.Reset();
    }
    image_mask.image.Resize({image_size});
    // copy the image
    dali::Index ret = current_image->Read(image_mask.image.mutable_data<uint8_t>(), image_size);
    DALI_ENFORCE(ret == image_size, dali::make_string("Failed to read image: ", image_pair.first));

    if (image_mask.mask.shares_data()) {
        image_mask.mask.Reset();
    }
    image_mask.mask.Resize({mask_size});
    // copy the image
    ret = current_mask->Read(image_mask.mask.mutable_data<uint8_t>(), mask_size);
    DALI_ENFORCE(ret == mask_size, dali::make_string("Failed to read mask: ", image_pair.second));

  } else {

    auto p = current_image->Get(image_size);
    DALI_ENFORCE(p != nullptr, dali::make_string("Failed to read file: ", image_pair.first));
    // Wrap the raw data in the Tensor object.
    image_mask.image.ShareData(p, image_size, {image_size});
    image_mask.image.set_type(dali::TypeInfo::Create<uint8_t>());

    p = current_mask->Get(mask_size);
    DALI_ENFORCE(p != nullptr, dali::make_string("Failed to read file: ", image_pair.second));
    // Wrap the raw data in the Tensor object.
    image_mask.mask.ShareData(p, mask_size, {mask_size});
    image_mask.mask.set_type(dali::TypeInfo::Create<uint8_t>());

  }

  // close the file handles
  current_image->Close();
  current_mask->Close();

  image_mask.image.SetMeta(imageMeta);
  image_mask.mask.SetMeta(maskMeta);
}

dali::Index FileMaskLoader::SizeImpl() {
  return static_cast<dali::Index>(image_mask_pairs_.size());
}

}  // namespace dali
