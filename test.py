import nvidia.dali.plugin_manager as plugin_manager
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class TestPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(TestPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.SegFileReader(file_root='data', file_list='data/file_list.txt')
        self.decodeMulti = ops.TiffDecoder(device="cpu")

    def define_graph(self):
        image, mask = self.input()
        image = self.decodeMulti(image)
        mask = self.decodeMulti(mask)
        return image, mask


def main():

    plugin_manager.load_library('./cmake-build-debug/libCustomOp.so')

    bs = 8
    pipe = TestPipeline(batch_size=bs, num_threads=4, device_id=0)
    pipe.build()
    pipe_out = pipe.run()
    gimage, gmask = pipe_out
    print(gmask.as_tensor())
    print(gmask.layout())


if __name__ == '__main__':
    main()
