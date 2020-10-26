import nvidia.dali.plugin_manager as plugin_manager
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class TestPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(TestPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.SegFileReader(file_root='data', file_list='data/file_list.txt', random_shuffle=True)
        self.decodeMulti = ops.TiffDecoder()
        self.rotate = ops.Rotate(device='gpu', interp_type=types.INTERP_NN, keep_size=True)
        self.rotate_range = ops.Uniform(range=(-27, 27))
        self.transpose = ops.Transpose(perm=[1, 2, 0])
        self.transposeBack = ops.Transpose(device='gpu', perm=[2, 0, 1])

    def define_graph(self):
        angle_range = self.rotate_range()
        image, mask = self.input()
        image = self.decodeMulti(image)
        mask = self.decodeMulti(mask)
        image = self.transpose(image)
        image = self.rotate(image.gpu(), angle=angle_range)
        image = self.transposeBack(image)
        mask = self.transpose(mask)
        mask = self.rotate(mask.gpu(), angle=angle_range)
        mask = self.transposeBack(mask)

        return image, mask


def main():
    plugin_manager.load_library('./cmake-build-debug/libCustomOp.so')
    pipe = TestPipeline(batch_size=8, num_threads=4, device_id=0)
    pipe.build()
    pipe_out = pipe.run()
    gimage, gmask = pipe_out
    print(gmask.as_tensor())
    print(gmask.layout())


if __name__ == '__main__':
    main()
