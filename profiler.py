from PIL import Image
import numpy as np
from scipy.signal import convolve2d

fir_fn = "fir.csv"


class Artist:

    edge_det_kernel = np.array([
        [0, -1,  0],
        [-1, 4, -1],
        [0, -1,  0]
    ])

    def __init__(self, pic_path="_data/profile.jpg", fir_path="fir.csv"):
        self.pic_path = pic_path
        self.fir_path = fir_path
        self.image = Image.open(pic_path)
        self.array = np.array(self.image)
        self.fir = np.genfromtxt(fir_path, delimiter="\t")

    def get_layer(self, layer):
        self.check_layer_dims(3)
        if layer == "red":
            idx = 0
        elif layer == "green":
            idx = 1
        elif layer == "blue":
            idx = 2
        else:
            raise RuntimeError("Layer argument not valid!")
        return self.array[:, :, idx]

    def togray(self):
        self.check_layer_dims(3)
        self.array = np.sum(self.array, axis=2) / 3
        self.update_img(self.array)
        return self.array

    def filter(self):
        self.check_layer_dims(2)
        for row in range(self.array.shape[0]):
            self.array[row, :] = np.convolve(
                self.array[row, :], self.fir, "same")
        for col in range(self.array.shape[1]):
            self.array[:, col] = np.convolve(
                self.array[:, col], self.fir, "same")
        self.array = self.map_layer(self.array)
        self.update_img(self.array)

    def filter2d(self, layer):
        # self.check_layer_dims(2)
        return convolve2d(
            layer, self.edge_det_kernel, boundary="symm", mode="same")

    @staticmethod
    def map_layer(layer, new_min=0, new_max=255):
        old_min = np.min(layer)
        old_max = np.max(layer)
        layer = layer - old_min  # start at zero
        layer = layer * ((new_max - new_min) / (old_max - old_min))  # scale
        layer = layer + new_min  # shift to correct range
        return layer

    def update_img(self, array=None):
        if array is None:
            array = self.array
        else:
            self.array = array
        if len(array.shape) == 3:
            mode = "RGB"
        else:
            mode = "L"
        array = array.astype(np.uint8)
        self.image = Image.fromarray(array, mode)

    def show(self):
        self.image.show()

    def check_layer_dims(self, n):
        assert len(
            self.array.shape) == n, f"Operation supported only on {n}d array"


if __name__ == "__main__":
    artist = Artist()

    rl = artist.get_layer("red")
    gl = artist.get_layer("green")
    bl = artist.get_layer("blue")

    # edge filtering on each color
    fr, fg, fb = map(artist.filter2d, [rl, gl, bl])

    rl = rl + fr
    gl = gl + fg
    bl = bl + fb

    rl = np.expand_dims(rl, 2)
    gl = np.expand_dims(gl, 2)
    bl = np.expand_dims(bl, 2)

    artsy = np.append(rl, gl, axis=2)
    artsy = np.append(artsy, bl, axis=2)

    artist.array = artsy
    artist.update_img(artist.array)
    artist.show()
