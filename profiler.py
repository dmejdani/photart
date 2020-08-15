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

    gaussian_kernel = 1/159 * np.array([
        [2,  4,  5,  4, 2],
        [4,  9, 12,  9, 4],
        [5, 12, 15, 12, 5],
        [4,  9, 12,  9, 4],
        [2,  4,  5,  4, 2]
    ])

    def __init__(self, pic_path="_data/profile.jpg", fir_path="fir.csv"):
        self.pic_path = pic_path
        self.fir_path = fir_path
        self.image = Image.open(pic_path)
        self.array = np.array(self.image)
        self.fir = np.genfromtxt(fir_path, delimiter="\t")

    def get_layer(self, layer):
        self.check_layer_dims(3)
        colors = {"red": 0, "green": 1, "blue": 2}
        idx = colors[layer]
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

    def filter2d(self, layer, filter, double_thresholding=False):
        self.check_layer_dims(2, layer)
        filtered = convolve2d(layer, filter, boundary="symm", mode="same")

        if double_thresholding:
            for row in range(filtered.shape[0]):
                for col in range(filtered.shape[1]):
                    filtered[row, col] = 0 if filtered[row, col] < 110 else 1

        return filtered

    def down_res(self, layer=[], nr_levels=10):
        def remap(v):
            delta = 255 // nr_levels
            return ((v // delta) + 1) * delta

        if not len(layer):
            layer = self.array

        self.check_layer_dims(2, layer)

        for row in range(layer.shape[0]):
            for col in range(layer.shape[1]):
                layer[row, col] = remap(layer[row, col])

        return layer

    @staticmethod
    def map_layer(layer, new_min=0, new_max=255):
        old_min = np.min(layer)
        old_max = np.max(layer)
        layer = layer - old_min  # start at zero
        layer = layer * ((new_max - new_min) / (old_max - old_min))  # scale
        layer = layer + new_min  # shift to correct range
        return layer

    def update_img(self, array=[]):
        if not len(array):
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

    def check_layer_dims(self, n, array=None):
        if array is None:
            array = self.array
        assert len(
            array.shape) == n, f"Operation supported only on {n}d array"


if __name__ == "__main__":
    artist = Artist()

    rl = artist.get_layer("red")
    gl = artist.get_layer("green")
    bl = artist.get_layer("blue")

    quantized = {"r": [], "g": [], "b": []}
    # Comparing the rgb layers
    for key, value in {"r": rl, "g": gl, "b": bl}.items():
        quantized[key] = value
        quantized[key] = artist.filter2d(quantized[key], artist.gaussian_kernel)
        quantized[key] = artist.down_res(quantized[key], nr_levels=6)

    edges = {"r": [], "g": [], "b": []}
    for key, value in quantized.items():
        edges[key] = artist.filter2d(quantized[key], artist.edge_det_kernel)
        quantized[key] = artist.map_layer(quantized[key] + edges[key]).astype(np.uint8)
    
    for key, value in quantized.items():
        quantized[key] = np.expand_dims(value, 2)

    rgb_img = np.append(quantized["r"], quantized["g"], axis=2)
    rgb_img = np.append(rgb_img, quantized["b"], axis=2)
    rgb_im = Image.fromarray(rgb_img, mode="RGB")
    rgb_im.save("_data/quantized.jpg")
    exit()

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
