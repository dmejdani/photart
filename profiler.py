from PIL import Image
import numpy as np

fir_fn = "fir.csv"


class Artist:

    def __init__(self, pic_path="_data/profile.jpg", fir_path="fir.csv"):
        self.pic_path = pic_path
        self.fir_path = fir_path
        self.image = Image.open(pic_path)
        self.array = np.array(self.image)
        self.fir = np.genfromtxt(fir_path, delimiter="\t")

    def togray(self):
        assert len(self.array.shape) == 3, "Array not of shape (*, *, 3)"
        self.array = np.sum(self.array, axis=2) / 3
        self.update_img(self.array)
        return self.array
    
    def filter(self):
        assert len(self.array.shape) == 2, "Filtering supported only on 2d array"
        for row in range(self.array.shape[0]):
            self.array[row, :] = np.convolve(self.array[row, :], self.fir, "same")
        for col in range(self.array.shape[1]):
            self.array[:, col] = np.convolve(self.array[:, col], self.fir, "same")
        self.array = self.map_layer(self.array)
        print(np.max(self.array))
        print(np.min(self.array))
        self.update_img(self.array)
    
    @staticmethod
    def map_layer(layer, new_min=0, new_max=255):
        old_min = np.min(layer)
        print(old_min)
        old_max = np.max(layer)
        print(old_max)
        layer = layer - old_min  # start at zero
        print(np.min(layer))
        layer = layer * ((new_max - new_min) / (old_max - old_min))  # scale
        print(np.min(layer))
        print(np.max(layer))
        layer = layer + new_min  # shift to correct range
        return layer

    def update_img(self, array):
        self.image = Image.fromarray(array)

    def show(self):
        self.image.show()


if __name__ == "__main__":
    artist = Artist()
    # artist.show()
    artist.togray()
    # layer = artist.map_layer(artist.array, 0, 255)
    # artist.update_img(layer)
    artist.filter()
    print(artist.array)
    artist.show()