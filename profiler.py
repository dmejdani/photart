from PIL import Image
import numpy as np

fir_fn = "fir.csv"


def normalize_arr(img_arr):
    min_v = np.min(img_arr)
    img_arr -= min_v
    max_v = np.max(img_arr)
    return img_arr * 255 / max_v


# opening file with PIL / Pillow
img = Image.open("_data/profile.jpg")
print(img.format, img.size, img.mode)

# convert to numpy array to process easier
arr = np.array(img)
print(arr.shape)

# RGB to grayscale image
gray_arr = np.sum(arr, 2, dtype=np.uint8) / 3
print(gray_arr.shape)

fir = np.genfromtxt(fir_fn, delimiter="\t")
# print(fir)

for row in range(gray_arr.shape[0]):
    gray_arr[row] = np.convolve(gray_arr[row], fir, "same").astype(np.uint8)
for col in range(gray_arr.shape[1]):
    gray_arr[:, col] = np.convolve(
        gray_arr[:, col], fir, "same").astype(np.uint8)

print(arr.dtype)
print(gray_arr.dtype)

out = arr
out[:, :, 0] += normalize_arr(gray_arr).astype(np.uint8)
out[:, :, 0] = out[:, :, 0] / 2

img_g = Image.fromarray(out)
img_g.show()
