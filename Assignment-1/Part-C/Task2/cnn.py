import numpy as np
import matplotlib.pyplot as plt
from EDF import *

IMAGE_PATH = 'nature.jpg'

blur_3x3 = np.array([[[[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]]], dtype=np.float32) / 16.0

blur_11 = np.ones((1, 1, 11, 11), dtype=np.float32) / 121.0

sobel_vertical = np.array([[[[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]]]], dtype=np.float32)

sobel_horizontal = np.array([[[[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]]]], dtype=np.float32)

def load_image(path):
    img = plt.imread(path).astype(np.float32)
    img = img / 255.0
    if img.ndim == 2:
        H, W = img.shape
        C = 1
        img = img.reshape(1, C, H, W)
    else:
        H, W, C = img.shape
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]
    return img, H, W, C

def prepare_kernel(kernel_1ch, C, out_channels):
    Kh, Kw = kernel_1ch.shape[2], kernel_1ch.shape[3]
    K = np.zeros((out_channels, C, Kh, Kw), dtype=np.float32)
    for c in range(out_channels):
        K[c, c % C] = kernel_1ch[0, 0]
    return K

def apply_conv(img, kernel_1ch):
    N, C, H, W = img.shape
    kernel = prepare_kernel(kernel_1ch, C, C)
    x = Input()
    x.value = img
    conv = Conv(x, input_channels=C, output_channels=C,
                kernel_height=kernel.shape[2],
                kernel_width=kernel.shape[3],
                kernel=kernel, bias=np.zeros((C,), dtype=np.float32),
                stride=1, padding=1)
    graph = [x, conv]
    for n in graph:
        n.forward()
    out = conv.value[0].transpose(1, 2, 0)
    return out

def apply_pool(img, ph, pw):
    x = Input()
    x.value = img
    pool = MaxPooling(x, pool_height=ph, pool_width=pw)
    graph = [x, pool]
    for n in graph:
        n.forward()
    out = pool.value[0].transpose(1, 2, 0)
    return out

def show(img1, img2, title1, title2):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img1)
    axs[0].set_title(title1)
    axs[0].axis('off')
    axs[1].imshow(img2)
    axs[1].set_title(title2)
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

img, H, W, C = load_image(IMAGE_PATH)
orig = img[0].transpose(1, 2, 0)

out_h1 = apply_conv(img, sobel_horizontal)
out_h2 = apply_conv(img, -sobel_horizontal)
show(orig, out_h1, "Original", "Horizontal Edge (down→up)")
show(orig, out_h2, "Original", "Horizontal Edge (up→down)")

out_v1 = apply_conv(img, sobel_vertical)
out_v2 = apply_conv(img, -sobel_vertical)
show(orig, out_v1, "Original", "Vertical Edge (left→right)")
show(orig, out_v2, "Original", "Vertical Edge (right→left)")

blur3 = apply_conv(img, blur_3x3)
blur11 = apply_conv(img, blur_11)
show(orig, blur3, "Original", "Blur 3x3")
show(orig, blur11, "Original", "Blur 11x11")

pool2 = apply_pool(img, 2, 2)
pool8 = apply_pool(img, 8, 8)
show(orig, pool2, "Original", "MaxPool 2x2")
show(orig, pool8, "Original", "MaxPool 8x8")
