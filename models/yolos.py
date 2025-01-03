import numpy as np
from utils.nn import convolution_2d

# Example usage
if __name__ == "__main__":
    # Create a dummy input image and kernel
    img = np.random.rand(3, 8, 8)  # 3 channels, 8x8 image
    kern = np.random.rand(3, 2, 3, 3)  # 3 input channels, 2 output channels, 3x3 kernel

    # Perform the convolution
    output = convolution_2d(img, kern, stride=2, padding=1)
    print("Output shape:", output.shape)
