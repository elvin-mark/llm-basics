import numpy as np


def bicubic_kernel(x, a=-0.75):
    abs_x = np.abs(x)
    abs_x2 = abs_x**2
    abs_x3 = abs_x**3

    result = np.where(
        abs_x <= 1,
        (a + 2) * abs_x3 - (a + 3) * abs_x2 + 1,
        np.where(
            abs_x <= 2,
            a * abs_x3 - 5 * a * abs_x2 + 8 * a * abs_x - 4 * a,
            0,
        ),
    )
    return result


def bicubic_interpolate(image, x, y):
    _, h, w = image.shape

    # Get integer and fractional parts of coordinates
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    dx = x - x0
    dy = y - y0

    # Initialize result
    result = 0.0

    # Iterate over the 4x4 neighborhood
    for m in range(-1, 3):
        for n in range(-1, 3):
            xi = np.clip(x0 + m, 0, w - 1)
            yi = np.clip(y0 + n, 0, h - 1)

            weight = bicubic_kernel(m - dx) * bicubic_kernel(n - dy)
            result += weight * image[:, yi, xi]

    return result


def resize_bicubic(image, new_width, new_height):
    n, h, w = image.shape
    resized_image = np.zeros((n, new_height, new_width))

    # Scale factors
    scale_x = w / new_width
    scale_y = h / new_height

    for j in range(new_height):
        for i in range(new_width):
            x = (i + 0.5) * scale_x - 0.5  # Map to source space
            y = (j + 0.5) * scale_y - 0.5  # Map to source space

            resized_image[:, j, i] = bicubic_interpolate(image, x, y)

    return resized_image


def resize_image(image, new_height, new_width):
    _, old_height, old_width = image.shape

    row_scale = old_height / new_height
    col_scale = old_width / new_width

    row_indices = (np.arange(new_height) * row_scale).astype(int)
    col_indices = (np.arange(new_width) * col_scale).astype(int)

    resized_image = image[:, row_indices[:, None], col_indices]

    return resized_image
