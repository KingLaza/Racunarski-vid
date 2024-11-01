import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def filter_image(image, treshold):
  # Find the center of the frequency domain
  rows, cols = image.shape
  center_row, center_col = rows // 2, cols // 2

  # Zero out the brightness of specific dots (you may need to adjust these positions slightly)
  offset = 50  # Distance from the center for the dots

  # image[center_row - offset, center_col - offset] = 0
  # image[center_row + offset, center_col + offset] = 0
  # image[center_row - offset, center_col + offset] = 0
  # image[center_row + offset, center_col -  offset] = 0
  curRow = curCol = 0
  #threshold = 3
  lista_zero = []

  # Create a mask to mark the bright spots to be zeroed out
  zero_mask = np.zeros_like(image, dtype=bool)
  image_abs = np.log(np.abs(image) + 1)

  for i in range(rows):
    for j in range(cols):
      neighbor_sum = 0
      count = 4

      # Check and add neighboring pixels while handling boundaries
      if i > 0:
        neighbor_sum += image_abs[i - 1, j]
      else:
        count -= 1

      if i < rows - 1:
        neighbor_sum += image_abs[i + 1, j]
      else:
        count -= 1

      if j > 0:
        neighbor_sum += image_abs[i, j - 1]
      else:
        count -= 1

      if j < cols - 1:
        neighbor_sum += image_abs[i, j + 1]
      else:
        count -= 1

      # Calculate the average of the neighbors
      avg_neighbor_value = neighbor_sum / count

      # Check if the current pixel is significantly brighter than its neighbors
      if (image_abs[i, j] - avg_neighbor_value > treshold):
        lista_zero.append([i, j])
        zero_mask[i, j] = True

  # Apply the mask to zero out bright spots
  print("lista zero")
  for i in lista_zero:
    print(i)
  image[zero_mask] = 0
  return image

if __name__ == '__main__':

  lena = cv.imread('./Pictures/slika_4.png', cv.IMREAD_GRAYSCALE)
  filtrirana = np.fft.fft2(lena)
  filtrirana_shifted = np.fft.fftshift(filtrirana)

  plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.imshow(np.log(np.abs(filtrirana_shifted) + 1), cmap='gray')
  plt.title("Original frequency")
  #plt.show()


  filter_image(filtrirana_shifted, 3)

  plt.subplot(1, 2, 2)
  plt.imshow(np.log(np.abs(filtrirana_shifted) + 1), cmap='gray')
  plt.title("Altered frequency")
  plt.show()

  # Inverse shift and inverse Fourier Transform to bring the image back to spatial domain
  filtrirana_filtered = np.fft.ifftshift(filtrirana_shifted)
  lena_filtered = np.fft.ifft2(filtrirana_filtered)

  # Take the real part of the inverse Fourier Transform result
  lena_filtered = np.abs(lena_filtered)

  # Display the original and filtered images
  plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.imshow(lena, cmap='gray')
  plt.title("Original Image")

  plt.subplot(1, 2, 2)
  plt.imshow(lena_filtered, cmap='gray')
  plt.title("Filtered Image")
  plt.show()