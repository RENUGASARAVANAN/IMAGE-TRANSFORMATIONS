# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import the necessary libraries and read the original image and save it as a image variable.

### Step2:
Translate the image using a function warpPerpective()

### Step3:
Scale the image by multiplying the rows and columns with a float value.

### Step4:
Shear the image in both the rows and columns.

### Step5:
Find the reflection of the image.

### Step6:
Rotate the image using angle function.

## Program:

### Developed By:RENUGA S
### Register Number:212222230118

```
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('tri.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
```
### OUTPUT:
![Screenshot 2024-10-03 105724](https://github.com/user-attachments/assets/a9a70c6b-fce3-4846-a738-b96bc12474e5)

### i)Image Translation
```
rows, cols, _ = image.shape
M_translate = np.float32([[1, 0, 50], [0, 1, 100]])  # Translate by (50, 100) pixels
translated_image = cv2.warpAffine(image_rgb, M_translate, (cols, rows))

plt.imshow(translated_image)
plt.title("Translated Image")
plt.axis('off')
```

### OUTPUT:
![Screenshot 2024-10-03 105732](https://github.com/user-attachments/assets/269e1e8f-27f1-4d63-8f1f-a0c42cdc22a3)


### ii) Image Scaling
```
scaled_image = cv2.resize(image_rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

plt.imshow(scaled_image)
plt.title("Scaled Image")
plt.axis('off')
```
### OUTPUT:
![Screenshot 2024-10-03 105742](https://github.com/user-attachments/assets/81eaee74-73dc-4a9f-9e8d-896a37bc22ac)


### iii)Image shearing
```
M_shear = np.float32([[1, 0.5, 0], [0.5, 1, 0]])  # Shear with factor 0.5
sheared_image = cv2.warpAffine(image_rgb, M_shear, (int(cols * 1.5), int(rows * 1.5)))

plt.imshow(sheared_image)
plt.title("Sheared Image")
plt.axis('off')
```
### OUTPUT:
![Screenshot 2024-10-03 105749](https://github.com/user-attachments/assets/03e6f8ba-d971-4373-acec-a6c31a16b9cf)


### iv)Image Reflection
```
reflected_image = cv2.flip(image_rgb, 1)

plt.imshow(reflected_image)
plt.title("Reflected Image")
plt.axis('off')
```

### OUTPUT:
![Screenshot 2024-10-03 105756](https://github.com/user-attachments/assets/9119f5c0-cd1d-4e08-b0ec-bca8e8d15dd5)

### v)Image Rotation
```
M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)  # Rotate by 45 degrees
rotated_image = cv2.warpAffine(image_rgb, M_rotate, (cols, rows))

plt.imshow(rotated_image)
plt.title("Rotated Image")
plt.axis('off')
```
### OUTPUT:
![Screenshot 2024-10-03 105809](https://github.com/user-attachments/assets/ffc45741-ce3a-444a-9ae4-fb2a658775f0)



### vi)Image Cropping
```
cropped_image = image_rgb[50:300, 100:400]
plt.figure(figsize=(12, 8))
plt.tight_layout()
plt.show()


plt.figure(figsize=(4, 4))
plt.imshow(cropped_image)
plt.title("Cropped Image")
plt.axis('off')
plt.show()
```

### OUTPUT:
![Screenshot 2024-10-03 105818](https://github.com/user-attachments/assets/2d41a19b-59dd-4176-a6b6-6af6c2839b77)

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
