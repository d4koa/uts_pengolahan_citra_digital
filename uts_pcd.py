'''
    Program : Pengolahan citra
    Tugas   : UTS
    Matkul  : Pengolahan citra digital
'''

'''
    Tugas UTS PCD
    Lakukan pengolahan pada 1 citra/gambar sebagai berikut!
    a. Ambil RoI pada sebuah gambar
    b. Lakukan image acquisition pada RoI tersebut (translasi/rotasi/resize)
    c. Lakukan image enhancement (thresholding/inverted/contrass&brightness)
    d. Lakukan image filtering (mean blur/median blur/gaussian blur/sharpening)
'''

# Catatan :
# Buka/tutup komentar ya untuk memilah program mana yang akan dijalankan!

''' Program Utama '''
# Import Library
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# End of Import Library

# Deklarasi beberapa variabel utama
imgPath = "pompom.jpg" # path gambar

# Variabel gambar
oriImg = cv.imread(imgPath) # var gambar original
img = cv.cvtColor(oriImg, cv.COLOR_BGR2RGB) # var gambar convert RGB

# Variabel gambar grayscale
abuImg = cv.imread(imgPath, cv.IMREAD_GRAYSCALE) # var gambar original abu
abu = cv.cvtColor(abuImg, cv.COLOR_BGR2RGB) # var gambar abu convert RGB
# End of Deklarasi beberapa variabel utama

'''Answer a'''
# Deklarasi variabel untuk soal-a
# Variabel gambar Region Of Interest (roi)
# variabel Koordinat
x1 = 50;  x2 = 450; y1 = 200; y2 = 600;
x3 = 480; x4 = 680; y3 = 200; y4 = 400;

# Menentukan roi
roi1 = img[x1:x2, y1:y2] # var roi1 original
roi2 = img[x3:x4, y3:y4] # var roi2 original
roi3 = abu[x1:x2, y1:y2] # var roi3 abu
roi4 = abu[x3:x4, y3:y4] # var roi4 abu
# End of Deklarasi variabel untuk soal-a

'''Sesi a'''
# Showing image soal-a
# Gambar original
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title('Original')

# Gambar original roi1
plt.subplot(2, 3, 2)
plt.imshow(roi1)
plt.title('Original Roi1')

# Gambar original roi2
plt.subplot(2, 3, 3)
plt.imshow(roi2)
plt.title('Original Roi2')

# Gambar original abu
plt.subplot(2, 3, 4)
plt.imshow(abu)
plt.title('Original Abu')

# Gambar original abu roi3
plt.subplot(2, 3, 5)
plt.imshow(roi3)
plt.title('Abu Roi3')

# Gambar original abu roi4
plt.subplot(2, 3, 6)
plt.imshow(roi4)
plt.title('Abu Roi4')
plt.show()
# End of Showing image soal-a
'''End of Answer a'''

'''Answer b'''
# Deklarasi beberapa variabel untuk soal-b
# Image Acquisition
# Mendeklarasikan variabel penampung img
srcB = roi1

# Mendapatkan ukuran img
print(f'Ukuran gambar adalah {srcB.shape}')

# Mendapatkan nilai tinggi dan lebar
tinggi, lebar = srcB.shape[:2]

# nilai x=0 dan y=0 dipindah ke mana -> translasi
tx1 = tinggi/2; ty1 = lebar*0
tx2 = (-tinggi/2); ty2 = lebar*0

# Matrix -> translasi
M1 = np.float32([[1, 0, tx1], 
                 [0, 1, ty1]])
M2 = np.float32([[1, 0, tx2], 
                 [0, 1, ty2]])

# Variabel Translasi
translasi1 = cv.warpAffine(srcB, M1, (lebar, tinggi))
translasi2 = cv.warpAffine(srcB, M2, (lebar, tinggi))

# Menentukan nilai center -> rotasi
center1 = (tinggi/2, lebar/2)
center2 = ((tinggi/2)+50, (lebar/2)+50)

# Menentukan derajat rotasi ->rotasi
mat1 = cv.getRotationMatrix2D(center1, 90, 1)
mat2 = cv.getRotationMatrix2D(center2, 180, 1)

# Variabel Rotasi
rotasi1 = cv.warpAffine(srcB, mat1, (lebar, tinggi))
rotasi2 = cv.warpAffine(srcB, mat2, (lebar, tinggi))

# Variabel Resize
resize1 = cv.resize(src=srcB, dsize=(int(lebar/2), int(tinggi/2)), interpolation=cv.INTER_AREA)
resize2 = cv.resize(src=srcB, dsize=(int(lebar*2), int(tinggi*2)), interpolation=cv.INTER_AREA)

# Melihat ukuran resize
# print(f"Ukuran resize1 = {resize1.shape}")
# print(f"Ukuran resize2 = {resize2.shape}")

# Deklarasi beberapa variabel untuk soal-b

# Showing image soal-b
# Gambar roi1
plt.subplot(2, 4, 1)
plt.imshow(srcB)
plt.title('Original Roi1')

# Gambar translasi1 roi1
plt.subplot(2, 4, 2)
plt.imshow(translasi1)
plt.title('Translasi1 Roi1')

# Gambar translasi2 roi1
plt.subplot(2, 4, 3)
plt.imshow(translasi2)
plt.title('Translasi2 Roi1')

# Gambar rotasi1 roi1
plt.subplot(2, 4, 5)
plt.imshow(rotasi1)
plt.title('Rotasi1 Roi1')

# Gambar rotasi2 roi1
plt.subplot(2, 4, 6)
plt.imshow(rotasi2)
plt.title('Rotasi2 Roi1')

# Gambar resize1 roi1
plt.subplot(2, 4, 7)
plt.imshow(resize1)
plt.title('Resize1 Roi1')

# Gambar resize1 roi1
plt.subplot(2, 4, 8)
plt.imshow(resize2)
plt.title('Resize2 Roi1')
plt.show()
# End of Showing image soal-b
'''End of Answer b'''

'''Answer c'''
# Deklarasi beberapa variabel untuk soal-c
# Image Enchanment
# Mendekralasikan variabel penampung img
srcC = abu #harus abu ya

# Mendeklarasikan variabel contrast dan brightness -> contrast dan brightness
contrast1 = 1.5; contrast2 = 2 # control 0 - 127
brightness1 = 1; brightness2 = 20 # control 0 - 100

# Variabel Treshold
ret1, threshold1 = cv.threshold(srcC, 127, 255, cv.THRESH_BINARY)
ret2, threshold2 = cv.threshold(srcC, 127, 255, cv.THRESH_BINARY_INV)
ret3, threshold3 = cv.threshold(srcC, 127, 255, cv.THRESH_TOZERO)
ret4, threshold4 = cv.threshold(srcC, 127, 255, cv.THRESH_TOZERO_INV)

# Variabel Inverted
inverted = 255 - srcC

# Variabel Contrast and Brightness
constrast_brightness1 = cv.addWeighted(srcC, contrast1, srcC, 1, brightness1)
constrast_brightness2 = cv.addWeighted(srcC, contrast2, srcC, 1, brightness2)
# End of Deklarasi beberapa variabel untuk soal-c

# Showing image soal-c
# Gambar Original
plt.subplot(2, 4, 1)
plt.imshow(srcC)
plt.title('Original')

# Gambar Contrast dan Brightness
plt.subplot(2, 4, 2)
plt.imshow(constrast_brightness1)
plt.title('Contrast dan Brightness')

# Gambar Contrast dan Brightness
plt.subplot(2, 4, 3)
plt.imshow(constrast_brightness2)
plt.title('Contrast dan Brightness')

# Gambar Inverted
plt.subplot(2, 4, 4)
plt.imshow(inverted)
plt.title('Inverted')

# Gambar Threshold
plt.subplot(2, 4, 5)
plt.imshow(threshold1)
plt.title('Threshold Binary')

# Gambar Threshold
plt.subplot(2, 4, 6)
plt.imshow(threshold2)
plt.title('Threshold Binary Inverted')

# Gambar Threshold
plt.subplot(2, 4, 7)
plt.imshow(threshold3)
plt.title('Threshold Tozero')

# Gambar Threshold
plt.subplot(2, 4, 8)
plt.imshow(threshold4)
plt.title('Threshold Binary Inverted')
plt.show()
# End of Showing image soal-c
'''End of Answer c'''

'''Answer d'''
# Deklarasi beberapa variabel untuk soal-d
# Image Filtering
# Mendekralasikan variabel penampung img
srcD = roi1

# Kernel -> blur dan sharpening
# Matrix 3x3
kernel3x3 = np.array([[1/9, 1/9, 1/9],
                      [1/9, 1/9, 1/9],
                      [1/9, 1/9, 1/9]])

# Matrix 5x5
kernel5x5 = np.array([[1/25, 1/25, 1/25, 1/25, 1/25],
                      [1/25, 1/25, 1/25, 1/25, 1/25],
                      [1/25, 1/25, 1/25, 1/25, 1/25],
                      [1/25, 1/25, 1/25, 1/25, 1/25],
                      [1/25, 1/25, 1/25, 1/25, 1/25]])

# Matrix 3x3 Sharpening
sharp_kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

# Variabel Mean Blur
mean_blur1 = cv.filter2D(src=srcD, ddepth=-1, kernel=kernel3x3)
mean_blur2 = cv.filter2D(src=srcD, ddepth=-1, kernel=kernel5x5)

# Variabel Median Blur
median_blur1 = cv.medianBlur(src=srcD, ksize=3)
median_blur2 = cv.medianBlur(src=srcD, ksize=5)

# Variabel Gaussian Blur
gaussian_blur1 = cv.GaussianBlur(src=srcD, ksize=(3,3), sigmaX=0, sigmaY=0)
gaussian_blur2 = cv.GaussianBlur(src=srcD, ksize=(5,5), sigmaX=0, sigmaY=0)

# Variabel Sharpening
sharpening = cv.filter2D(src=srcD, ddepth=-1, kernel=sharp_kernel)
# End of Deklarasi beberapa variabel untuk soal-d

# Showing image soal-d
# Gambar Original
plt.subplot(2, 4, 1)
plt.imshow(srcD)
plt.title('Original')

# Gambar Sharpening
plt.subplot(2, 4, 2)
plt.imshow(sharpening)
plt.title('Sharpening')

# Gambar Mean Blur 3x3
plt.subplot(2, 4, 3)
plt.imshow(mean_blur1)
plt.title('Mean Blur 3x3')

# Gambar Mean Blur 5x5
plt.subplot(2, 4, 4)
plt.imshow(mean_blur2)
plt.title('Mean Blur 5x5')

# Gambar Median Blur 3x3
plt.subplot(2, 4, 5)
plt.imshow(median_blur1)
plt.title('Median Blur 3x3')

# Gambar Median Blur 5x5
plt.subplot(2, 4, 6)
plt.imshow(median_blur2)
plt.title('Median Blur 5x5')

# Gambar Gaussian Blur 3x3
plt.subplot(2, 4, 7)
plt.imshow(gaussian_blur1)
plt.title('Gaussian Blur 3x3')

# Gambar Gausssian Blur 5x5
plt.subplot(2, 4, 8)
plt.imshow(gaussian_blur2)
plt.title('Gaussian Blur 5x5')
plt.show()
# End of Showing image soal-d
'''End of Answer d'''
''' End of Program Utama '''
