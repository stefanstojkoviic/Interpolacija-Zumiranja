#ukljucivanje biblioteka
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Import slike i konvertovanje je u RGB(U open-cv biblioteci default je BGR)
img = cv.imread('zaba.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#2x2 plotovi za prikaz
fig, ax = plt.subplots(2, 2, figsize=(12, 6))

scale = 2.0 #uvelicavac

#Velicina slike i broj kanala
rows, cols, kanal = img.shape
#print(img.shape)

#Nove dimenzije (dimenzije img * scale)
new_rows = int(rows * scale)
new_cols = int(cols * scale)

#Kreirana matrica sa povecanim dimenzijama i sa pocetnim vrednostima 0 (crnim pixelima)
zoomed = np.zeros((new_rows, new_cols, kanal), dtype=np.uint8)

#Zumiranje slike bez interpolacije
"""
Svaki piksel u originalnoj slici se kopira na odgovarajuce mesto u zumiranoj slici.
Kako bi se postiglo uvecanje, indeksi piksela se mnoze sa 'scale'.
Ovde nema interpolacije, pa ce slika imati puno crnih 'rupa' izmedju piksela.
"""
for i in range(rows):
    for j in range(cols):
        zoomed[int(i * scale), int(j * scale)] = img[i, j]

#Export slike:
cv.imwrite('zoomed.jpg', cv.cvtColor(zoomed, cv.COLOR_RGB2BGR))

#Originalna slika
ax[0,0].set_title('Originalna slika')
ax[0,0].imshow(img)

#Zumirana slika bez interpolacije
ax[0,1].set_title('#Zumirana slika bez interpolacije')
ax[0,1].set_xlim([1000,1600])
ax[0,1].set_ylim([500,200])
ax[0,1].imshow(zoomed)


#Interpolacija najblizeg komsije

"""
Ovde racunamo odgovarajucu vrednost piksela u originalnoj slici koristeci interpolaciju najblizeg komsije.
Za svaki piksel u zumiranoj slici, pronalazimo najblizi odgovarajuci piksel u originalnoj slici.
Ovo postizemo deljenjem koordinata piksela u zumiranoj slici promenjivom 'scale',
a zatim uzimamo celobrojni deo da bismo dobili najblizi piksel u originalnoj slici.
"""
for i in range(new_rows):
    for j in range(new_cols):
        zoomed[i,j] = img[int(i/scale), int(j/scale)]

#Zumirana slika sa interpolaciom najblizeg suseda
ax[1,0].set_title('Slika sa interpolacijom najblizeg suseda')
ax[1,0].set_xlim([1000,1600])
ax[1,0].set_ylim([500,200])
ax[1,0].imshow(zoomed)


#Zumirana slika sa bilinearnom interpolaciom
ax[1,1].set_title('Slika sa drugom interpolacijiom')
# ax[1,1].imshow()

plt.show()