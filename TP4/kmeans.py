import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
# Importation de l'image
from sklearn.datasets import load_sample_image

image = load_sample_image("china.jpg")
image = np.array(image, dtype=np.float64) / 255
# Affichage
plt.figure()
plt.clf()
plt.axis("off")
plt.title("Image originale")
plt.imshow(image)

#plt.show()

w, h, d = original_shape = tuple(image.shape)
im_size = w*h
assert d == 3
image_array = np.reshape(image, (w * h, d))
print(w,h,d)

cls = KMeans(n_clusters=3)
print(image_array)
cls.fit(image_array[0,1], image_array[2])

print(cls.cluster_centers_)