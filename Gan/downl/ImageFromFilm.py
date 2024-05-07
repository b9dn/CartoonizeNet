import skvideo.io
from matplotlib import pyplot as plt
from skimage.transform import resize
import skimage.filters as sif

path_to_video = "M:/data/film.mp4"
path_to_save = "M:/data/data_cart/cart/"
path_to_save_smooth = "M:/data/data_cart_smooth/film_cart_smooth/"

reader = skvideo.io.FFmpegReader(path_to_video)

sigma = 3
i = 0
ile_zapisanych = 0
size = (256,256)
for frame in reader.nextFrame():
    i += 1
    if i % 3 == 0:
        ile_zapisanych += 1
        obr = resize(frame[:,200:1000], size)
        plt.imsave(path_to_save + str(ile_zapisanych) + ".jpg", obr)
        plt.imsave(path_to_save_smooth + str(ile_zapisanych) + ".jpg", sif.gaussian(obr,sigma))
        if ile_zapisanych % 100 == 0:
            print("Zapisanych - " + str(ile_zapisanych))
