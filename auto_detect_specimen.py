import os
from skimage import io
from skimage.transform import rotate
import matplotlib.pyplot as plt

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import numpy as np

def run_fz_skimage(image_in):
    import cv2
    from skimage.segmentation import find_boundaries
    from skimage import exposure
    from skimage.morphology import binary_erosion

    img = img_as_float(image_in)
    segments_fz = felzenszwalb(img, scale=600, sigma=3, min_size=200000,channel_axis=2)
    #print(segments_fz)
    smoothed = find_boundaries(segments_fz, connectivity=1, mode='thick', background=0)
    #segments_fz = slic(img, n_segments=5, compactness=10, sigma=4,start_label=1)
    #segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    #gradient = sobel(rgb2gray(img))
    #segments_watershed = watershed(gradient, markers=250, compactness=0.001)

    #print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
    #print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
    #print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    #print(smoothed)
    #masked_image = image * segments_fz
    img[segments_fz] = 0
    ax[0, 0].imshow(mark_boundaries(img, smoothed))
    ax[0, 0].set_title("Felzenszwalbs's method")

    for (i, segVal) in enumerate(np.unique(segments_fz)):
        # construct a mask for the segment
        #print
        #"[x] inspecting segment %d" % (i)
        mask = np.zeros(img.shape[:2], dtype="uint8")
        mask[segments_fz == segVal] = 255
        # show the masked region
        ax[0, 1].imshow(cv2.bitwise_and(img, img, mask = mask))
        ax[0, 1].set_title('SLIC')


    #ax[1, 0].imshow(mark_boundaries(img, segments_quick))
    #ax[1, 0].set_title('Quickshift')
    #ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
    #ax[1, 1].set_title('Compact watershed')

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()




for filename in os.listdir("./Raw_images/"):
    if filename.endswith(".jpg"):
        print(filename)
        image = io.imread("./Raw_images/"+ filename)
        print(image.shape)
        if image.shape[0] > image.shape[1]:
            new_pic = rotate(image, 90,resize=True)
            print("rotate")
        else:
            new_pic = image
        cropped_image = new_pic[0:1150, 0:1650]
        run_fz_skimage(cropped_image)

    else:
        continue
