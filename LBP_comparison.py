import numpy as np
from skimage import io, color
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import os
from skimage.filters import gaussian
from skimage.filters import threshold_otsu

def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def lbp_histogram(gray_image):
    #img = color.rgb2gray(color_image)
    cropped = crop_image(gray_image)
    #blurred = gaussian(cropped,2, multichannel=True, mode='reflect')
    #patterns = local_binary_pattern(cropped, 8, 5)
    #hist, _ = (np.histogram(patterns, bins=np.arange(16), density=True))
    #patterns = local_binary_pattern(cropped, 10, 10, method="ror")
    patterns = local_binary_pattern(cropped, 5, 5, method="ror")

    hist, _ = (np.histogram(patterns, bins=np.arange(16), density=True))
    print(patterns)
    return hist, patterns

def hog_histogram(gray_image):
    #img = color.rgb2gray(color_image)
    #blurred = gaussian(cropped,2, multichannel=True, mode='reflect')
    hist, hog_im = hog(gray_image, orientations=10, pixels_per_cell=(4,4), cells_per_block=(2,2), transform_sqrt = True, visualize=True,feature_vector=True)
    print(hist)
    return hist, hog_im

def kullback_leibler_divergence(p, q):
    ## For Biston classification, this comparison metric appears to perform poorly
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def test_plotting_lbp():
    ##function to generate LBP distances for some test images
    insularia = io.imread('./Raw_images/BMNH(E)504354_Geometridae_Biston_betularia_Biston betularia ab. insularia Thierry-Mieg, 1886.jpg_out.png', as_gray=True)
    carbonaria = io.imread('./Raw_images/BMNH(E)504356_Geometridae_Biston_betularia_Biston betularia ab. carbonaria Jordan, 1869.jpg_out.png', as_gray=True)
    typica = io.imread('./test_imgs/BMNH(E)1838892_Geometridae_Biston_betularia_Biston betularia (Linnaeus, 1758).jpg_out.png', as_gray=True)
    typica2 = io.imread('./test_imgs/BMNH(E)1838897_Geometridae_Biston_betularia_Biston betularia (Linnaeus, 1758).jpg_out.png', as_gray=True)


    insularia_feats, insularia_patt = lbp_histogram(insularia)
    carbonaria_feats, carbonaria_patt = lbp_histogram(carbonaria)
    typica_feats, typica_patt = lbp_histogram(typica)
    typica2_feats, typica2_patt = lbp_histogram(typica2)


    print(euclidean(carbonaria_feats, typica_feats))
    print(euclidean(insularia_feats, typica_feats))
    print(euclidean(insularia_feats, carbonaria_feats))
    print(euclidean(insularia_feats, typica2_feats))
    print(euclidean(carbonaria_feats, typica2_feats))
    print(euclidean(typica_feats, typica2_feats),"\n")

    hmax = max([carbonaria_feats.max(), insularia_feats.max(), typica_feats.max()])

    fig, ((ax1 ,ax2, ax3),(ax4 ,ax5, ax6)) = plt.subplots(nrows=2, ncols=3)

    plt.gray()



    ax1.imshow(typica2_patt)
    ax1.axis('off')
    ax4.plot(typica2_feats)
    ax4.set_ylabel('Percentage')
    ax4.set_ylim([0, hmax])

    ax2.imshow(carbonaria_patt)
    ax2.axis('off')
    ax5.plot(carbonaria_feats)
    ax5.set_ylabel('Percentage')
    ax5.set_ylim([0, hmax])

    ax3.imshow(typica_patt)
    ax3.axis('off')
    ax6.plot(typica_feats)
    ax6.set_ylabel('Percentage')
    ax6.set_ylim([0, hmax])

    plt.show()

def test_plotting_hog():
    ##function to generate HOG distances for some test images

    insularia = io.imread('./Raw_images/BMNH(E)504354_Geometridae_Biston_betularia_Biston betularia ab. insularia Thierry-Mieg, 1886.jpg_out.png')
    carbonaria = io.imread('./Raw_images/BMNH(E)504356_Geometridae_Biston_betularia_Biston betularia ab. carbonaria Jordan, 1869.jpg_out.png')
    typica = io.imread('./test_imgs/BMNH(E)1838892_Geometridae_Biston_betularia_Biston betularia (Linnaeus, 1758).jpg_out.png')
    typica2 = io.imread('./test_imgs/BMNH(E)1838897_Geometridae_Biston_betularia_Biston betularia (Linnaeus, 1758).jpg_out.png')


    insularia_feats, insularia_patt = hog_histogram(insularia)
    carbonaria_feats, carbonaria_patt = hog_histogram(carbonaria)
    typica_feats, typica_patt = hog_histogram(typica)
    typica2_feats, typica2_patt = hog_histogram(typica2)


    print(euclidean(carbonaria_feats, typica_feats))
    print(euclidean(insularia_feats, typica_feats))
    print(euclidean(insularia_feats, carbonaria_feats))
    print(euclidean(insularia_feats, typica2_feats))
    print(euclidean(carbonaria_feats, typica2_feats))
    print(euclidean(typica_feats, typica2_feats),"\n")

    hmax = max([carbonaria_feats.max(), insularia_feats.max(), typica_feats.max()])

    fig, ((ax1 ,ax2, ax3),(ax4 ,ax5, ax6)) = plt.subplots(nrows=2, ncols=3)

    plt.gray()



    ax1.imshow(typica2_patt)
    ax1.axis('off')
    ax4.plot(typica2_feats)
    ax4.set_ylabel('Percentage')
    ax4.set_ylim([0, hmax])

    ax2.imshow(carbonaria_patt)
    ax2.axis('off')
    ax5.plot(carbonaria_feats)
    ax5.set_ylabel('Percentage')
    ax5.set_ylim([0, hmax])

    ax3.imshow(typica_patt)
    ax3.axis('off')
    ax6.plot(typica_feats)
    ax6.set_ylabel('Percentage')
    ax6.set_ylim([0, hmax])

    plt.show()


def get_short_name(name):
    splitname = name.split("_")
    ID = splitname[0]
    lab = splitname[4]
    ab = lab.split(" ")
    return str(ID+"_"+ab[3])

def get_file_list(dir):
    temp_list = []
    short_list = []
    for filename in os.listdir(dir):

        if filename.endswith("out.png"):
            temp_list.append(os.path.join(dir, filename))
            short_list.append(get_short_name(os.path.join(dir, filename)))
        else:
            continue
    return temp_list, short_list

def get_lbp_hash(dir):
    lbp_dict = dict()
    for filename in os.listdir(dir):

        if filename.endswith("out.png"):
            temp_img = io.imread(os.path.join(dir, filename), as_gray=True)
            hist, patterns = lbp_histogram(temp_img)
            lbp_dict[filename] = hist
        else:
            continue
    return lbp_dict


def distance_matrix_py(pts):
    """Returns matrix of pairwise Euclidean distances. Pure Python version."""
    n = len(pts)
    m = np.zeros((n, n))
    print(end=",")
    for x in pts:
        print(get_short_name(x), end=",")
    print()
    for i in range(n):
        file_a = io.imread(pts[i], as_gray=True)
        file_a_feats, _ = lbp_histogram(file_a)
        print(get_short_name(pts[i]), end="")
        for j in range(n):
            if pts[i] != pts[j]:

                file_b = io.imread(pts[j], as_gray=True)
                file_b_feats, _ = lbp_histogram(file_b)

                s = euclidean(file_a_feats,file_b_feats)
                #print(s,get_short_name(pts[i]),get_short_name(pts[j]),sep="\t",end="\t")
                print(",",s, sep="", end="")
            else:
                s = 0
                print(",",s, sep="", end="")
            m[i, j] = s
        print()
    return m

def print_distance_matrix_from_hash(lbp_dict):
    """Returns matrix of pairwise Euclidean distances. Pure Python version."""
    n = len(lbp_dict.keys())
    m = np.zeros((n, n))
    print(end=",")
    for x in lbp_dict.keys():
        print(get_short_name(x), end=",")
    print()
    for i in lbp_dict.keys():
        file_a_feats = lbp_dict[i]
        print(get_short_name(i), end="")
        for j in lbp_dict.keys():
            if i != j:
                file_b_feats = lbp_dict[j]
                s = euclidean(file_a_feats,file_b_feats)
                #print(s,get_short_name(pts[i]),get_short_name(pts[j]),sep="\t",end="\t")
                print(",",s, sep="", end="")
            else:
                s = 0
                print(",",s, sep="", end="")
#            m[i, j] = s
        print()
    return m

#file_list, short_list = get_file_list('./Raw_images/')


#lbp_hash = get_lbp_hash('./Raw_images/')
#dist_mat = (print_distance_matrix_from_hash(lbp_hash))


#from skbio.stats.distance import DissimilarityMatrix
#dm = DissimilarityMatrix(dist_mat,short_list)
#fig = dm.plot(cmap='Reds', title='Heatmap')

#fig

#fig.show()

test_plotting_lbp()
