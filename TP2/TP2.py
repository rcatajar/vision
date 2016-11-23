import numpy as np
from prettytable import PrettyTable
from os import walk
import cv2
import matplotlib.pyplot as plt
import csv



# 1) - Construction des methodes de detection de gradient:


def sobelxy(src, show=False):

    img = cv2.imread(src, 0)

    seuil = 1 # must be odd and no larger than 31

    sobelx = cv2.Sobel(img,cv2.CV_8U, 1, 0, ksize=seuil)
    sobely = cv2.Sobel(img,cv2.CV_8U, 0, 1, ksize=seuil)

    sobelxy = sobelx + sobely


    sobelxy = sobelxy.astype(np.uint8)


    ret, sobelxy = cv2.threshold(sobelxy, 30, 255, cv2.THRESH_BINARY)

    if show:
        plt.subplot(121),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(sobelxy ,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

        plt.show()

    return sobelxy

def laplacian(src, show=False):
    img = cv2.imread(src, 0)

    img = cv2.GaussianBlur(img, (5,5),0)


    laplacian = cv2.Laplacian(img, cv2.CV_8U)

    ret, laplacian = cv2.threshold(laplacian, 4, 255, cv2.THRESH_BINARY)

    if show:
        plt.subplot(121),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(laplacian ,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

        plt.show()

    return laplacian


def canny(src, show):

    img = cv2.imread(src, 0)
    edges = cv2.Canny(img, 249, 255)

    if show:
        plt.subplot(121),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

        plt.show()

    return edges


def mesure_perf(image_reference, image_algorithme, seuil):

    contour_detectes = 0
    contour_reference = 0
    contour_correct = 0
    faux_positifs = 0
    faux_negatifs = 0
    vrais_negatifs = 0

    for i in range(len(image_algorithme)):
        for j in range(len(image_algorithme[i])):
            if image_algorithme[i][j] > seuil:
                contour_detectes += 1
            if image_reference[i][j] > seuil:
                contour_reference += 1
            if image_reference[i][j] >= seuil and image_algorithme[i][j] >= seuil:
                contour_correct += 1
            if image_reference[i][j] < seuil and image_algorithme[i][j] >= seuil:
                faux_positifs += 1
            if image_reference[i][j] >= seuil and image_algorithme[i][j] < seuil:
                faux_negatifs += 1
            if image_reference[i][j] < seuil and image_algorithme[i][j] < seuil:
                vrais_negatifs += 1

    return contour_detectes, contour_reference, contour_correct, faux_positifs, faux_negatifs, vrais_negatifs

def comparateur_methodes(seuil):

    f = []
    for (dirpath, dirnames, filenames) in walk("contour/"):
        f.extend(filenames)
        break


    liste_noms_contours = []
    liste_noms_image = []

    for filename in filenames:
        if filename == '.DS_Store':
            pass
        elif filename[-7:] == ".gt.jpg":
            liste_noms_contours.append('contour/' + filename)
        else:
            liste_noms_image.append('contour/' + filename)

    table = PrettyTable(["images", "Canny", ".", " . ", "Sobel", ". ", " .",  "Laplacian", " .  ", "  . ", "Prewitt", ' ', '  '])
    table.add_row(["  ", "P1", "TFP1", "TFN1", "P2", "TFP2", "TFN2", "P3", "TFP3", "TFN3", "P4", "TFP4", "TFN4"])
    tableau = ["images", "Canny", ".", " . ", "Sobel", ". ", " .",  "Laplacian", " .  ", "  . ", "Prewitt", ' ', '  ']
    tableau.append(["  ", "P1", "TFP1", "TFN1", "P2", "TFP2", "TFN2", "P3", "TFP3", "TFN3", "P4", "TFP4", "TFN4"])

    for idx, src in enumerate(liste_noms_image):

        source_contour = liste_noms_image[idx]

        image_canny = canny(src, False)
        image_sobel = sobelxy(src, False)
        image_laplacian = laplacian(src, False)
        image_prewitt = prewitt(src, False)

        images_a_tester = [image_canny, image_sobel, image_laplacian, image_prewitt]

        image_reference = cv2.imread(source_contour, 0)
        ret, image_reference = cv2.threshold(image_reference, seuil, 255, cv2.THRESH_BINARY)

        resultat = []
        for idx, image in enumerate(images_a_tester):
            contour_detectes, contour_reference, contour_correct, faux_positifs, faux_negatifs, vrais_negatifs = mesure_perf(image_reference, image, seuil)

            print src
            print contour_detectes, contour_reference, contour_correct, faux_positifs, faux_negatifs, vrais_negatifs

            try:
                P = 1. * contour_correct / (contour_correct + faux_positifs)
            except:
                P = 0

            try:
                TFP = 1. * faux_positifs / (faux_positifs + vrais_negatifs)
            except:
                TFP = 0

            try:
                TFN = 1. * faux_negatifs / (contour_correct + faux_negatifs)
            except:
                TFN = 0

            resultat = resultat + [P, TFP, TFN]

        table.add_row([src] + resultat)
        tableau.append([src] + resultat)
    print table

    with open("testcsv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(table)

def get_matrix(img, i, j):
    height, width = img.shape
    if i < 1 or j < 1 or i >= height -1 or j >= width -1:
        return np.zeros((3, 3))
    else:
        return np.array([[img[i - 1 + p][j - 1 + k] for k in range(3)] for p in range(3)])


def prewitt(src, show=False):
    img = cv2.imread(src, 0)
    img_edged = np.zeros(img.shape)

    prewitt_matrix_moyenneur =  np.array(   [[-1, -1, -1],
                                            [0, 0, 0   ],
                                            [1, 1, 1  ]])

    prewitt_matrix_derivee =   np.array(    [[-1, 0, 1],
                                            [-1, 0, 1],
                                            [-1, 0, 1]])


    height, width = img.shape


    for i in range(height):
            for j in range(width):
                img_edged[i][j] = abs(np.vdot(get_matrix(img, i, j), prewitt_matrix_moyenneur) + \
                                np.vdot(get_matrix(img, i, j), prewitt_matrix_derivee))


    img_edged = img_edged.astype(dtype=np.uint8)

    # Trois options differentes ensuite pour varier le treshholding.

    ret, img_edged = cv2.threshold(img_edged, 90, 255, cv2.THRESH_BINARY)
    #img_edged = cv2.adaptiveThreshold(img_edged ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #img_edged = cv2.adaptiveThreshold(img_edged,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 5 , 0)

    if show:
        plt.subplot(121),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img_edged,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

        plt.show()

    return img_edged

#prewitt("contour/300091.jpg", show=True)
#sobelxy("contour/300091.jpg", show=True)
#canny("contour/300091.jpg", show=True)
#laplacian("contour/300091.jpg", show=True)

comparateur_methodes(150)

