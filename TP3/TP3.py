# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

filename1 = 'points/bikes/img_7073.ppm'
filename2 = 'points/bikes/img_7075.ppm'
filename3 = 'points/bikes/img_7077.ppm'
graf1 = 'points/graffiti/img1.ppm'
graf2 = 'points/graffiti/img2.ppm'
boat1 = 'points/boat/img0.pgm'
boat2 = 'points/boat/img1.pgm'
michelangelo_origin = 'points/dataset_MorelYu09/Absolute_Tilt_Tests/painting_zoom_x10/adam_zoom10_front.pgm'
michelangelo_tilt = 'points/dataset_MorelYu09/Absolute_Tilt_Tests/painting_zoom_x10/adam_zoom10_45degR.pgm'


# Question 1.1
def corner_harris(source):
    img = cv2.imread(source)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('dst', img)
    # plt.show()
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    return dst


# Question 1.1
def corner_sift(source):
    img = cv2.imread(source)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)

    img = cv2.drawKeypoints(gray, kp, None)

    plt.imshow(img)
    plt.show()

    return kp


# Question 1.1 (celui qu'on utilise effectivement dans la suite)
def corner_shi_tomasi(source):
    """
    :param source: le chemin de l'image
    :return: la liste des points selon la methode de shi_thomasi
    """

    img = cv2.imread(source)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    return corners


# Question 1.2
def calcul_similarite(source1, source2):
    """

    :param source1: le chemin de l'image de gauche
    :param source2: le chemin de l'image de droite
    :return: la fenetre de plus grande similarite entre les deux images
    """

    # Dans un premier temps on recupere la liste de tous les points
    # d'importance des images 1 et 2

    corners1 = corner_shi_tomasi(source1)
    corners2 = corner_shi_tomasi(source2)

    img1 = cv2.imread(source1, 0)
    img2 = cv2.imread(source2, 0)

    for corner in corners1:
        x, y = corner.ravel()
        cv2.circle(img1, (x, y), 3, 255, -1)
    for corner in corners2:
        x, y = corner.ravel()
        cv2.circle(img2, (x, y), 3, 255, -1)

    n = 30
    p = 30

    #  Puis pour chaque point des listes corners, on fait une fenetre de taille (2N+1)x(2P+1)
    # On stocke ces fenetres dans deux listes contenant les intensites des
    # points de cette fenetre
    list_windows1 = window_around(img1, corners1, n, p)
    list_windows2 = window_around(img2, corners2, n, p)

    # On initialise le maximum avec une valeur tres elevee afin d'etre sur
    # d'obtenir un minima par la suite
    max_similarity = 10000000000

    #  On va ensuite calculer la similarite de chaque combinaison possible de fenetre de corner dans les deux images
    # On ne gardera que les points extremes de chaque fenetre minimisant la
    # dissimilarite
    for window1, point11, point12 in list_windows1:
        for window2, point21, point22 in list_windows2:
            similarity = cost(window1, window2, "SAD")
            if similarity < max_similarity:
                max_similarity = similarity
                image_1_top_left = point11
                image_1_bot_right = point12
                image_2_top_left = point21
                image_2_bot_right = point22

                plot_windows(source1, source2, image_1_top_left,
                            image_1_bot_right, image_2_top_left, image_2_bot_right)


# Question 1.2
def plot_windows(source1, source2, image_1_top_left, image_1_bot_right, image_2_top_left, image_2_bot_right):
    """

    :param source1: La source de l'image de gauche
    :param source2: La source de l'image de droite

    Les autres parmetres sont les point extreme des deux rectangles a afficher
    :return:
    """

    #  On dessine des rectanges pour montrer la location de ces fenetres

    img1 = cv2.imread(source1, 0)
    img2 = cv2.imread(source2, 0)

    cv2.rectangle(img1, image_1_top_left, image_1_bot_right, (255, 0, 0), 5)
    cv2.rectangle(img2, image_2_top_left, image_2_bot_right, (255, 0, 0), 5)

    #  On affiche ces deux fenetres
    plt.title("Meilleure similarite entre les images")
    plt.subplot(121), plt.imshow(img1, cmap='gray')
    plt.title('Image 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.title('Image 2'), plt.xticks([]), plt.yticks([])

    plt.show()


# Question 1.2
def cost(window1, window2, type="SSD"):
    """

    :param window1: la fenetre de l'image de gauche
    :param window2: la fenetre de l'image de droite
    :param type: le type de cout que l'on souhaite utiliser ici. Seulement SSD et SAD sont implementes
    :return: resultat de la fonction de dissimilarite
    """

    if not len(window1) == len(window2) and len(window1[0]) == len(window2[0]):
        print "Les fenetres n'ont pas la meme taille"
        return KeyError

    sum = 0

    for j in range(len(window1)):
        for i in range(len(window1[0])):
            if type == "SSD":
                sum += (window1[j][i] - window2[j][i]) ** 2
            elif type == "SAD":
                sum += abs(window1[j][i] - window2[j][i])
            else:
                print "L'argument type n'est pas bon dans la fonction cost"
                raise KeyError

    return sum


# Question 1.2
def window_around(img, corners, n, p):
    """
        Cette fonction prend en argument    img = l'image initiale
                                            corners = une liste de coordonnees de points autour desquelles on veut une fenetre
                                            n = la largeur de la fenetre
                                            p = la hauteur de la fenetre
    """

    list_of_windows = []

    size_y, size_x = img.shape

    for corner in corners:
        x = corner[0][0]
        y = corner[0][1]

        # On verifie qu'on ne sort pas de l'image
        if x > n and y > p and x + n < size_x and y + p < size_y:
            window = []

            for j in range(2*p + 1):
                window.append([img[j - p][x - i - n] for i in range(2*n + 1)])

            # list_of_windows.append([window, (y-p, x-n), (y+p+1, x+n+1)])
            list_of_windows.append(
                [window, (x - n, y - p), (x + n + 1, y + p + 1)])

    return list_of_windows


# Question 1.3
def bfmatcher(source1, source2):
    img1 = cv2.imread(source1, 0)  # queryImage
    img2 = cv2.imread(source2, 0)  # trainImage


    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

    plt.imshow(img3), plt.show()


# Question 1.3
def flannmatcher(source1, source2):
    # FIXME: soucis sur Ubuntu, OpenCV 3.1.0. Il semblerait que ce soit un bug
    # https://github.com/Itseez/opencv/issues/5667
    # Update : même soucis sur Mac. C'est bien lié à OpenCV

    img1 = cv2.imread(source1, 0)  # queryImage
    img2 = cv2.imread(source2, 0)  # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    plt.imshow(img3, ), plt.show()


# Question 2
def homemade_harris(source):
    img = cv2.imread(source, 0)

    # On effectue des transformation de Sobel pour garder les gradients selon
    # x et y

    seuil = 5
    Ix = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=seuil)
    Iy = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=seuil)

    # On calcule puis on lisse Ix2 , Ix2 et Ixy avec un filtre gaussien
    smoothing_factor = (5, 5)

    Ix2 = cv2.GaussianBlur(np.multiply(Ix, Ix), smoothing_factor, 0)
    Iy2 = cv2.GaussianBlur(np.multiply(Iy, Iy), smoothing_factor, 0)
    Ixy = cv2.GaussianBlur(np.multiply(Ix, Iy), smoothing_factor, 0)

    #  On utilise la formule de Harris pour obtenir une nouvelle image
    det = np.multiply(Ix2, Iy2) - np.multiply(Ixy, Ixy)
    trace = np.add(Iy2, Ix2)

    image_harrised = det - 0.04 * np.array(np.multiply(trace, trace))

    #  Maintenant qu'on a une image de harris, on va lui faire quelques modifications
    #  On met les points sur les bords et les points d'intensite negative a 0
    #  Et on selectionne les maxima locaux sur une largeur de 3

    new_image_harris = []

    for j in range(len(image_harrised)):
        line = []
        for i in range(len(image_harrised[0])):
            #  Si on est sur un bord, on met la valeur 0
            if image_harrised[j][i] < 0 or i <= 1 or j <= 1 or i >= len(image_harrised[0]) - 1 or j >= len(
                    image_harrised) - 1:
                line.append(0)
            else:
                # Sinon, on calcule si la valeur actuelle est plus elevee que
                # toutes celles autour.
                if image_harrised[j][i] > max(max([image_harrised[j - 1][i - 1 + k] for k in range(3)]),
                                              max([image_harrised[j][i - 1 + k]
                                                   for k in range(3) if k != 1]),
                                              max([image_harrised[j + 1][i - 1 + k] for k in range(3)])):
                    line.append(image_harrised[j][i])

                else:
                    line.append(0)

        new_image_harris.append(line)

    new_image_harris = np.array(new_image_harris)

    #  On obtient une nouvelle image apres transformation. On va maintenant prendre tous les points
    #  d'intensite non negative, et les trier dans une liste

    liste_points = []

    for j in range(len(new_image_harris)):
        for i in range(len(new_image_harris[0])):
            if new_image_harris[j][i] > 0:
                liste_points = tri_insertion(
                    liste_points, (new_image_harris[j][i], i, j))

    # J'affiche ensuite l'image initiale avec les 50 points les plus
    # importants:
    liste_points = liste_points[:50]

    #  Je trace des petits cercles autour de ces points importantss
    for point in liste_points:
        x = point[1]
        y = point[2]
        cv2.circle(img, (x, y), 3, 255, 4)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Image initiale'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(new_image_harris, cmap='gray')
    plt.title('Image avec transformation de Harris'), plt.xticks(
        []), plt.yticks([])

    plt.show()


def tri_insertion(liste, (intensite, x, y)):
    #  Tri insertion suppose que la liste en argument est deja triee
    if len(liste) == 0:
        return [(intensite, x, y)]
    else:
        for i in range(len(liste)):
            if intensite > liste[i][0]:
                return liste[:i] + [(intensite, x, y)] + liste[i:]

        return liste + [(intensite, x, y)]


# calcul_similarite(michelangelo_origin, michelangelo_tilt)
calcul_similarite(filename2, filename1)
# homemade_harris(filename2)

# corner_harris(filename1)

#corner_shi_tomasi(filename2)
#corner_sift(michelangelo_origin)

#bfmatcher(michelangelo_origin, michelangelo_tilt)
#flannmatcher(michelangelo_origin, michelangelo_tilt)
