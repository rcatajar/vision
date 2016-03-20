# coding=utf-8
import cv2

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter

bike1 = 'points/bikes/img_7073.ppm'
bike2 = 'points/bikes/img_7075.ppm'
bike3 = 'points/bikes/img_7077.ppm'
graf1 = 'points/graffiti/img1.ppm'
graf2 = 'points/graffiti/img2.ppm'
boat1 = 'points/boat/img0.pgm'
boat2 = 'points/boat/img1.pgm'
michelangelo_origin = 'points/dataset_MorelYu09/Absolute_Tilt_Tests/painting_zoom_x10/adam_zoom10_front.pgm'
michelangelo_tilt = 'points/dataset_MorelYu09/Absolute_Tilt_Tests/painting_zoom_x10/adam_zoom10_45degR.pgm'


class ImageMatcher:
    """
    Question 1
    """
    def __init__(self, source1, source2, corner_method="shi_tomasi"):

        # On importe les images, et on les convertit directement en nuances de gris
        self.img1 = cv2.imread(source1, 0)
        self.img2 = cv2.imread(source2, 0)

        # Pour passer la méthode qu'on veux utiliser, on utilise un dictionnaire
        switcher = {
            0: self.corner_harris,
            1: self.corner_sift,
            2: self.corner_shi_tomasi,
        }

        if corner_method == "harris":
            switch = 0
        elif corner_method == "sift":
            switch = 1
        elif corner_method == "shi_tomasi":
            switch = 2
        else:
            print "Mauvaise corner_method"

        # On prend la fonction du dictionnaire
        self.corner_method = switcher.get(switch, lambda: "nothing")

    def calcul_similarite(self, n=20, p=30):
        """
        Question 1.2

        Trace les fenêtres de plus grandes similarités entre les deux images.

        Parameters
        ----------
        n: largeur des fenêtres
        p: hauteur des fenêtres
        """

        # Dans un premier temps on récupère la liste de tous les points d'importance des images 1 et 2
        corners1 = self.corner_method(self.img1)
        corners2 = self.corner_method(self.img2)

        for corner in corners1:
            x, y = corner.ravel()
            cv2.circle(self.img1, (x, y), 1, 255, -1)
        for corner in corners2:
            x, y = corner.ravel()
            cv2.circle(self.img2, (x, y), 1, 255, -1)

        # Puis pour chaque point des listes corners, on fait une fenêtre de taille (2N+1)x(2P+1)
        # On stocke ces fenêtres dans deux listes contenant les intensités des
        # points de cette fenêtre

        blur1 = cv2.blur(self.img1, (3, 3), 0)
        blur2 = cv2.blur(self.img2, (3, 3), 0)

        list_windows1 = self.window_around(blur1, corners1, n, p)
        list_windows2 = self.window_around(blur2, corners2, n, p)

        # On initialise le maximum avec une valeur très élevée afin d'être sûr
        # d'obtenir un minimum par la suite
        max_similarity = 10000000000

        c = 0

        # On va ensuite calculer la similarité de chaque combinaison possible de fenêtre de corner dans les deux images
        # On ne gardera que les points extrêmes de chaque fenêtre minimisant la dissimilarité
        for window1, point11, point12 in list_windows1:
            for window2, point21, point22 in list_windows2:

                similarity = self.cost(window1, window2)
                if similarity < max_similarity:
                    max_similarity = similarity
                    print max_similarity
                    image_1_top_left = point11
                    image_1_bot_right = point12
                    image_2_top_left = point21
                    image_2_bot_right = point22

                    # On plot en dessous d'un certain seuil afin d'afficher plusieurs points de similarité
                    if max_similarity < 150000:
                        self.plot_windows(self.img1, self.img2, image_1_top_left, image_1_bot_right, image_2_top_left,
                              image_2_bot_right)

    @staticmethod
    def corner_harris(source):
        """
        Question 1.1

        Parameters
        ----------
        source: l'image initiale

        Returns
        -------
        corners: liste de coordonées des points d'intérêt
        """
        # On récupère les distances minimales pour chaque point de l'image
        dst = cv2.cornerHarris(source, 2, 3, 0.04)

        # Seuil empirique pour définir les valeurs que l'on garde
        dst[dst < 0.001 * dst.max()] = 0

        corners = []
        for index, x in np.ndenumerate(dst):
            if x > 0.00001:
                corners.append([[index[1], index[0]]])

        corners = np.array(corners)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(source, (x, y), 3, 255, -1)

        return corners

    @staticmethod
    def corner_sift(source):
        """
        Question 1.1

        Parameters
        ----------
        source: l'image initiale

        Returns
        -------
        corners: liste de coordonées des points d'intérêt
        """
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(source, None)

        corners = []
        for point in kp:
            corners.append([[int(point.pt[0]), int(point.pt[1])]])

        corners = np.array(corners)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(source, (x, y), 3, 255, -1)

        return corners

    @staticmethod
    def corner_shi_tomasi(source):
        """
        Question 1.1

        Parameters
        ----------
        source: l'image initiale

        Returns
        -------
        corners: liste de coordonées des points d'intérêt
        """
        corners = cv2.goodFeaturesToTrack(source, 25, 0.01, 10)
        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(source, (x, y), 3, 255, -1)

        return corners

    @staticmethod
    def cost(window1, window2, type="SAD"):
        """
        Question 1.2

        Calcul du coût pour deux fenêtres données. Plusieurs types de fonction possibles.

        Parameters
        ----------
        window1: la fenêtre de l'image de gauche
        window2 la fenêtre de l'image de droite
        type: le type de coût que l'on souhaite utiliser ici. Seuls SSD et SAD sont implementés

        Returns
        -------
        sum: résultat de la fonction de dissimilarité
        """

        if not len(window1) == len(window2) and len(window1[0]) == len(window2[0]):
            print "Les fenêtres n'ont pas la même taille"
            return KeyError

        sum = 0.

        if type == "SSD":
            sum += np.sum(np.power(window1-window2, 2))
        elif type == "SAD":
            sum += np.sum(window1-window2)

        else:
            print "L'argument type n'est pas bon dans la fonction cost"
            raise KeyError

        return sum

    @staticmethod
    def window_around(source, corners, n, p):
        """
        Question 1.2

        Parameters
        ----------
        source: l'image initiale
        corners: une liste de coordonnées de points autour desquelles on veut une fenêtre
        n: la largeur de la fenêtre
        p: la hauteur de la fenêtre

        Returns
        -------
        list_of_windows: liste sous la forme [fenêtre, coin_haut_gauche(x,y), coin_bas_droit(x,y)],
            "fenêtre" étant un tableau numpy de taille (2N+1)x(2P+1)
        """

        list_of_windows = []

        size_y, size_x = source.shape

        for corner in corners:
            x = corner[0][0]
            y = corner[0][1]

            # On vérifie qu'on ne sort pas de l'image
            if x > n and y > p and x + n < size_x and y + p < size_y:
                window = source[y - p:y + p + 1, x - n:x + n + 1]

                list_of_windows.append(
                    [window, (x - n, y - p), (x + n, y + p)])

        return list_of_windows

    @staticmethod
    def plot_windows(source1, source2, image_1_top_left, image_1_bot_right, image_2_top_left, image_2_bot_right):
        """
        Question 1.2

        Affiche les deux images sources avec les fenêtres de similarité

        Parameters
        ----------
        source1: La source de l'image de gauche
        source2: La source de l'image de droite
        Les autres parmètres sont les points extrêmes des deux rectangles à afficher
        """

        #  On dessine des rectangles pour montrer la localisation de ces fenêtres
        cv2.rectangle(source1, image_1_top_left, image_1_bot_right, (255, 0, 0), 5)
        cv2.rectangle(source2, image_2_top_left, image_2_bot_right, (255, 0, 0), 5)

        #  On affiche ces deux fenêtres
        plt.title("Meilleure similarite entre les images")
        plt.subplot(121), plt.imshow(source1, cmap='gray')
        plt.title('Image 1'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(source2, cmap='gray')
        plt.title('Image 2'), plt.xticks([]), plt.yticks([])

        plt.show()

    @staticmethod
    def bf_matcher(source1, source2):
        """
        Question 1.3

        Trace les mises en correspondance des caractéristiques de deux images selon la méthode de "Brute Force"

        Parameters
        ----------
        source1: la source de l'image de gauche
        source2: la source de l'image de droite
        """
        # Initiate SIFT detector
        orb = cv2.ORB_create()
        cv2.ocl.setUseOpenCL(False)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(source1, None)
        kp2, des2 = orb.detectAndCompute(source2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches.
        img3 = cv2.drawMatches(source1, kp1, source2, kp2, matches[:10], None, flags=2)

        plt.imshow(img3), plt.show()

    @staticmethod
    def flann_matcher(source1, source2):
        """
        Question 1.3

        Trace les mises en correspondance des caractéristiques de deux images selon la méthode de "Brute Force"

        Problèmes sur Ubuntu et Mac, OpenCV 3.1.0. Il semblerait que ce soit un bug lié à OpenCV
        https://github.com/Itseez/opencv/issues/5667

        Parameters
        ----------
        source1: la source de l'image de gauche
        source2: la source de l'image de droite
        """

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(source1, None)
        kp2, des2 = sift.detectAndCompute(source2, None)

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

        img3 = cv2.drawMatchesKnn(source1, kp1, source2, kp2, matches, None, **draw_params)

        plt.imshow(img3, ), plt.show()


class HomemadeHarris:
    """
    Question 2
    """

    def __init__(self, source, seuil_sobel=5, smoothing_factor=(1, 1), window_size=3, nb_best_points=10000):
        self.source = source
        self.img = cv2.imread(source, 0)

        self.window_size = window_size
        self.nb_best_points = nb_best_points

        self.Ix, self.Iy = self.sobel_transformation(seuil_sobel)

        self.Ixy, self.Ix2, self.Iy2 = self.gaussian_blur(smoothing_factor)

        self.img_harris = self.harris_function(self.Ix2, self.Iy2, self.Ixy)

        print "Initialisation finie, je passe à l'amélioration de l'image"
        self.image_improvement()

        print "Amélioration de l'image finie"

    def sobel_transformation(self, seuil):
        """
        On effectue des transformation de Sobel pour garder les gradients selon
        x et y

        Parameters
        ----------
        seuil : taille du noyau de Sobel; doit être 1, 3, 5 ou 7.

        Returns
        -------
        Ix, Iy : gradients en x et y de l'image
        """
        Ix = cv2.Sobel(self.img, cv2.CV_8U, 1, 0, ksize=seuil)
        Iy = cv2.Sobel(self.img, cv2.CV_8U, 0, 1, ksize=seuil)

        return Ix, Iy

    def gaussian_blur(self, smoothing_factor):
        """
        On calcule puis on lisse Ix2 , Ix2 et Ixy avec un filtre gaussien

        Parameters
        ----------
        smoothing_factor

        Returns
        -------
        Ix2, Iy2, Ixy

        """
        Ix2 = cv2.GaussianBlur(np.multiply(self.Ix, self.Ix), smoothing_factor, 0)
        Iy2 = cv2.GaussianBlur(np.multiply(self.Iy, self.Iy), smoothing_factor, 0)
        Ixy = cv2.GaussianBlur(np.multiply(self.Ix, self.Iy), smoothing_factor, 0)

        return Ix2, Iy2, Ixy

    def image_improvement(self):
        """
        Maintenant qu'on a une image de harris, on va lui faire quelques modifications
        On met les points sur les bords et les points d'intensité négative à 0
        Et on sélectionne les maxima locaux sur une largeur de 3
        """

        # Les intensités négatives sont mises à 0
        np.clip(self.img_harris, 0, 255)

        # On met les bords à 0 pour avoir plus de fiabilité sur les maxima
        self.img_harris = self.border_to_zeros(self.img_harris)

        # On calcule les maxima locaux sur une largeur de 20
        self.img_harris = self.local_maxima(self.img_harris, self.window_size)

    def plot(self):
        """
        On obtient une nouvelle image après transformation. On va maintenant prendre tous les points
        d'intensité non négative, et les trier dans une liste, pour enfin tracer
        """

        # On prend l'image initiale avec les 1000 points les plus importants
        # NB : le tri par insertion est long pour une fenêtre de 3
        liste_points, intensites = self.insertion_sort(self.img_harris, self.nb_best_points)

        # On trace des petits cercles autour de ces points importants
        for point in liste_points:
            x = point[1]
            y = point[0]
            cv2.circle(self.img, (x, y), 1, 255, 1)

        plt.subplot(121), plt.imshow(self.img, cmap='gray')
        plt.title('Image initiale supperposee avec la transformation de Harris'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(self.img_harris, cmap='gray')
        plt.title('Image avec transformation de Harris'), plt.xticks(
            []), plt.yticks([])

        plt.show()

    @staticmethod
    def harris_function(Ix2, Iy2, Ixy):
        """
        On utilise la formule de Harris pour obtenir une nouvelle image
        """
        det = np.multiply(Ix2, Iy2) - np.multiply(Ixy, Ixy)
        trace = np.add(Iy2, Ix2)

        img_harris = det - 0.04 * np.array(np.multiply(trace, trace))
        img_harris = np.array(img_harris)

        return img_harris

    @staticmethod
    def local_maxima(img, size):
        """
        Ne garde que les maxima locaux d'une image (numpy array) pour une taille de voisinage donnée

        Parameters
        ----------
        img : image dont on veut les maxima
        size : taille du voisinage (3 dans l'énoncé)

        Returns
        -------
        img: une image (numpy array) avec les maxima et la taille de l'image de départ
        """
        mx = maximum_filter(img, size=size)
        img = np.where(mx == img, img, 0)

        return img

    @staticmethod
    def border_to_zeros(img):
        """
        On met à 0 tous les éléments du bord d'une image

        Parameters
        ----------
        img : image (numpy array) dont on veut mettre les bords à 0

        Returns
        -------
        result: l'image de départ (numpy array) avec ses bords à 0
        """
        result = np.zeros(img.shape, img.dtype)
        reduced_harris = img[1:img.shape[0] - 1, 1:img.shape[1] - 1]
        result[1:-1, 1:-1] = reduced_harris

        return result

    @staticmethod
    def insertion_sort(array, n):
        """
        Prend un tableau d'intensités et sort les coordonnées des n plus grands termes

        Parameters
        ----------
        array : un tableau d'intensités
        n : le nombre de termes que nous voulons

        Returns
        -------
        Les indices des n plus grands termes d'un tableau trié,
        Les n plus grands termes d'un tableau trié
        """
        sorted_list = []
        index_list = []
        c = -1

        for index, x in np.ndenumerate(array):

            if x > 0:  # On ne prend que les points d'intensité non nulle
                insertion_idx = np.searchsorted(sorted_list, x)
                sorted_list = np.insert(sorted_list, insertion_idx, x)
                index_list = index_list[:insertion_idx] + [index] + index_list[insertion_idx:]

            # Pour suivre la progression du tri
            if c != index[0]:
                c = index[0]
                print "Tri par insertion : ", index[0], " - ", len(array)

        return index_list[::-1][:n], sorted_list[::-1][:n]


if __name__ == "__main__":
    image_matcher = ImageMatcher(michelangelo_origin, michelangelo_tilt, corner_method="sift")
    image_matcher.calcul_similarite()

    # homemade_harris = HomemadeHarris(bike1, window_size=10)
    # homemade_harris.plot()
