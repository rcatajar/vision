# coding=utf-8

'''
Introduction a la vision par ordinateur
TP 1 : Detection de visages

Code testé avec OpenCV 3.1.0 et python 2.7.11

Romain CATAJAR
romain.catajar@student.ecp.fr
'''

import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

# active des print de debug
DEBUG = True

# où sont enregistrer les images, et où les sauvegarder
CURRENT_PATH = os.getcwd()
IMAGE_PATH = CURRENT_PATH + "/pictures/original/"
TRUTH_PATH = CURRENT_PATH + "/pictures/truth/"
RESULT_PATH = CURRENT_PATH + "/pictures/result_%s/"


DATASET_SIZE = 78
# On sépare le dataset en deux partie
# 2/3 pour l'entrainement
# 1/3 pour tester


def train_dataset():
    """
    Liste le nom des images pour l'entrainement
    """
    # 2/3 du dataset
    limit = (2 * DATASET_SIZE) / 3
    for i in range(1, limit):
        if DEBUG:
            print("[train] image %s/%s" % (i, limit - 1))
        # Les fichiers des images sont nommés 1(.jpg), 2, 3, ...
        yield str(i)


def test_dataset():
    """
    Liste le nom des images pour les tests
    """
    limit = (2 * DATASET_SIZE) / 3
    for i in range(limit, DATASET_SIZE + 1):
        if DEBUG:
            print("[test] image %s/%s" % (i - limit + 1, DATASET_SIZE - limit + 1))
        yield str(i)


class Image(object):
    def __init__(self, name, _type=None):
        """
        Load ou crée l'image au nom donnée
        du type donné

        Args:
            - name (str): nom de l'image
            - _type (Optional[str]): type de l'image (original par defaut,
                                     "truth" ou nom de la méthode)
        """
        self.name = name

        if _type is None:
            self.set_original()
        elif _type == "truth":
            self.set_truth()
        else:
            self.set_result(_type)

        self.load()

    def set_original(self):
        self.path = IMAGE_PATH + self.name + '.jpg'

    def set_truth(self):
        self.path = TRUTH_PATH + self.name + '.png'

    def set_result(self, _type):
        # pour un resultat, on commence par faire une copie
        # de l'image originale
        self.set_original()
        self.load()
        self.path = RESULT_PATH % _type + self.name + '.jpg'
        self.save()

    def load(self):
        self.img = cv2.imread(self.path, 1)

    def save(self):
        cv2.imwrite(self.path, self.img)

    def show(self):
        """
        Affiche l'image
        """
        cv2.imshow(self.name, self.img)
        cv2.waitKey(0)
        cv2.destroyWindow(self.name)

    def switch_color_space(self, space):
        # La converstion ne marche que si l'image est
        # actuellement en BGR (ce qui est le cas par default)
        if space == "HSV":
            color = cv2.COLOR_BGR2HSV
        elif space == "RGB":
            color = cv2.COLOR_BGR2RGB
        elif space == "LAB":
            color = cv2.COLOR_BGR2LAB
        self.img = cv2.cvtColor(self.img, color)
        return self

    @property
    def size(self):
        """
        Taille de l'image (hauteur, largeur)
        """
        height, width, _ = self.img.shape
        return height, width

    def pixels(self, with_coords=False):
        """
        Itère sur tous les pixels de l'image

        Par default, yield les pixels un par un

        Si with_coords est True, yield des triplets
        (i, j, pixel) avec i, j coordonnées du pixel
        """
        height, width = self.size
        for i in range(height):
            for j in range(width):
                if with_coords:
                    yield i, j, self.img[i][j]
                else:
                    yield self.img[i][j]

    def set_pixel(self, i, j, pixel):
        """
        Modifie le pixel (i, j) de l'image
        par le pixel passé en argument
        """
        self.img[i][j] = pixel


class Pixel(object):
    WHITE = [255, 255, 255]
    BLACK = [0, 0, 0]

    @staticmethod
    def is_skin(pixel):
        '''
        Renvoi True si le pixel est de la peau, False sinon

        Utilise la regle de Peer et al
        un pixel RGB est de la peau si:
            - R > 96
            - G > 40
            - B > 20
            - max([R, G, B]) - min([R, G, B]) > 15
            - abs(R - G) > 15
            - R > G
            - R > B
        '''
        B, G, R = pixel  # pixel donnee en BGR par default
        rules = [
            R > 96,
            G > 40,
            B > 20,
            max(pixel) - min(pixel) > 15,
            # par default les pixels sont des unsigned ints
            # on a besoin de les caster en int au cas ou R - G < 0
            abs(int(R) - int(G)) > 15,
            R > G,
            R > B
        ]
        # Renvoi True si toutes les regles sont verifies sinon False
        return True if all(rules) else False


class AbstractSkinDetector(object):
    METHOD_NAME = ""

    def __init__(self, img_name):
        self.original = Image(img_name)
        self.truth = Image(img_name, "truth")
        self.result = Image(img_name, self.METHOD_NAME)
        self.process()

    def process(self):
        """
        Applique l'algo de detection de la peau et colorie
        l'image en blanc pour la peau et noir pour le reste
        """
        raise NotImplementedError()

    def rates(self):
        """
        Calcule les taux de bonne et mauvaise detection
        """
        true_positives = 0.  # nb de vrai positif
        skin = 0.  # nb de pixels de peau dans l'original
        false_positives = 0.  # nb de faux positif
        not_skin = 0.  # nb de pixels de non peau dans l'original

        for (pixel_result, pixel_truth) in zip(self.result.pixels(), self.truth.pixels()):
            if np.array_equal(pixel_truth, Pixel.WHITE):
                # pixel de peau
                skin += 1
                if np.array_equal(pixel_result, Pixel.WHITE):
                    # vrai positif
                    true_positives += 1
            else:
                # pixel de non peau
                not_skin += 1
                if np.array_equal(pixel_result, Pixel.WHITE):
                    # faux positif
                    false_positives += 1

        true_positive_rate = true_positives / skin
        false_positive_rate = false_positives / not_skin
        return true_positive_rate, false_positive_rate


class PeerSkinDetector(AbstractSkinDetector):
    """
    Utilise la méthode de Peer et Al pour detecter la peau
    """
    METHOD_NAME = "peer_et_al"

    def process(self):
        for i, j, pixel in self.result.pixels(with_coords=True):
            if Pixel.is_skin(pixel):
                self.result.set_pixel(i, j, Pixel.WHITE)
            else:
                self.result.set_pixel(i, j, Pixel.BLACK)
        self.result.save()


def benchmark_peer():
    """
    Run la méthode de Peer et Al sur tout le dataset de test.
    Calcule les taux de bonne et mauvaise detection moyen et
    leur ecart type
    """
    true_positive_rates = []
    false_positive_rates = []

    for image_name in test_dataset():
        detector = PeerSkinDetector(image_name)
        true_positive_rate, false_positive_rate = detector.rates()
        true_positive_rates.append(true_positive_rate)
        false_positive_rates.append(false_positive_rate)

    tp_avg = np.mean(true_positive_rates)
    tp_std = np.std(true_positive_rates)
    print("Taux bonne detection: %2f (écart-type: %2f)" % (tp_avg, tp_std))

    fp_avg = np.mean(false_positive_rates)
    fp_std = np.mean(false_positive_rates)
    print("Taux mauvaise detection: %2f (écart-type: %2f)" % (fp_avg, fp_std))


def histograms(color_space):
    """
    Construit les 2 histogrammes peau, non peau pour
    l'espace de couleur donnée
    """
    # Construit une liste de tout les pixels de peau
    # et de non peau à partir des résultats du datases
    pixels_skin = []
    pixels_not_skin = []

    for name in train_dataset():
        img_real = Image(name).switch_color_space(color_space)
        img_truth = Image(name, "truth")
        for pixel_real, pixel_truth in zip(img_real.pixels(), img_truth.pixels()):
            if np.array_equal(pixel_truth, Pixel.WHITE):
                pixels_skin.append(pixel_real)
            else:
                pixels_not_skin.append(pixel_real)

    # Use the correct np array type
    pixels_skin = np.array([pixels_skin], dtype=np.float32)
    pixels_not_skin = np.array([pixels_not_skin], dtype=np.float32)

    # Paramètres des histogrammes en fonction du l'espace de couleur
    if color_space == "RGB":
        channels = [0, 1]  # R et G
        ranges = [0, 256, 0, 256]  # R et G de 0 à 256

    elif color_space == "HSV":
        channels = [0, 1]  # H et S
        ranges = [0, 360, 0, 1]  # H de 0 à 360, S de 0 à 1

    elif color_space == "LAB":
        channels = [1, 2]  # a et b
        ranges = [-128, 127, -128, 127]  # a et b de -128 à 127

    hist_size = [32, 32]  # réduction de la quantification
    hist_skin = cv2.calcHist([pixels_skin], channels, None, hist_size, ranges)
    hist_not_skin = cv2.calcHist([pixels_not_skin], channels, None, hist_size, ranges)

    if DEBUG:
        print_histogram(hist_skin)
        print_histogram(hist_not_skin)
    return hist_skin, hist_not_skin


def print_histogram(histogram):
    """
    Affiche l'histogramme passé en argument
    """
    plt.imshow(histogram, interpolation="nearest")
    plt.show()

# Build the histograms
hist_skin_RGB, hist_not_skin_RGB = histograms("RGB")
hist_skin_HSV, hist_not_skin_HSV = histograms("HSV")
hist_skin_LAB, hist_not_skin_LAB = histograms("LAB")


# Detecter avec ces histogrammes
# 2 methodes:
#   - probas basique
#   - methode de Bayes (tester differentes valeurs de seuil)

# Detecter avec Viola Jones


if __name__ == "__main__":
    pass
    # benchmark_peer()
    # Taux bonne detection: 0.911231 (écart-type: 0.089854)
    # Taux mauvaise detection: 0.158280 (écart-type: 0.158280)
    # Exemple d'image ou ca marche mal: 6, 16, 29
    #                             bien: 59, 76, 78
