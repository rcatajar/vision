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
            print("[test] image %s/%s" %
                  (i - limit + 1, DATASET_SIZE - limit + 1))
        yield str(i)


class Image(object):
    def __init__(self, name, _type=None):
        """
        Load ou crée l'image au nom donnée
        du type donné

        Args:
            - name (str): nom de l'image
            - _type (Optional[str]): type de l'image (original par defaut,
                                     "truth", ou nom de la méthode)
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


def histograms(color_space):
    """
    Construit les 2 histogrammes peau, non peau pour
    l'espace de couleur donnée et calcule le ratio peau / pixels dans le dataset
    """
    hist_skin = np.zeros((32, 32))
    hist_not_skin = np.zeros((32, 32))

    # Paramètres des histogrammes en fonction du l'espace de couleur
    if color_space == "RGB":
        channels = [0, 1]  # R et G
    elif color_space == "HSV":
        channels = [0, 1]  # H et S
    elif color_space == "LAB":
        channels = [1, 2]  # a et b

    # quelque soit l'espace de couleur, opencv ramene toujours les
    # valeurs dans les pixels entre 0 et 255 (images 8bits)
    ranges = [0, 256, 0, 256]
    hist_size = [32, 32]  # réduction de la quantification

    for name in train_dataset():

        # On calcule les histogrammes image par image
        # en utilisant la truth (et son inverse) comme masque
        real = Image(name).switch_color_space(color_space)
        img = real.img.astype(np.float32)
        truth = Image(name, "truth")

        # construit le masque
        mask_skin = np.zeros(real.size, dtype=np.uint8)
        for i, j, pixel in truth.pixels(with_coords=True):
            if np.array_equal(pixel, Pixel.WHITE):
                mask_skin[i][j] = 1
        mask_not_skin = (1 - mask_skin)

        hist_skin += cv2.calcHist(
            [img], channels, mask_skin, hist_size, ranges)
        hist_not_skin += cv2.calcHist(
            [img], channels, mask_not_skin, hist_size, ranges)

    if DEBUG:
        print("Histogram %s done" % color_space)
        # print_histogram(hist_skin)
        # print_histogram(hist_not_skin)

    # Calcule ratio pixel peau / non peau
    ratio_skin = np.sum(
        hist_skin) / (np.sum(hist_skin) + np.sum(hist_not_skin))
    print ratio_skin

    # Normalise les histogrammes
    hist_skin = hist_skin / np.sum(hist_skin)
    hist_not_skin = hist_not_skin / np.sum(hist_not_skin)

    return hist_skin, hist_not_skin, ratio_skin


def print_histogram(histogram):
    """
    Affiche l'histogramme passé en argument
    """
    plt.imshow(histogram, interpolation="nearest")
    plt.show()

# Build the histograms
hist_skin_RGB, hist_not_skin_RGB, ratio_skin_RGB = histograms("RGB")
hist_skin_HSV, hist_not_skin_HSV, ratio_skin_HSV = histograms("HSV")
hist_skin_LAB, hist_not_skin_LAB, ratio_skin_LAB = histograms("LAB")


class BasicHistogramDetector(AbstractSkinDetector):
    """
    Detecteur de peau basé sur les histogrammes

    Premiere approche "basique". Pour une couleur (a, b)
        - p(peau|c)  = p(c|peau)  = HistoPeau(a, b)
        - p(!peau|c) = p(c|!peau) = HistoNonPeau(a, b)
    La plus grande proba donne la classification

    Supporte les espaces de couleurs "RGB", "HSV", "LAB"
    en argument
    """

    def __init__(self, img_name, color_space):
        self.color_space = color_space
        self.set_histogram()
        self.set_name()
        super(BasicHistogramDetector, self).__init__(img_name)

    def set_name(self):
        self.METHOD_NAME = "histogram_%s" % self.color_space

    def set_histogram(self):
        """
        Choisit les histogrammes du bon espace de couleur
        """
        if self.color_space == "RGB":
            self.hist_skin = hist_skin_RGB
            self.hist_not_skin = hist_not_skin_RGB
            self.ratio_skin = ratio_skin_RGB

        elif self.color_space == "HSV":
            self.hist_skin = hist_skin_HSV
            self.hist_not_skin = hist_not_skin_HSV
            self.ratio_skin = ratio_skin_HSV

        elif self.color_space == "LAB":
            self.hist_skin = hist_skin_LAB
            self.hist_not_skin = hist_not_skin_LAB
            self.ratio_skin = ratio_skin_LAB

    def is_skin(self, a, b):
        """
        Calcule proba d'etre peau ou pas à partir des
        histogrammes et retourne True si peau et False sinon
        """
        # Reduction de la quantification
        # On ramene a et b entre 0 et 32
        a = int(a * 31. / 255.)
        b = int(b * 31. / 255.)
        p_skin = self.hist_skin[a][b]
        p_not_skin = self.hist_not_skin[a][b]
        return bool(p_skin > p_not_skin)

    def process(self):
        self.original = self.original.switch_color_space(self.color_space)
        for i, j, pixel in self.original.pixels(with_coords=True):
            # Deux premières coords du pixel utilisé en RGB et HSV
            # Deux dernières pour LAB
            if self.color_space in ["RGB", "HSV"]:
                a, b = pixel[0], pixel[1]
            else:
                a, b = pixel[1], pixel[2]
            if self.is_skin(a, b):
                self.result.set_pixel(i, j, Pixel.WHITE)
            else:
                self.result.set_pixel(i, j, Pixel.BLACK)
        self.result.save()


class BayesHistogramDetector(BasicHistogramDetector):
    """
    Utilise la formule de Bayes pour detecter les pixels de peau
    """

    def __init__(self, img_name, color_space, seuil):
        """
        Même paremetre qu'avec la formule "basique" avec le seuil en plus
        """
        self.seuil = seuil
        super(BayesHistogramDetector, self).__init__(img_name, color_space)

    def set_name(self):
        self.METHOD_NAME = "bayes_%s" % self.color_space

    def is_skin(self, a, b):
        """
        Calcule proba d'etre peau ou pas à partir des
        histogrammes et retourne True si peau et False sinon
        """
        # Reduction de la quantification
        # On ramene a et b entre 0 et 32
        a = int(a * 31. / 255.)
        b = int(b * 31. / 255.)
        p_skin = self.hist_skin[a][b]
        p_not_skin = self.hist_not_skin[a][b]
        ratio_skin = self.ratio_skin
        ratio_not_skin = 1 - self.ratio_skin

        p = (p_skin * ratio_skin) / \
            (p_skin * ratio_skin + p_not_skin * ratio_not_skin)
        return bool(p > self.seuil)


# Optimisation du seuil
def find_best_seuil(color_space):
    """
    Benchmark la détection via méthode de Bayes avec différentes valeurs de seuil et
    retourne le meilleur seuil
    """
    best_seuil = 0
    best_score = 0
    # Pour chercher le meilleur score on va chercher a maximiser le taux de bonne detection
    # et minimiser celui de mauvaise Detection sur les données d'entrainement.
    # Pour cela, on cherche à maximiser leur différence
    for seuil in np.arange(0.05, 0.3, 0.05):
        tp_avg, tp_std, fp_avg, fp_std = benchmark(
            BayesHistogramDetector, [color_space, seuil], False)
        score = tp_avg - fp_avg
        if DEBUG:
            print("Seuil: %2f, Bonne detection: %2f, Mauvaise detection: %2f" %
                  (seuil, tp_avg, fp_avg))
        if score > best_score:
            best_score = score
            best_seuil = seuil
    print("Meilleur seuil pour %s: %s" % (color_space, best_seuil))
    return best_seuil


def benchmark(Detector, extra_args, use_test_data=True):
    """
    Run un benchmark du detecteur passé en arg
    """
    print("Benchmark started for %s(%s)" % (Detector.__name__, extra_args))

    true_positive_rates = []
    false_positive_rates = []

    for image_name in (test_dataset() if use_test_data else train_dataset()):
        detector = Detector(
            image_name, *extra_args) if extra_args else Detector(image_name)
        true_positive_rate, false_positive_rate = detector.rates()
        true_positive_rates.append(true_positive_rate)
        false_positive_rates.append(false_positive_rate)

    # tp: true positive / fp: false positive
    # avg: moyenne     / std: ecart type
    tp_avg = np.mean(true_positive_rates)
    tp_std = np.std(true_positive_rates)
    fp_avg = np.mean(false_positive_rates)
    fp_std = np.mean(false_positive_rates)

    print("Benchmark finished for %s(%s)" % (Detector.__name__, extra_args))

    return tp_avg, tp_std, fp_avg, fp_std


# Benchmark
def full_benchmark(use_test_data=True):
    """
    Benchmark les différentes méthodes sur le dataset de test par default
    (si use_test_data = False, benchmark sur les données d'entrainement)
    """
    dataset_name = "test" if use_test_data else "training"

    # seuils pour la méthode de Bayes
    # obtenu via la fonction find_best_seuil
    seuil_RGB = 0.2
    seuil_LAB = 0.15
    seuil_HSV = 0.15

    METHODS = {
        "Peer et Al": (PeerSkinDetector, None),
        "Histogramme basique (RGB)": (BasicHistogramDetector, ["RGB"]),
        "Histogramme basique (LAB)": (BasicHistogramDetector, ["LAB"]),
        "Histogramme basique (HSV)": (BasicHistogramDetector, ["HSV"]),
        "Methode de Bayes (RGB, seuil = %s)" % seuil_RGB: (BayesHistogramDetector, ["RGB", seuil_RGB]),
        "Methode de Bayes (LAB, seuil = %s)" % seuil_LAB: (BayesHistogramDetector, ["LAB", seuil_LAB]),
        "Methode de Bayes (HSV, seuil = %s)" % seuil_HSV: (BayesHistogramDetector, ["HSV", seuil_HSV]),
    }
    results = {}

    # Pour chaque méthode, on teste le detecteur sur tout le dataset de test
    # On calcule les taux de bonne et mauvaise detection moyens et leurs
    # écart-types
    for name, data in METHODS.items():
        Detector, extra_args = data
        results[name] = benchmark(Detector, extra_args, use_test_data)

    # Affichage des résultats
    for name, result in results.items():
        tp_avg, tp_std, fp_avg, fp_std = result
        print("[%s] Taux bonne detection sur le dataset de %s: %2f (écart-type: %2f)" %
              (name, dataset_name, tp_avg, tp_std))
        print("[%s] Taux mauvaise detection sur le dataset de %s: %2f (écart-type: %2f)" %
              (name, dataset_name, fp_avg, fp_std))

    return results

if __name__ == "__main__":
    # Calcul des meilleurs seuils pour la methode de Bayes
    # Commenté car prends bcp de temps. Les resultats sont indiqués dans le commentaire
    #(test 5 valeurs de seuil differentes sur les ~50 photos d'entrainements)

    # find_best_seuil("RGB")  # Meilleur seuil pour RGB: 0.2
    # find_best_seuil("LAB")  # Meilleur seuil pour LAB: 0.15
    # find_best_seuil("HSV")  # Meilleur seuil pour HSV: 0.15

    # Benchmark sur les données d'entrainement
    full_benchmark(use_test_data=False)
    # Benchmark sur les données de tests
    full_benchmark(use_test_data=True)


# TODO:
# Dernier modèle (classifieur Viola Jones)
