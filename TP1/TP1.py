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

# où sont enregistrer les images, et où les sauvegarder
CURRENT_PATH = os.getcwd()
IMAGE_PATH = CURRENT_PATH + "/pictures/original/"
TRUTH_PATH = CURRENT_PATH + "/pictures/truth/"
RESULT_PATH = CURRENT_PATH + "/pictures/result_%s/"


DATASET_SIZE = 78


class Image(object):
    def __init__(self, name, _type=None):
        """
        Load ou crée l'image au nom donnée
        du type donné

        Args:
            - name (str): nom de l'image
            - _type (Optional[str]): type de l'image
                                    (None pour original, "truth" ou nom de la méthode)
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

        Par default, yield les pixels [B, G, R]
        un par un

        Si with_coords est True, yield des triplets
        (i, j, pixel) avec:
            - i, j coordonnées du pixel
            - pixel: liste [B, G, R]
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
        B, G, R = pixel  # pixel donnee en BGR par opencv
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

    def accuracy(self):
        """
        Calcule la précision de la detection en comparant aux
        resultats fournis avec le dataset
        """
        count = 0  # compteur de pixels bien prédit
        for (pixel_result, pixel_truth) in zip(self.result.pixels(), self.truth.pixels()):
            if np.array_equal(pixel_result, pixel_truth):
                count += 1

        # Calcul de l'accuracy:  nb_bonne_prédiction / (largeur * hauteur)
        height, width = self.original.size
        return count / float(height * width)


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
    Run la méthode de Peer et Al sur tout le dataset.
    Calcule la précision moyenne et son écart-type
    """
    accuracies = []
    for i in range(1, DATASET_SIZE + 1):
        print("Processing image %s/%s" % (i, DATASET_SIZE))
        detector = PeerSkinDetector(str(i))
        accuracies.append(detector.accuracy())

    avg = np.mean(accuracies)
    std = np.std(accuracies)
    print("Précision moyenne: %s \nEcart type: %s" % (avg, std))
    return avg, std

if __name__ == "__main__":
    avg, std = benchmark_peer()
    # Précision moyenne: 0.782445833341
    # Ecart type: 0.168097004244
