# coding=utf-8

'''
Introduction a la vision par ordinateur
TP 1 : Detection de visages

Code a été testé avec OpenCV 3.1.0 et python 2.7.11

Romain CATAJAR
romain.catajar@student.ecp.fr
'''

import os
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
import cv2

# où sont enregistrer les images, et où les sauvegarder
CURRENT_PATH = os.getcwd()
IMAGE_PATH = CURRENT_PATH + "/pictures/original/"
TRUTH_PATH = CURRENT_PATH + "/pictures/truth/"
RESULT_PATH = CURRENT_PATH + "/pictures/result/"


def print_image(image):
    '''
    Affiche l'image donnee en argument
    '''
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyWindow('image')


class Colors(object):
    white = [255, 255, 255]
    black = [0, 0, 0]


class SkinDetector(object):
    '''
    Applique differentes methodes de detection de la peau
    a une image et compare les resultats a ceux du dataset
    '''

    def __init__(self, image_name):
        print IMAGE_PATH + image_name
        self.original = cv2.imread(IMAGE_PATH + image_name + '.jpg', 1)
        self.truth = cv2.imread(TRUTH_PATH + image_name + '.png', 1)
        self.name = image_name

    # question 4.2.2
    def is_skin(self, pixel):
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

    # question 4.2.2
    def first_method(self):
        '''
        Applique la regle de Peer et al sur l'image.
        Le resultat est une image ou les pixels sont en blanc
        pour la peau, noir sinon.
        '''
        # On fait une copie de l'image originale pour etre sur
        # de ne pas l'alterer
        image = deepcopy(self.original)
        height, width, _ = image.shape
        # Parcoure tous les pixels
        for i in range(height):
            for j in range(width):
                if self.is_skin(image[i][j]):
                    # peau colorie en blanc
                    image[i][j] = Colors.white
                else:
                    # non peau colorie en noir
                    image[i][j] = Colors.black
        self.first_method_result = image
        # Sauvegarde le resultat dans un fichier
        cv2.imwrite(RESULT_PATH + 'first_method_' + self.name + '.jpg', image)

    # question 4.2.2
    def accuracy_first_method(self):
        """
        Calcule la precision de chaque methode en comparant
        les resultats a celui du dataset (truth)
        """
        height, width, _ = self.truth.shape
        count = 0  # nombre de pixels bien predit
        for i in range(height):
            for j in range(width):
                if np.array_equal(self.truth[i][j], self.first_method_result[i][j]):
                    count += 1
        return count / float(height * width)


if __name__ == "__main__":
    c = SkinDetector('06Apr03Face')
    c.first_method()
    print c.accuracy_first_method()
