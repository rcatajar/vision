# coding=utf-8

"""
Renomme les images dans truth et original
1, 2, 3, 4, etc
et affiche le dernier numéro
"""
import os

CURRENT_PATH = os.getcwd()
IMAGE_PATH = CURRENT_PATH + "/original/"
TRUTH_PATH = CURRENT_PATH + "/truth/"

i = 1
for file in os.listdir(IMAGE_PATH):
    current_name = file.split('.jpg')[0]
    for file2 in os.listdir(TRUTH_PATH):
        if file2 == (current_name + '.png'):
            os.rename(IMAGE_PATH + current_name + '.jpg', IMAGE_PATH + str(i) + '.jpg')
            os.rename(TRUTH_PATH + current_name + '.png', TRUTH_PATH + str(i) + '.png')
            i += 1
