import numpy as np


def window_around(img, corners, n, p):
    """
        Cette fonction prend en argument    img = l'image initiale
                                            corners = une liste de coordonnees de points autour desquelles on veut une fenetre
                                            n = la largeur de la fenetre
                                            p = la hauteur de la fenetre
    :return: list_of_windows:
    """

    list_of_windows = []

    size_y, size_x = img.shape

    for corner in corners:
        y = corner[0]  # Inversion des coordonnees dans corner
        x = corner[1]  # x,i,n  y,j,p

        # On verifie qu'on ne sort pas de l'image
        if x > n and y > p and x + n < size_x and y + p < size_y:
            window = []
            # TODO: problemes d'indicest
            for i in range(-n, n + 1):
                window.append([img[x + i, y + j] for j in range(-p, p + 1)])

            # list_of_windows.append([window, (y-p, x-n), (y+p+1, x+n+1)])
            list_of_windows.append(
                [window, (x - n, y - p), (x + n, y + p)])

    return np.array(list_of_windows)

img = np.arange(100).reshape(10,10)
corners = [[2, 4]]
print img
print window_around(img, corners, 1, 1)
