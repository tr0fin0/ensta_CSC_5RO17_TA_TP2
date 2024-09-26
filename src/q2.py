"""Contains Question 2 algorithms."""

import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_path(file_name: str, folder: str = 'MultiView') -> str:
    """
    Return absolute file path as a string.

    Args:
        file_name (str) : file name.
        folder (str) : folder storing file.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', folder, file_name))


def get_common_prefix(name_1: str, name_2: str) -> str:
    """
    Return common prefix between 2 names as a string.

    Note: Return "" if there is no common prefix.

    Args:
        name_1 (str) : name of variable 1.
        name_2 (str) : name of variable 2.
    """
    # Remove file extensions
    name_1 = name_1.rsplit('.', 1)[0]  # Split by '.' and take the first part
    name_2 = name_2.rsplit('.', 1)[0]  # Split by '.' and take the first part

    common = ''

    for char_1, char_2 in zip(name_1, name_2):
        if char_1 == char_2:
            common += char_1
        else:
            break  # Stop if characters don't match

    print(common)
    common = common.replace('.', '-')
    common = common.replace('/', '_')

    return common


def get_points(file_name_1, file_name_2) -> list:
    """Return list of variables."""
    image_1 = cv2.cvtColor(cv2.imread(get_path(file_name_1)), cv2.COLOR_BGR2GRAY)
    image_2 = cv2.cvtColor(cv2.imread(get_path(file_name_2)), cv2.COLOR_BGR2GRAY)

    # using default values
    kaze = cv2.KAZE_create(
        upright=False, threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=2
    )

    # dÃ©tection des points KAZE et calcul des descripteurs M-SURF
    kp1, des1 = kaze.detectAndCompute(image_1, None)
    kp2, des2 = kaze.detectAndCompute(image_2, None)

    print(f'Nb de points KAZE : {str(len(kp1))} (gauche) {str(len(kp2))} (droite)')
    imgd = image_1
    imgd = cv2.drawKeypoints(image_1, kp1, imgd, -1, flags=4)
    plt.imshow(imgd)
    plt.title(f'{len(kp1)} Points KAZE (Image de gauche)')

    file_name = f'points_{get_common_prefix(file_name_1, file_name_2)}.png'
    # print(file_name)
    plt.savefig(get_path(file_name, 'output'))
    # plt.show()

    return image_1, image_2, kp1, kp2, des1, des2


def get_lines(file_name_1, file_name_2) -> list:
    """Return list."""
    # pylint: disable=too-many-locals
    image_1, image_2, kp1, kp2, des1, des2 = get_points(file_name_1, file_name_2)
    pts1 = []
    pts2 = []

    # Distance L2 pour descripteur M-SURF (KAZE)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    # Extraction de la liste des 2-plus-proches-voisins
    matches = bf.knnMatch(des1, des2, k=2)
    # Filtrage des appariements par application du ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            good.append([m])

    mfilt_image = np.array([])
    draw_params = {'matchColor': (0, 255, 0), 'singlePointColor': (255, 0, 0), 'flags': 0}
    mfilt_image = cv2.drawMatchesKnn(image_1, kp1, image_2, kp2, good, None, **draw_params)
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    print(f'Nb de paires sÃ©lectionnÃ©es : {str(pts1.shape[0])}')

    plt.figure(figsize=(15, 5))
    plt.imshow(mfilt_image)
    plt.title(f'Appariement filtrÃ© : {pts1.shape[0]} paires conservÃ©es')
    file_name = f'points0_{get_common_prefix(file_name_1, file_name_2)}.png'
    # print(file_name)
    plt.savefig(get_path(file_name, 'output'))
    # plt.show()

    return image_1, image_2, pts1, pts2


def get_matrix(file_name_1, file_name_2) -> list:
    """Return matrix."""
    image_1, image_2, pts1, pts2 = get_lines(file_name_1, file_name_2)

    fransac, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        ransacReprojThreshold=0.5,  # Distance max de reprojection en pixels pour un inlier
        confidence=0.99,
    )  # Niveau de confiance dÃ©sirÃ©
    print('Nb inliers RANSAC : ' + str(mask.sum()))

    # on affiche que les inliers
    inlierpts1 = pts1[mask.ravel() == 1]
    inlierpts2 = pts2[mask.ravel() == 1]

    # tracer les droites epipolaires
    img_left, img_right = draw_fundamental(image_1, image_2, inlierpts1, inlierpts2, fransac)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.imshow(img_left)
    plt.title(f'Lignes Ã©pipolaires des {mask.sum()} inliers (gauche)')
    plt.subplot(122)
    plt.imshow(img_right)
    plt.title(f'Lignes Ã©pipolaires des {mask.sum()} inliers (droite)')
    file_name = f'points1_{get_common_prefix(file_name_1, file_name_2)}.png'
    # print(file_name)
    plt.savefig(get_path(file_name, 'output'))
    # plt.show()


def q2() -> None:
    """Compute Q2."""
    func_name = q2.__name__
    print(f'{func_name}')

    file_names = [
        ['POP01.jpg', 'POP02.jpg'],
        ['Corridor/bt.001.pgm', 'Corridor/bt.002.pgm'],
        ['Merton1/002.jpg', 'Merton1/003.jpg'],
        ['MOVI_ToyHouse1/im002_1.png', 'MOVI_ToyHouse1/im003_1.png'],
    ]

    for file_name_1, file_name_2 in file_names:
        get_matrix(file_name_1, file_name_2)


def draw_lines(img1, img2, lines, pts1, pts2):
    """
    Img1 - image on which we draw the epilines for the points in img2.

    lines - corresponding epilines.
    """
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(
            cv2.cvtColor(
                np.asarray([[[np.random.randint(0, 180), 255, 255]]], dtype=np.uint8),
                cv2.COLOR_HSV2BGR,
            )[0, 0, :].tolist()
        )
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 2)
        # print(f'type: img1 {type(img1)}')
        # print(f'type: pt1  {type(pt1)} {pt1} {tuple(pt1.astype(int))}')
        # print(f'type: color{type(color)} {color}')
        img1 = cv2.circle(img1, tuple(tuple(pt1.astype(int))), 5, color, -1)
        img2 = cv2.circle(img2, tuple(tuple(pt2.astype(int))), 5, color, -1)
    return img1, img2


def draw_fundamental(img1, img2, pts1, pts2, f_var):
    """Draw lines."""
    # Find epilines corresponding to some points in right image (second image) and
    # drawing its lines on left image
    # indexes = np.random.randint(0, pts1.shape[0], size=(10))
    indexes = range(pts1.shape[0])
    sample_pt1 = pts1[indexes, :]
    sample_pt2 = pts2[indexes, :]

    lines1 = cv2.computeCorrespondEpilines(sample_pt2.reshape(-1, 1, 2), 2, f_var)
    lines1 = lines1.reshape(-1, 3)
    img5, _ = draw_lines(img1, img2, lines1, sample_pt1, sample_pt2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(sample_pt1.reshape(-1, 1, 2), 1, f_var)
    lines2 = lines2.reshape(-1, 3)
    img3, _ = draw_lines(img2, img1, lines2, sample_pt2, sample_pt1)
    return img5, img3
