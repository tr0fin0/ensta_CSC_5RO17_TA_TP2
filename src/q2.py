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


def get_common_name(names: list[str]) -> str:
    """
    Return common prefix between 2 names as a string.

    Note: Return "" if there is no common prefix.

    Args:
        names (list[str]) : variables names
    """
    # remove file extensions
    name_0 = names[0].rsplit('.', 1)[0]  # Split by '.' and take the first part
    name_1 = names[1].rsplit('.', 1)[0]  # Split by '.' and take the first part

    common_name = ''

    for char_0, char_1 in zip(name_0, name_1):
        if char_0 == char_1:
            common_name += char_0
        else:
            break  # stop if characters don't match

    # remove undesired chars
    common_name = common_name.replace('.', '-')
    common_name = common_name.replace('/', '_')

    return common_name


def plot_kaze_points(files_names: list[str]) -> list[np.array]:
    """
    Plot KAZE points of an image pair.

    Note: function return KAZE points data.

    Args:
        files_names (list[str]) : array of files names.
    """
    # read images
    image_0 = cv2.cvtColor(cv2.imread(get_path(files_names[0])), cv2.COLOR_BGR2GRAY)
    image_1 = cv2.cvtColor(cv2.imread(get_path(files_names[1])), cv2.COLOR_BGR2GRAY)

    # create KAZE object with default values
    kaze = cv2.KAZE_create(
        upright=False, threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=2
    )

    # compute KAZE points
    kaze_points_0, des1 = kaze.detectAndCompute(image_0, None)
    kaze_points_1, des2 = kaze.detectAndCompute(image_1, None)

    # create plot
    plot_name = f'{plot_kaze_points.__name__}_{get_common_name(files_names)}'
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), num=plot_name)
    fig.suptitle(plot_name)
    fig.set_dpi(300)

    axs[0].imshow(cv2.drawKeypoints(image_0, kaze_points_0, image_0, -1, flags=4))
    axs[0].set_title(f'[{files_names[0]}] {len(kaze_points_0)} KAZE points')

    axs[1].imshow(cv2.drawKeypoints(image_1, kaze_points_1, image_1, -1, flags=4))
    axs[1].set_title(f'[{files_names[1]}] {len(kaze_points_1)} KAZE points')

    # save plot
    plt.tight_layout()
    plt.savefig(get_path(f'{plot_name}.png', 'output'))

    return image_0, image_1, kaze_points_0, kaze_points_1, des1, des2


def plot_epipolar_lines(files_names: list[str]) -> list:
    """
    Plot epipolar lines of an image pair.

    Note: function return epipolar lines data.

    Args:
        files_names (list[str]) : array of files names.
    """
    # pylint: disable=too-many-locals
    image_0, image_1, kaze_points_0, kaze_points_1, des1, des2 = plot_kaze_points(files_names)

    # Distance L2 pour descripteur M-SURF (KAZE)
    # Extraction de la liste des 2-plus-proches-voisins
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Filtrage des appariements par application du ratio test
    valid_points = []
    valid_points_image_0 = []
    valid_points_image_1 = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            valid_points.append([m])
            valid_points_image_0.append(kaze_points_0[m.queryIdx].pt)
            valid_points_image_1.append(kaze_points_1[m.trainIdx].pt)

    draw_params = {'matchColor': (0, 255, 0), 'singlePointColor': (255, 0, 0), 'flags': 0}
    epipolar_image = cv2.drawMatchesKnn(
        image_0, kaze_points_0, image_1, kaze_points_1, valid_points, None, **draw_params
    )
    valid_points_image_0 = np.float32(valid_points_image_0)
    valid_points_image_1 = np.float32(valid_points_image_1)

    plt.figure(figsize=(15, 5))
    plt.imshow(epipolar_image)
    plt.title(f'Appariement filtrÃ© : {valid_points_image_0.shape[0]} paires conservÃ©es')
    file_name = f'points0_{get_common_name(files_names)}.png'
    # print(file_name)
    plt.savefig(get_path(file_name, 'output'))
    # plt.show()

    return image_0, image_1, valid_points_image_0, valid_points_image_1


def get_matrix(files_names, show_plot: bool = False) -> list:
    """Return matrix."""
    image_0, image_1, valid_points_image_0, valid_points_image_1 = plot_epipolar_lines(files_names)

    fransac, mask = cv2.findFundamentalMat(
        valid_points_image_0,
        valid_points_image_1,
        cv2.FM_RANSAC,
        ransacReprojThreshold=0.5,  # Distance max de reprojection en pixels pour un inlier
        confidence=0.99,
    )  # Niveau de confiance dÃ©sirÃ©
    # print('Nb inliers RANSAC : ' + str(mask.sum()))

    # on affiche que les inliers
    inlierpts1 = valid_points_image_0[mask.ravel() == 1]
    inlierpts2 = valid_points_image_1[mask.ravel() == 1]

    # tracer les droites epipolaires
    img_left, img_right = draw_fundamental(image_0, image_1, inlierpts1, inlierpts2, fransac)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.imshow(img_left)
    plt.title(f'Lignes Ã©pipolaires des {mask.sum()} inliers (gauche)')
    plt.subplot(122)
    plt.imshow(img_right)
    plt.title(f'Lignes Ã©pipolaires des {mask.sum()} inliers (droite)')

    file_name = f'points1_{get_common_name(files_names)}.png'
    plt.savefig(get_path(file_name, 'output'))
    if show_plot:
        plt.show()


def draw_lines(img1, img2, lines, valid_points_image_0, valid_points_image_1):
    """
    Img1 - image on which we draw the epilines for the points in img2.

    lines - corresponding epilines.
    """
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, valid_points_image_0, valid_points_image_1):
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


def draw_fundamental(img1, img2, valid_points_image_0, valid_points_image_1, f_var):
    """Draw lines."""
    # Find epilines corresponding to some points in right image (second image) and
    # drawing its lines on left image
    # indexes = np.random.randint(0, valid_points_image_0.shape[0], size=(10))
    indexes = range(valid_points_image_0.shape[0])
    sample_pt1 = valid_points_image_0[indexes, :]
    sample_pt2 = valid_points_image_1[indexes, :]

    lines1 = cv2.computeCorrespondEpilines(sample_pt2.reshape(-1, 1, 2), 2, f_var)
    lines1 = lines1.reshape(-1, 3)
    img5, _ = draw_lines(img1, img2, lines1, sample_pt1, sample_pt2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(sample_pt1.reshape(-1, 1, 2), 1, f_var)
    lines2 = lines2.reshape(-1, 3)
    img3, _ = draw_lines(img2, img1, lines2, sample_pt2, sample_pt1)
    return img5, img3


def q2() -> None:
    """Compute Q2."""
    func_name = q2.__name__
    print(f'{func_name}: initialized')

    file_names_arr = [
        ['POP01.jpg', 'POP02.jpg'],
        ['Corridor/bt.001.pgm', 'Corridor/bt.002.pgm'],
        ['Merton1/002.jpg', 'Merton1/003.jpg'],
        ['MOVI_ToyHouse1/im002_1.png', 'MOVI_ToyHouse1/im003_1.png'],
    ]

    for files_names in file_names_arr:
        get_matrix(files_names)
