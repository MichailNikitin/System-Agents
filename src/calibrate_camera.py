#!/usr/bin/env python3
"""Калибровка одной камеры по снимкам сетки чёрных кругов.

Снимки берутся из папки. Разрешение камеры не задаётся вручную, а
определяется по размеру самих файлов. Все снимки одной калибровки должны
быть одного размера, иначе скрипт остановится и покажет найденные размеры.

Поддерживаются две сетки кругов (cv2.findCirclesGrid):
    asymmetric -- асимметричная сетка (шахматный порядок), как в
                  стандартном шаблоне OpenCV acircles_pattern 4x11;
    symmetric  -- симметричная сетка (круги строго по строкам и столбцам).

Как считать --cols, --rows и --spacing:
    symmetric:  cols -- кругов в строке, rows -- строк;
                spacing -- расстояние между центрами соседних кругов.
    asymmetric: cols -- кругов в одной строке, rows -- всего строк
                (соседние строки сдвинуты на полшага);
                spacing -- расстояние по вертикали между соседними
                строками; центры кругов в одной строке отстоят
                на 2 * spacing.

Результат (.npz):
    camera_matrix  -- матрица камеры 3x3;
    dist_coeffs    -- коэффициенты дисторсии (k1, k2, p1, p2, k3);
    rms_error      -- RMS ошибка репроекции из cv2.calibrateCamera, пикс.;
    mean_error     -- среднее по кадрам RMS-ошибок репроекции, пикс.;
    image_size     -- размер изображения (ширина, высота);
    pattern        -- тип сетки ('asymmetric' или 'symmetric');
    grid_size      -- размер сетки (cols, rows);
    spacing        -- шаг сетки в заданных единицах;
    image_files    -- имена использованных файлов;
    rvecs, tvecs   -- положения шаблона на каждом кадре.

Пример:
    python3 calibrate_camera.py ./calib_images --cols 4 --rows 11 \\
        --spacing 20 --output camera_left.npz
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
MIN_FRAMES = 10


def parse_args():
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description='Калибровка камеры по снимкам сетки чёрных кругов.'
    )
    parser.add_argument('images', help='папка со снимками шаблона')
    parser.add_argument(
        '--pattern', choices=('asymmetric', 'symmetric'),
        default='asymmetric',
        help='тип сетки кругов (по умолчанию asymmetric)',
    )
    parser.add_argument(
        '--cols', type=int, default=4,
        help='кругов в одной строке (по умолчанию 4)',
    )
    parser.add_argument(
        '--rows', type=int, default=11,
        help='число строк кругов (по умолчанию 11)',
    )
    parser.add_argument(
        '--spacing', type=float, default=20.0,
        help='шаг сетки, мм; смысл зависит от типа сетки, см. описание '
             'в начале файла (по умолчанию 20.0)',
    )
    parser.add_argument(
        '--output', default='camera_calibration.npz',
        help='файл результата (по умолчанию camera_calibration.npz)',
    )
    parser.add_argument(
        '--debug-dir', default=None,
        help='папка, куда сохранить снимки с отмеченными кругами',
    )
    return parser.parse_args()


def make_object_points(grid_size, spacing, pattern):
    """Построить 3D-координаты центров кругов в плоскости Z=0.

    Порядок точек совпадает с порядком, который возвращает
    cv2.findCirclesGrid.
    """
    cols, rows = grid_size
    points = []
    for i in range(rows):
        for j in range(cols):
            if pattern == 'asymmetric':
                points.append(((2 * j + i % 2) * spacing, i * spacing, 0.0))
            else:
                points.append((j * spacing, i * spacing, 0.0))
    return np.array(points, dtype=np.float32)


def make_blob_detector(image_size):
    """Создать детектор тёмных пятен с учётом размера изображения.

    Стандартный детектор ищет пятна площадью не более 5000 пикс., чего
    мало для снимков высокого разрешения, поэтому верхний предел
    масштабируется по площади кадра.
    """
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = image_size[0] * image_size[1] * 0.05
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    return cv2.SimpleBlobDetector_create(params)


def find_centers(gray, grid_size, pattern):
    """Найти центры кругов; вернуть None, если сетка не найдена."""
    flags = (cv2.CALIB_CB_ASYMMETRIC_GRID if pattern == 'asymmetric'
             else cv2.CALIB_CB_SYMMETRIC_GRID)
    detector = make_blob_detector(gray.shape[::-1])
    found, centers = cv2.findCirclesGrid(
        gray, grid_size, flags=flags, blobDetector=detector
    )
    if not found:
        found, centers = cv2.findCirclesGrid(
            gray, grid_size, flags=flags | cv2.CALIB_CB_CLUSTERING,
            blobDetector=detector,
        )
    return centers if found else None


def list_images(folder):
    """Вернуть отсортированный список изображений в папке."""
    paths = sorted({
        path for ext in IMAGE_EXTENSIONS
        for pattern in (ext, ext.upper())
        for path in glob.glob(os.path.join(folder, pattern))
    })
    if not paths:
        sys.exit(f'В папке {folder} нет изображений.')
    return paths


def check_image_sizes(paths):
    """Проверить, что все снимки одного размера; вернуть (ширина, высота)."""
    sizes = {}
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'  пропуск (не читается): {path}')
            continue
        sizes.setdefault(image.shape[::-1], []).append(path)
    if not sizes:
        sys.exit('Ни один файл не удалось прочитать.')
    if len(sizes) > 1:
        print('Снимки разного размера:')
        for (width, height), files in sorted(sizes.items()):
            print(f'  {width}x{height}: {len(files)} файл(ов), '
                  f'например {os.path.basename(files[0])}')
        sys.exit('Положите снимки каждой камеры в отдельную папку.')
    return next(iter(sizes))


def load_samples(paths, grid_size, pattern, debug_dir):
    """Найти сетку на снимках; вернуть список (путь, центры кругов)."""
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    samples = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        centers = find_centers(gray, grid_size, pattern)
        if centers is None:
            print(f'  сетка не найдена: {path}')
        else:
            samples.append((path, centers))
            print(f'  ok: {path}')
        if debug_dir:
            cv2.drawChessboardCorners(
                image, grid_size,
                centers if centers is not None
                else np.empty((0, 1, 2), np.float32),
                centers is not None,
            )
            cv2.imwrite(
                os.path.join(debug_dir, os.path.basename(path)), image
            )
    return samples


def calibrate(samples, image_size, args):
    """Выполнить калибровку и вернуть словарь с результатами."""
    grid_size = (args.cols, args.rows)
    obj = make_object_points(grid_size, args.spacing, args.pattern)
    object_points = [obj] * len(samples)
    image_points = [centers for _, centers in samples]

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    errors = []
    for obj_pts, img_pts, rvec, tvec in zip(
            object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, camera_matrix, dist_coeffs
        )
        diff = img_pts.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rms_error': np.float64(rms),
        'mean_error': np.float64(np.mean(errors)),
        'image_size': np.array(image_size, dtype=np.int32),
        'pattern': np.array(args.pattern),
        'grid_size': np.array(grid_size, dtype=np.int32),
        'spacing': np.float64(args.spacing),
        'image_files': np.array(
            [os.path.basename(path) for path, _ in samples]
        ),
        'rvecs': np.array(rvecs),
        'tvecs': np.array(tvecs),
    }, errors


def print_report(result, errors, output):
    """Вывести краткий отчёт о калибровке."""
    np.set_printoptions(precision=4, suppress=True)
    print('\nОшибка репроекции по кадрам, пикс.:')
    for name, error in zip(result['image_files'], errors):
        print(f'  {name}: {error:.4f}')
    print('\nМатрица камеры:')
    print(result['camera_matrix'])
    print('Коэффициенты дисторсии:')
    print(result['dist_coeffs'].ravel())
    print(f"RMS ошибка репроекции: {result['rms_error']:.4f} пикс.")
    print(f"Средняя ошибка по кадрам: {result['mean_error']:.4f} пикс.")
    width, height = result['image_size'].tolist()
    print(f'Размер изображения: {width}x{height}')
    print(f'Сохранено в {output}')
    if result['rms_error'] > 1.0:
        print('Внимание: ошибка > 1 пикс. Проверьте параметры сетки и '
              'уберите кадры с наибольшей ошибкой.')


def main():
    """Точка входа."""
    args = parse_args()
    grid_size = (args.cols, args.rows)

    paths = list_images(args.images)
    image_size = check_image_sizes(paths)
    print(f'Найдено снимков: {len(paths)}, размер {image_size[0]}x'
          f'{image_size[1]}')

    samples = load_samples(paths, grid_size, args.pattern, args.debug_dir)
    print(f'Снимков с найденной сеткой: {len(samples)}')
    if len(samples) < MIN_FRAMES:
        sys.exit(f'Нужно минимум {MIN_FRAMES} снимков с найденной сеткой.')

    result, errors = calibrate(samples, image_size, args)
    np.savez(args.output, **result)
    print_report(result, errors, args.output)


if __name__ == '__main__':
    main()
