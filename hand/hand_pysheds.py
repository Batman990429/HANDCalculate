# -*- coding: utf-8 -*-
# @Time : 2021/8/1 17:54
# @Author : Tty
# @Email : 1843222968@qq.com
# @File : hand_pysheds.py

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from calculate_acc_thresh import get_threshold_automatically
from array_to_raster import array2Raster


def fill_nan(array: np.ndarray) -> np.ndarray:
    """Replace NaNs with values interpolated from their neighbors
    Replace NaNs with values interpolated from their neighbors using a 2D Gaussian
    kernel, see: https://docs.astropy.org/en/stable/convolution/#using-astropy-s-convolution-to-replace-bad-data
    """
    try:
        import astropy.convolution
    except ImportError:
        raise ImportError('fill_nan calculation requires astropy to be installed')
    kernel = astropy.convolution.Gaussian2DKernel(x_stddev=3)  # kernel x_size=8*stddev
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        array = astropy.convolution.interpolate_replace_nans(
            array, kernel, convolve=astropy.convolution.convolve
        )

    return array


# N    NE    E    SE    S    SW    W    NW
ArcGIS = (64, 128, 1, 2, 4, 8, 16, 32)


def calculate_hand(dem_file, msn=None, acc_thresh=None, dirmap=ArcGIS):
    """
    Calculate Height Above Nearest Drainage
    参考：
    https://github.com/mdbartos/pysheds/blob/master/examples/hand.ipynb
    https://github.com/ASFHyP3/asf-tools/blob/9f83b9f6cdbe20f5785b0130b051d3c01c2661aa/asf_tools/hand/calculate.py#L77
    :param dem_file:dem文件路径
    :param msn:map stream network，精确的河道数据
    :param acc_thresh:界定河道的阈值
    :param dirmap:流向图的流向表示，默认为ArcGIS流向图的表示方法
    :return:
    """
    try:
        from pysheds.grid import Grid
    except ImportError:
        raise ImportError('HAND calculation requires Pysheds to be installed')

    # 读取DEM
    grid = Grid.from_raster(dem_file, data_name='dem')
    dem_plot(grid.dem, grid.extent)

    # fill填洼
    print('filling depressions')
    grid.fill_depressions('dem', out_name='filled_dem')
    if np.isnan(grid.filled_dem).any():
        print('NaNs encountered in filled DEM; filling.')
        grid.filled_dem = fill_nan(grid.filled_dem)

    # Resolving flats
    print('Resolving flats')
    grid.resolve_flats('filled_dem', out_name='inflated_dem')
    if np.isnan(grid.inflated_dem).any():
        print('NaNs encountered in inflated DEM; replacing NaNs with original DEM values')
        grid.inflated_dem[np.isnan(grid.inflated_dem)] = grid.dem[np.isnan(grid.inflated_dem)]

    array2Raster(grid.inflated_dem, 'ps_corrected_dem.tif', refImg=dem_file)

    # 计算流向图
    # D8 flow directions
    print('Obtaining flow direction')
    grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)
    if np.isnan(grid.dir).any():
        print('NaNs encountered in flow direction; filling.')
        grid.dir = fill_nan(grid.dir)
    dir_plot(grid.dir, grid.extent, dirmap)

    array2Raster(grid.dir, 'dir.tif', refImg=dem_file)

    # 计算流量图
    print('Calculating flow accumulation')
    grid.accumulation(data='dir', out_name='acc')
    if np.isnan(grid.acc).any():
        print('NaNs encountered in accumulation; filling.')
        grid.acc = fill_nan(grid.acc)
    acc_plot(grid.acc, grid.extent)

    array2Raster(grid.acc, 'acc.tif', refImg=dem_file)

    # 若未设置界定河道的阈值，则依据MSN自动计算阈值
    if acc_thresh is None:
        acc_thresh = get_threshold_automatically(msn, 'acc.tif')

    print(f'Calculating HAND using accumulation threshold of {acc_thresh}')
    hand = grid.compute_hand('dir', 'inflated_dem', grid.acc > acc_thresh, inplace=False)
    if np.isnan(hand).any():
        print('NaNs encountered in HAND; filling.')
        hand = fill_nan(hand)

    hand_plot(hand, extent=grid.extent)

    array2Raster(hand, 'hand_' + str(acc_thresh) + '.tif', refImg=dem_file)


def dem_plot(dem, extent):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0)

    plt.imshow(dem, extent=extent, cmap='terrain', zorder=1)
    plt.colorbar(label='Elevation (m)')
    plt.grid(zorder=0)
    plt.title('Digital elevation map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()


def dir_plot(dir, extent, dirmap):
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_alpha(0)

    plt.imshow(dir, extent=extent, cmap='viridis', zorder=2)
    boundaries = ([0] + sorted(list(dirmap)))
    plt.colorbar(boundaries=boundaries,
                 values=sorted(dirmap))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flow direction grid')
    plt.grid(zorder=-1)
    plt.tight_layout()
    plt.show()


def acc_plot(acc, extent):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0)
    plt.grid('on', zorder=0)
    im = ax.imshow(acc, extent=extent, zorder=2,
                   cmap='cubehelix',
                   norm=colors.LogNorm(1, acc.max()))
    plt.colorbar(im, ax=ax, label='Upstream Cells')
    plt.title('Flow Accumulation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def hand_plot(hand, extent):
    plt.subplots(figsize=(8, 6))
    # minvalue must be positive
    plt.imshow(hand + 1, zorder=1, cmap='terrain', interpolation='bilinear', extent=extent,
               norm=colors.LogNorm(vmin=1, vmax=np.nanmax(hand)))
    plt.colorbar(label='Height above nearest drainage (m)')
    plt.title('Height above nearest drainage', size=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import os
    os.chdir('../data')
    hand = calculate_hand('zhengzhou_dem.tif', msn='zhengzhou_river.shp')
