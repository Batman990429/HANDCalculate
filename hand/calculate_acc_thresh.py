# -*- coding: utf-8 -*-
# @Time : 2021/8/3 16:46
# @Author : Tty
# @Email : 1843222968@qq.com
# @File : calculate_acc_thresh.py


import gdal
import ogr
import numpy as np
import os
import sys


def readImage(image: str, bandNum: int):
    """
    读取图像的一个波段，同时返回及其行列数和波段数
    :param image:
    :param bandNum:
    :return:
    """
    data_set = gdal.Open(image)

    im_width = data_set.RasterXSize
    im_height = data_set.RasterYSize
    im_bands = data_set.RasterCount
    print('im_width:', im_width, 'im_height:', im_height, 'bands_num:', im_bands)

    im = data_set.GetRasterBand(bandNum).ReadAsArray()

    del data_set

    return im, im_width, im_height, im_bands


def clip_shapefile(input_filename, clip_filename, output_filename):
    """
    对shapefile进行裁剪
    :param input_filename: 输入shapefile文件名
    :param clip_filename: 裁剪shapefile文件名
    :param output_filename:输出shapefile文件名
    :return:
    """

    driver = ogr.GetDriverByName('ESRI Shapefile')
    input_datasource = driver.Open(input_filename, 0)
    clip_datasource = driver.Open(clip_filename, 0)
    output_datasource = driver.CreateDataSource(output_filename)

    input_layer = input_datasource.GetLayer(0)
    clip_layer = clip_datasource.GetLayer(0)
    print("Input shapefile's GeomType is %s" % (input_layer.GetGeomType()))
    # TODO 获取shapefile文件的坐标系
    output_layer = output_datasource.CreateLayer('clipped', None, input_layer.GetGeomType())

    defn = input_layer.GetLayerDefn()
    for i in range(defn.GetFieldCount()):
        # 复制字段
        output_layer.CreateField(defn.GetFieldDefn(i))
    output_layer = output_datasource.GetLayer(0)

    input_layer.Clip(clip_layer, output_layer)

    output_datasource.Release()


def shp_to_raster(shapefile, raster, save_file):
    """
    shapefile转栅格
    :param shapefile:
    :param raster:
    :param save_file:
    :return:
    """

    dataset = gdal.Open(raster, gdal.GA_ReadOnly)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    shape = ogr.Open(shapefile)
    layer = shape.GetLayer()

    target_dataset = gdal.GetDriverByName('GTiff').Create(save_file, im_width, im_height, 1, gdal.GDT_Byte)
    target_dataset.SetGeoTransform(dataset.GetGeoTransform())
    target_dataset.SetProjection(dataset.GetProjection())
    band = target_dataset.GetRasterBand(1)
    nodata_value = -9999
    band.SetNoDataValue(nodata_value)
    band.FlushCache()
    gdal.RasterizeLayer(target_dataset, [1], layer)

    print('%s to %s success!' % (shapefile, save_file))


def get_unique_point(line_layer, start_of_drainages='p_o_start'):
    """
    找到Channel initiation的点
    参考房哲师兄的代码：
    https://github.com/fangzhegeo/DIMERS/blob/38c5bbcd2576c1476fd40dcea894881f8b62e737/analysis/HAND.py
    :param line_layer:
    :param start_of_drainages:
    :return:
    """
    startlist = []
    lastlist = []

    line_driver = ogr.GetDriverByName('ESRI Shapefile')
    line_data_source = line_driver.Open(line_layer, 0)
    layer = line_data_source.GetLayer()

    # 创建点要素存储CI点
    point_driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(start_of_drainages):
        # 创建新数据时不能覆盖现有的数据源
        print('{0} already exists, try to delete it !'.format(start_of_drainages))
        point_driver.DeleteDataSource(start_of_drainages)
        print('{0} delete successfully !'.format(start_of_drainages))
    point_data_source = point_driver.CreateDataSource(start_of_drainages)
    point_layer = point_data_source.CreateLayer(start_of_drainages, srs=layer.GetSpatialRef(),
                                                geom_type=ogr.wkbPoint)

    # 在MSN上寻找CI点
    for feature in layer:
        geom = feature.GetGeometryRef()
        startlist.append([round(geom.GetX(0), 4), round(geom.GetY(0), 4)])
        point_count = geom.GetPointCount()
        lastlist.append([round(geom.GetX(point_count - 1), 4), round(geom.GetY(point_count - 1), 4)])

    unique_list = []
    for x in startlist:
        if x not in lastlist:
            unique_list.append(x)

    # 存储CI点
    for tmp in unique_list:
        wkt = 'POINT (' + str(tmp[0]) + ' ' + str(tmp[1]) + ')'
        geom = ogr.CreateGeometryFromWkt(wkt)
        feat = ogr.Feature(point_layer.GetLayerDefn())
        feat.SetGeometry(geom)
        point_layer.CreateFeature(feat)

    line_data_source.Destroy()
    point_data_source.Destroy()


def get_pixels(shp, im):
    """
    根据矢量点获取点对应的像素
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(shp, 0)
    if ds is None:
        print('Could not open' + shp)
        sys.exit(1)

    layer = ds.GetLayer()

    x_values = []
    y_values = []
    feature = layer.GetNextFeature()
    count = 0

    while feature:
        count = count + 1
        geometry = feature.GetGeometryRef()
        x = geometry.GetX()
        y = geometry.GetY()
        x_values.append(x)
        y_values.append(y)
        feature = layer.GetNextFeature()

    gdal.AllRegister()

    ds = gdal.Open(im, gdal.GA_ReadOnly)
    if ds is None:
        print('Could not open image')
        sys.exit(1)

    rows = ds.RasterYSize
    cols = ds.RasterXSize
    bands = ds.RasterCount

    transform = ds.GetGeoTransform()
    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]

    values = []
    for i in range(len(x_values)):
        x = x_values[i]
        y = y_values[i]

        x_offset = min(int((x - x_origin) / pixel_width), cols - 1)
        y_offset = min(int((y - y_origin) / pixel_height), rows - 1)

        bs = ds.ReadAsArray(x_offset, y_offset, 1, 1)
        values.append(bs[0, 0])

    return np.array(values)


def count_cells(your_raster):
    # nodata value = -9999
    mynumpy_no_data = np.ma.masked_array(your_raster, your_raster == 0)
    # how_many = len(np.where(your_raster > -9999))
    how_many = len(np.where(your_raster > 0)[0])

    return how_many, np.min(mynumpy_no_data), np.max(mynumpy_no_data), np.mean(mynumpy_no_data)


def sorter(myset):
    myset2 = {}
    for x in myset:
        myset2[x] = x
    import operator
    sorted_x = sorted(myset2.items(), key=operator.itemgetter(1))
    sorted2 = []
    for cc in sorted_x:
        sorted2.append(cc[1])
    return sorted2


def find_exp(flow_acc, exper, cellcount_of_river, acc_list):
    global result_expr
    if len(acc_list) > 1:
        cellcount_of_out = len(np.where(flow_acc > exper)[0])
        if cellcount_of_river < cellcount_of_out:
            # 当msn的河道像素个数小于该阈值提取河道像素个数时
            length = int(len(acc_list) / 2)
            median_1 = acc_list[length]

            acc_list = acc_list[length:]
            find_exp(flow_acc, median_1, cellcount_of_river, acc_list)
        else:
            # 当msn的河道像素个数大于该阈值提取河道像素个数时
            length = int(len(acc_list) / 2)
            median_1 = acc_list[length]

            acc_list = acc_list[:length]
            find_exp(flow_acc, median_1, cellcount_of_river, acc_list)
    else:
        result_expr = int(exper)
        return 1


def get_threshold_automatically(msn, acc):
    """
    由map stream network（MSN）自动计算得出划分河道的最优值
    :param msn: 精确度较高的河道网络
    :param acc: 流量图
    :return:
    """
    start_of_drainages = 'p_o_start'
    get_unique_point(msn, start_of_drainages)

    save_file = 'msn_raster.tif'

    shp_to_raster(msn, acc, save_file)

    msn_array = readImage(save_file, 1)[0]
    acc_array = readImage(acc, 1)[0]

    acc_of_river = get_pixels(start_of_drainages, acc)
    # 记录MSN河道像素的个数
    cellcount_of_river = count_cells(msn_array)[0]

    myset = list(set(acc_of_river))
    # 对MSN河道像素对应的流向值进行排序
    sorted2 = sorter(myset)

    find_exp(acc_array, sorted2[0], cellcount_of_river, sorted2)

    return result_expr


if __name__ == '__main__':
    os.chdir(r'E:\experiment\HAND')
    print(get_threshold_automatically(r'China_river\zhengzhou_river.shp',
                                      r'E:\experiment\HAND\HANDModel\output_ACC.tif'))
