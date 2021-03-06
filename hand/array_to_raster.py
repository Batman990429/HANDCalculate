# -*- coding: utf-8 -*-
# @Time : 2021/8/1 17:43
# @Author : Tty
# @Email : 1843222968@qq.com
# @File : array_to_raster.py
# -*- coding: utf-8 -*-
# @Time : 2021/4/5 10:37
# @Author : Tty
# @Email : 1843222968@qq.com
# @File : array_to_raster.py

from osgeo import gdal, gdal_array
import numpy as np
from typing import Tuple, Union, List
from pathlib import Path


def array2Raster(array: Union[np.ndarray, List], outputPath: str, refImg=None, geoTransform=None, crs=None, gType=None,
                 noDataValue=None, driverName='GTiff'):
    '''
    :param array: np.array(rows * cols) or np.array(rows * cols * band_num) or List[np.array(rows * cols)]
    :param outputPath: 输出路径
    :param refImg: 参考影像，提供geoTransform, crs和nodata, 如果后面提供了对应参数，该参数会被覆盖
    :param geoTransform: geoTransform[0]：左上角像素经度
                        geoTransform[1]：影像宽度方向上的分辨率(经度范围/像素个数)
                        geoTransform[2]：旋转, 0表示上面为北方
                        geoTransform[3]：左上角像素纬度
                        geoTransform[4]：旋转, 0表示上面为北方
                        geoTransform[5]：影像宽度方向上的分辨率(纬度范围/像素个数)
    :param crs: 坐标系
    :param gType: 像元类型，默认与array相同
    :param noDataValue:
    :param driverName: gdal driver name
    :return:
    '''

    if isinstance(array, list):
        array = np.array(array).transpose(1, 2, 0)

    refDs = None
    refTransform = None
    refCrs = None
    refNoDataValue = None

    if refImg:
        if isinstance(refImg, str) and Path(refImg).is_file():
            refDs = gdal.Open(refImg)
        elif isinstance(refImg, gdal.Dataset):
            refDs = refImg

    if refDs:
        refTransform = refDs.GetGeoTransform()
        refCrs = refDs.GetProjection()
        refNoDataValue = refDs.GetRasterBand(1).GetNoDataValue()

    if not geoTransform:
        geoTransform = refTransform

    if not crs:
        crs = refCrs

    if not gType:
        gType = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)

    if not noDataValue:
        noDataValue = refNoDataValue

    # 全部转为rows * cols * band_num
    cols = array.shape[1]
    rows = array.shape[0]

    if array.ndim == 2:
        array = array.reshape((rows, cols, 1))
    bandNum = array.shape[2]

    driver: gdal.Driver = gdal.GetDriverByName(driverName)
    outDs: gdal.Dataset = driver.Create(outputPath, cols, rows, bandNum, gType)
    outDs.SetGeoTransform(geoTransform)
    outDs.SetProjection(crs)

    for i in range(0, bandNum):
        band: gdal.Band = outDs.GetRasterBand(i + 1)  # 在GDAL中, band是从1开始索引的
        band.WriteArray(array[..., i])
        if noDataValue:
            band.SetNoDataValue(noDataValue)
        band.FlushCache()
    print(f'Get {outputPath}')
