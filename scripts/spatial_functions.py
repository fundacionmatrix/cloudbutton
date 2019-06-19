import gdal, ogr, osr
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely as shp
import os
from rsgislib.segmentation import segutils
from rsgislib import vectorutils
import rasterstats
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import richdem
import psycopg2
from rpy2.robjects.packages import importr
from rpy2 import robjects
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.packages as rpackages

def create_folders(tmp):
    dict_paths = {
        "LiDAR_subset" : f"{tmp}/LiDAR_subset",
        "layers" : f"{tmp}/layers",
        "images" : f"{tmp}/images",
        "crop_images" : f"{tmp}/image_tiles/crop_images",
        "crop_means" : f"{tmp}/image_tiles/crop_means",
        "crop_stacks" : f"{tmp}/image_tiles/crop_stacks",
        "lidar_tifs" : f"{tmp}/image_tiles/lidar_tifs",
        "las_tiles" : f"{tmp}/las_tiles",
        "seg_crop" : f"{tmp}/image_tiles/seg_crop",
        "segemented_tiles" : f"{tmp}/image_tiles/segmented_tiles",
        "crop_stackForStats" : f"{tmp}/image_tiles/crop_stackForstats",
        "segmented_stats" : f"{tmp}/image_tiles/segmented_stats",
        "tile_prediction" : f"{tmp}/image_tiles/tile_prediction",
        "forest_fuel" : f"{tmp}/image_tiles/forest_fuel"
    }

    out = list(map(lambda x: os.makedirs(x[1], exist_ok = True), dict_paths.items()))
    return(dict_paths)

def get_layerextent(layer_path):
    longitud = len(layer_path.split("."))
    driver_name = layer_path.split(".")[longitud - 1]
    if driver_name == "gpkg":
        driver = ogr.GetDriverByName("GPKG")
    if driver_name == "shp":
        driver = ogr.GetDriverByName("ESRI Shapefile")

    ds = driver.Open(layer_path)
    xmin, xmax, ymin, ymax = ds.GetLayer().GetExtent()
    extent = f"{xmin}, {ymin}, {xmax}, {ymax}"

    del ds

    return (extent)


def get_rasterextent(raster, dictionary = False):
    r = gdal.Open(raster)
    ulx, xres, xskew, uly, yskew, yres = r.GetGeoTransform()
    lrx = ulx + (r.RasterXSize * xres)
    rly = uly + (r.RasterYSize * yres)

    # xmin, xmax, ymin and ymax
    extent = [ulx, lrx, rly, uly]
    if dictionary:
        return({raster: extent})
    else:
        return (extent)


def createGrid(path_to_layer, spacing = 10000, epsg = 25830, buffer = None, output=None):
    gdf = gpd.read_file(path_to_layer)
    xmin, ymin, xmax, ymax = gdf.total_bounds

    # To be sure the bounding box created has all the set of points inside and it is multiple of the spacing.

    ytop = np.ceil(np.ceil(ymax) / spacing) * spacing
    ybottom = np.floor(np.floor(ymin) / spacing) * spacing
    xright = np.ceil(np.ceil(xmax) / spacing) * spacing
    xleft = np.floor(np.floor(xmin) / spacing) * spacing

    # Defining number of rows and columns
    rows = int((ytop - ybottom) / spacing)
    cols = int((xright - xleft) / spacing)

    polygons = []
    it = 0
    listfid = []
    for i in np.arange(xleft, xright, spacing):
        xleft = i
        xright = xleft + spacing
        ytop_backup = ytop
        for j in np.arange(ytop, ybottom, -spacing):
            ytop = j
            ybottom = ytop - spacing

            polygon = shp.geometry.Polygon([
                (xleft, ytop),
                (xright, ytop),
                (xright, ybottom),
                (xleft, ybottom)
            ]
            )
            polygons.append(polygon)
            listfid.append(it)
            it += 1
        ytop = ytop_backup
        # print(f"xleft: {xleft} xright: {xright} \n ytop: {ytop} ybottom: {ybottom}")

    # print(polygons)
    srs = f"epsg:{epsg}"
    fid = pd.DataFrame({"fid_id": listfid})
    grid = gpd.GeoDataFrame(fid, geometry=polygons, crs={"init": srs})

    if output is not None:
        print("Writing grid into disk")
        grid.to_file(output, driver="GPKG")

    #################################################
    ## BETTER TO RETURN JUST INTERSECTING POLYGONS ##
    #################################################
    if buffer:
        buf = grid.geometry.buffer(buffer)
        envelope = buf.envelope
        return(envelope)
    else:
        return(grid)


def reproject(image, output_folder, epsg_to=3035, return_output_path = False):
    splitted = image.split("/")
    lenout = len(splitted)
    out_name = splitted[lenout-1]
    output = f"{output_folder}/reprojeted_{out_name}"

    dataset = gdal.Open(image)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_to)
    vrt_ds = gdal.AutoCreateWarpedVRT(dataset, None, srs.ExportToWkt(), gdal.GRA_NearestNeighbour)

    # cols = vrt_ds.RasterXSize
    # rows = vrt_ds.RasterYSize
    gdal.GetDriverByName("GTiff").CreateCopy(output, vrt_ds)

    if return_output_path:
        return(output)
    else:
        return(vrt_ds)


# def masking_tiles(layer_tiles, raster_path, output_folder, field="fid_id"):
#     if os.path.exists(output_folder) is False:
#         os.mkdir(output_folder)
#
#     driver = ogr.GetDriverByName("GPKG")
#     ds = driver.Open(layer_tiles)
#     layer = ds.GetLayer()
#     for feature in layer:
#         geom = feature.geometry()
#         fid = feature.GetField(field)
#         out_name = naming_convention(raster_path, geom)
#         # print(out_name)
#         output = f"{output_folder}/{out_name}.tif"
#
#
#         ds2 = gdal.Warp(output,
#                         raster_path,
#                         format="GTiff",
#                         cutlineDSName=layer_tiles,
#                         cutlineWhere=f"{field} == '{fid}'",
#                         cropToCutline=True)
#     layer.ResetReading()
#     ds.FlushCache()
#     del ds2
#     del ds

def masking_tiles(layer_tiles,
                  raster_path,
                  output_folder,
                  field="fid_id",
                  naming=False,
                  extent=False,
                  lesser_lextent=False,
                  reproyectar=False,
                  epsg=25830
                  ):
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)

    if reproyectar:
        raster_path2 = raster_path
        raster_path = reproject(raster_path, "/tmp", epsg_to=epsg, return_output_path=True)
        print(raster_path)

    driver = ogr.GetDriverByName("GPKG")
    ds = driver.Open(layer_tiles)
    layer = ds.GetLayer()
    for feature in layer:
        geom = feature.geometry()
        fid = feature.GetField(field)
        if naming:
            if reproyectar:
                out_name = naming_convention(raster_path2, geom)
            else:
                out_name = naming_convention(raster_path, geom)
        else:
            out_tmp = raster_path.split("/")
            out_tmp2 = out_tmp[len(out_tmp) - 1]
            out_name = out_tmp2.split(".")[0]

        output = f"{output_folder}/{out_name}.tif"

        if extent:
            raster_extent = get_rasterextent(raster_path)
            sepuede = layer_within_raster(raster_extent, geom, lesser_lextent=lesser_lextent)

            if sepuede:
                xmin, xmax, ymin, ymax = geom.GetEnvelope()
                lextent = [xmin, ymin, xmax, ymax]

                ds2 = gdal.Warp(output,
                                raster_path,
                                format="GTiff",
                                outputBounds=lextent)

                del ds2

        else:
            ds2 = gdal.Warp(output,
                            raster_path,
                            format="GTiff",
                            cutlineDSName=layer_tiles,
                            cutlineWhere=f"{field} = '{fid}'",
                            cropToCutline=True)
            del ds2

    layer.ResetReading()
    ds.FlushCache()

    del ds



def mask_all_images(folder_images, layer_tiles, output_folder, field = "fid_id", lesser_lextent = False, reproyectar = False, naming = False, bandas = ["B02", "B03", "B04", "B08"]):
    for folder in os.listdir(folder_images):
        if ".zip" not in folder:
            newfolder = f"{folder_images}/{folder}"
            for folder2 in os.listdir(newfolder):
                if "GRANULE" in folder2:
                    newnewfolder = f"{newfolder}/{folder2}"
                    for folder3 in os.listdir(newnewfolder):
                        newnewnewfolder = f"{newnewfolder}/{folder3}"
                        for img_data in os.listdir(newnewnewfolder):
                            if "IMG_DATA" in img_data:
                                newnewnewnewfolder = f"{newnewnewfolder}/{img_data}"
                                for image in os.listdir(newnewnewnewfolder):
                                    if image.endswith(".jp2"):

                                        # Subsetting bands
                                        tmp = image.split(".")[0]
                                        tile, date, band = tmp.split("_")
                                        if band in bandas:
                                            path_image = f"{newnewnewnewfolder}/{image}"
                                            print(path_image)
                                            masking_tiles(layer_tiles = layer_tiles,
                                                          raster_path = path_image,
                                                          output_folder = output_folder,
                                                          field = field,
                                                          reproyectar = reproyectar,
                                                          naming = naming,
                                                          lesser_lextent = lesser_lextent)


def mask_lidarImages_folder(tiles_folder, variable, field="fid_id", tile_spacing = 250, lesser_lextent = False):
    for folder in os.listdir(tiles_folder):
        print(folder)
        newfolder = f"{tiles_folder}/{folder}"  # Coordinates folder
        layer_tiles = f"{newfolder}/{folder}_tiles_{tile_spacing}m.gpkg"
        #         print(layer_tiles)
        if os.path.isdir(newfolder):
            for folder2 in os.listdir(newfolder):

                # DEM folder
                newnewfolder = f"{newfolder}/{folder2}"
                if folder2 == variable:

                    # images inside folder
                    for image in os.listdir(newnewfolder):
                        if os.path.isdir(f"{newnewfolder}/{image}") is False:
                            path_image = f"{newnewfolder}/{image}"
                            output = f"{newnewfolder}/mask"

                            masking_tiles(layer_tiles=layer_tiles,
                                          raster_path=path_image,
                                          output_folder=output,
                                          field=field, extent=True, lesser_lextent = lesser_lextent)


def create_raster(in_ds, fn, data, data_type, nodata=None, driver="GTiff", band_names=None):
    """
    Based on Geoprocessing with python.
    Create a one-band GeoTiff

    in_ds         - datasource to copy projection and geotransform from
    fn            - path to the file to create
    data          - NumPy array containing data to write
    data_type     - output data type
    nodata        - optional NoData value
    band_names    - optional. It gives a name to each band for easier identification. It has to have same length than data dimensons.
    """

    driver = gdal.GetDriverByName(driver)
    #     print(band_names)
    # Creating out raster framework
    columns = in_ds.RasterXSize
    rows = in_ds.RasterYSize
    try:
        nbands = int(data.shape[2])
    except:
        nbands = 1
    out_ds = driver.Create(fn, columns, rows, nbands, data_type)

    # Assigning out raster projection and geotransform
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())

    # Iterate through bands if necessary
    if nbands > 1:
        for k in range(0, nbands):
            out_band = out_ds.GetRasterBand(k + 1)
            if nodata is not None:
                out_band.SetNoDataValue(nodata)
            out_band.WriteArray(data[:, :, k])

            if band_names is not None:
                out_band.SetDescription(band_names[k])
                metadata = out_band.GetMetadata()
                metadata = f"TIFFTAG_IMAGEDESCRIPTION={band_names[k]}"
                out_band.SetMetadata(metadata)

    else:
        out_band = out_ds.GetRasterBand(1)
        if nodata is not None:
            out_band.SetNoDataValue(nodata)

        out_band.WriteArray(data)
    #         print(out_band.ReadAsArray())

    out_band.FlushCache()
    out_band.ComputeStatistics(False)
    del out_ds


def temporal_band_means(path, out_path, bandas=["B02", "B03", "B04", "B08"]):
    def get_array(file):
        path_to_file = f"{path}/{file}"
        ds = gdal.Open(path_to_file)
        arr = ds.GetRasterBand(1).ReadAsArray()
        return (arr)
        del ds

    def get_band_names(name):
        band = name[31:34]
        out = f"{band}"
        return (out)

    date = []
    coordinates = []
    band = []
    tile = []

    listfiles = os.listdir(path)
    for file in listfiles:
        date.append(file[:8])
        coordinates.append(file[9:27])
        band.append(file[35:38])
        tile.append(file[28:34])

    date = set(date)
    coordinates = set(coordinates)
    band = set(band)
    tile = set(tile)



    # Grouping by tile
    for t in tile:
        tmp1 = list(filter(lambda x: x[28:34] == t, listfiles))

        #         print(tmp1)
        for c in coordinates:
            # getting files based on coordinates
            tmp2 = list(filter(lambda x: x[9:27] == c, tmp1))
            tmp2.sort()
            #             print(tmp2)
            lista_means = []
            bnames = []
            for b in band:
                if b in bandas:
                    list_bandas = list(filter(lambda x: x[35:38] == b, tmp2))
                    #                     bnames = list(map(lambda x: get_band_names(x), list_bandas))
                    arrays = list(map(get_array, list_bandas))
                    forstack = np.stack(arrays, axis=2)
                    lista_means.append(np.mean(forstack, axis=2))
                    bnames.append(b)

            #             print(bnames)
            # Stacking band means on a array
            stack_mean = np.stack(lista_means, axis=2)
            pathds = f"{path}/{list_bandas[0]}"
            ds_tmp = gdal.Open(pathds)
            create_raster(in_ds=ds_tmp, fn=f"{out_path}/tbandMean_{list_bandas[0][9:34]}.tif",
                          data=stack_mean, data_type=gdal.GDT_Float32, band_names=bnames)


def tile_temporal_stack(path, out_path, bandas=["B02", "B03", "B04", "B08"]):
    def get_array(file):
        path_to_file = f"{path}/{file}"
        ds = gdal.Open(path_to_file)
        arr = ds.GetRasterBand(1).ReadAsArray()
        return (arr)
        del ds

    def get_band_names(name):
        date = name[:8]
        band = name[35:38]
        out = f"{date}_{band}"
        return (out)

    # Saving date, coordinates and band of tiles
    date = []
    coordinates = []
    band = []
    tile = []
    listfiles = os.listdir(path)
    for file in listfiles:
        date.append(file[:8])
        coordinates.append(file[9:27])
        band.append(file[35:38])
        tile.append(file[28:34])

    date = set(date)
    coordinates = set(coordinates)
    band = set(band)
    tile = set(tile)

    # getting all date and bands for each tile
    for t in tile:
        tmp1 = list(filter(lambda x: x[28:34] == t, listfiles))
        tmp2 = list(filter(lambda x: x[35:38] in bandas, tmp1))
        for c in coordinates:
            tmp3 = list(filter(lambda x: x[9:27] == c, tmp2))
            tmp3.sort()
            bnames = list(map(lambda x: get_band_names(x), tmp3))

            # Getting arrays and stacking arrays
            arrays = list(map(get_array, tmp3))
            data = np.stack(arrays, axis=2)

            # saving out
            pathds = f"{path}/{tmp3[0]}"
            ds_tmp = gdal.Open(pathds)
            create_raster(in_ds=ds_tmp, fn=f"{out_path}/tStack_{tmp3[0][9:34]}.tif",
                          data=data, data_type=gdal.GDT_Float32, band_names=bnames)

def create_grid_from_name(folder_with_tiles, spacing = 500, intile_length = 1000, epsg = 25830):
    for folder in os.listdir(folder_with_tiles):
        xmin, ymax = folder.split("-")
        xmax = float(xmin) + intile_length
        ymin = float(ymax) - intile_length

        ytop = float(ymax)
        ybottom = ymin
        xleft = float(xmin)
        xright = xmax

        # Defining number of rows and columns
        rows = int((ytop - ybottom) / spacing)
        cols = int((xright - xleft) / spacing)

        polygons = []
        it = 0
        listfid = []
        for i in np.arange(xleft, xright, spacing):
            xleft = i
            xright = xleft + spacing
            ytop_backup = ytop
            for j in np.arange(ytop, ybottom, -spacing):
                ytop = j
                ybottom = ytop - spacing

                polygon = shp.geometry.Polygon([
                    (xleft, ytop),
                    (xright, ytop),
                    (xright, ybottom),
                    (xleft, ybottom)
                ]
                )
                polygons.append(polygon)
                listfid.append(it)
                it += 1
            ytop = ytop_backup
            # print(f"xleft: {xleft} xright: {xright} \n ytop: {ytop} ybottom: {ybottom}")

        # print(polygons)
        srs = f"epsg:{epsg}"
        fid = pd.DataFrame({"fid_id": listfid})
        grid = gpd.GeoDataFrame(fid, geometry=polygons, crs={"init": srs})

        print("Writing grid into disk")
        output = f"{folder_with_tiles}/{folder}/{folder}_tiles_{spacing}m.gpkg"

        if os.path.exists(output):
            os.remove(output)
        grid.to_file(output, driver="GPKG")

def naming_convention(raster_path, geometry):
    splitted = raster_path.split("/")
    len_splitted = len(splitted)
    name_tmp1 = splitted[len_splitted - 1]
    name = name_tmp1.split(".")[0]
    name_splitted = name.split("_")
    sent_tile = name_splitted[0]
    band = name_splitted[2]
    date_tmp = name_splitted[1]
    date = date_tmp.split("T")[0]
    # xmin, xmax, ymin, ymax
    xmin, xmax, ymin, ymax = geometry.GetEnvelope()
    # outaname = f"{date}_{int(xmin)}_{int(ymax)}_{sent_tile}_{band}"
    outaname = f"{date}_{float(xmin)}-{float(ymax)}_{sent_tile}_{band}"
    return (outaname)


def layer_within_raster(raster_extent, layer_geom, lesser_lextent=False):
    rxmin, rxmax, rymin, rymax = raster_extent
    lxmin, lxmax, lymin, lymax = layer_geom.GetEnvelope()

    if lesser_lextent:
        # Getting a smaller bounding box
        lxmin = lxmin + 100
        lxmax = lxmax - 100
        lymin = lymin + 100
        lymax = lymax - 100

    i = 0
    if lxmin >= rxmin:  # 1. upper left corner
        i += 1
    if lymax <= rymax:  # 2. upper right corner
        i += 1
    if lxmax <= rxmax:  # 3. lower right corner
        i += 1
    if lymin >= rymin:  # 4. lower left corner
        i += 1

    if i == 4:
        out = True
    else:
        out = False
    return (out)


def upto_ImageTile(ImageTiles_layer, rasterTiles_folder, output_folder, res=10, variable="DEM"):
    def list_right_rasters(folders_inside, previous_path, variable=variable):
        if len(folders_inside) > 0:
            folder = list(folders_inside.keys())[0]
            after_path = f"{variable}/mask"
            final_path = f"{previous_path}/{folder}/{after_path}"

            listfiles = list(map(lambda x: x.path, os.scandir(final_path)))
        print(listfiles)
        return (listfiles)

    def layer_within_folderExtent(folder, layer_geom, spacing=2000):
        rxmin, rymax = list(map(lambda x: float(x), folder))
        folder_name = f"{folder[0]}-{folder[1]}"
        rxmax = rxmin + spacing
        rymin = rymax - spacing
        lxmin, lxmax, lymin, lymax = layer_geom.GetEnvelope()

        i = 0
        if lxmin >= rxmin:  # 1. upper left corner
            i += 1
        if lymax <= rymax:  # 2. upper right corner
            i += 1
        if lxmax <= rxmax:  # 3. lower right corner
            i += 1
        if lymin >= rymin:  # 4. lower left corner
            i += 1

        if i == 4:
            out = {folder_name: True}
        else:
            out = {folder_name: False}

        #     print(lxmin, lxmax, lymin, lymax)

        #     print(layer_geom.GetEnvelope())
        return (out)



    tiles_folder = rasterTiles_folder
    layer_path = ImageTiles_layer
    output_variable = f"{output_folder}/{variable}"

    if os.path.exists(output_variable) is False:
        os.makedirs(output_variable, exist_ok = True)

    ds = ogr.Open(layer_path)
    layer = ds.GetLayer()
    vrt_options = gdal.BuildVRTOptions(resampleAlg="nearest", addAlpha=False, xRes=res, yRes=res)

    folders = os.listdir(tiles_folder)
    # path_folders = list(map(lambda x: f"{tiles_folder}/{x}", folders))

    folders_xy = list(map(lambda x: [x.split("-")[0], x.split("-")[1]], folders))
    # print(path_folders)

    for feature in layer:
        geom = feature.GetGeometryRef()
        lxmin, lxmax, lymin, lymax = geom.GetEnvelope()


        is_it_inside = list(map(layer_within_folderExtent, folders_xy, [geom] * len(folders_xy)))
        wecan = list(filter(lambda x: list(x.values())[0] is True, is_it_inside))
        # print(wecan)
        listoflists = list(map(list_right_rasters, wecan, [rasterTiles_folder]*len(wecan)))
        finallist = sum(listoflists, [])

        if len(finallist) > 0:
            dsvrt = gdal.BuildVRT(f"{output_variable}/a_tmp.vrt", finallist, options=vrt_options)
            dsvrt.ReadAsArray()
            dstif = gdal.Translate(f"{output_variable}/{variable}_{lxmin}-{lymax}.tif", dsvrt)
            del dsvrt
            del dstif

            os.remove(f"{output_variable}/a_tmp.vrt")

    del ds


def tiles_for_segmentation(crop_means, path_dsms, output):
    def get_band_name(band):
        option_name = "TIFFTAG_IMAGEDESCRIPTION"
        md = band.GetMetadata()
        for k, v in md.items():
            if k == option_name:
                return (v)

    def same_coordinates(a, b):
        avariable, acoordinates, atile = a.split("_")
        bvariable, bcoordinates = b.split("_")
        bcoordinates = bcoordinates.split(".tif")[0]
        #     print({acoordinates : bcoordinates})
        if acoordinates == bcoordinates:
            return (True)
        else:
            return (False)

    lista_crops = os.listdir(crop_means)
    lista_dsm = os.listdir(path_dsms)

    crop_means_paths = list(map(lambda x: f"{crop_means}/{x}", lista_crops))
    for crop in crop_means_paths:

        for dsm in lista_dsm:
            crop_tmp = crop.split("/")
            crop_tmp2 = crop_tmp[len(crop_tmp) - 1]
            #         print(crop_tmp2)
            if same_coordinates(crop_tmp2, dsm):
                #             print(crop_tmp2)
                ds = gdal.Open(crop)
                count = ds.RasterCount
                diccionario = {}

                for bandi in range(1, count + 1):
                    band = ds.GetRasterBand(bandi)
                    band_name = get_band_name(band)
                    arr = band.ReadAsArray()
                    diccionario[band_name] = arr

                path_dsm = f"{path_dsms}/{dsm}"
                ds_dsm = gdal.Open(path_dsm)
                dsm_band = ds_dsm.GetRasterBand(1)
                arr_dsm = dsm_band.ReadAsArray()
                diccionario["DSM"] = arr_dsm

                bands_list = []
                arrays_list = []
                for k, v in diccionario.items():
                    bands_list.append(k)
                    arrays_list.append(v)

                array_stacked = np.stack(arrays_list, axis=2)
                out_path = f"{output}/SegCrop_{crop_tmp2[10:]}"

                create_raster(in_ds=ds,
                              fn=out_path,
                              data=array_stacked,
                              data_type=gdal.GDT_Float32,
                              band_names=bands_list
                              )

        ds = None
        ds_dsm = None


def segmenta(entrada, salida="resultado.shp",minPxls=100,distThres=100):
    '''
    Realiza la segmentación del archivo que se pasa como entrada en un archivo shp de salida
    '''
    GRUPOS="tempGrupos.kea"
    RESULTADO="tempImg.kea"
    segutils.runShepherdSegmentation(entrada, GRUPOS, RESULTADO, minPxls=minPxls, distThres=distThres)
    vectorutils.polygoniseRaster(RESULTADO,salida)
    os.remove(GRUPOS)
    os.remove(RESULTADO)

def segmenta_all_tiles(seg_crop, segmented_tiles, minPxls = 100, distThres = 100):
    """
    :param seg_crop: folder with raster images prepared for segmentation
    :param segmented_tiles: output folder for segmentation layers.
    :return:
    """
    def get_outputs(x):
        out = x.split('.tif')[0][8:]
        return(out)

    list_segcrop = os.listdir(seg_crop)
    list_forseg = list(map(lambda x: f"{seg_crop}/{x}", list_segcrop))
    list_outputs = list(map(lambda x: f"{segmented_tiles}/Segmented_{get_outputs(x)}.shp", list_segcrop))
    OUT = list(map(lambda x, y, min, dist: segmenta(entrada = x, salida = y),
                   list_forseg,
                   list_outputs,
                   [minPxls] * len(list_forseg),
                   [distThres] * len(list_forseg)
                   ))


# def merge_layers(input_folder, output, format = "shp"):
#     list_files = list(map(lambda x: f"{input_folder}/{x}", os.listdir(input_folder)))
#     list_shps = list(filter(lambda x: format in x, list_files))
#     # print(list_shps)
#     vectorutils.mergeShapefiles(list_shps, output)

def merge_layers(inputFolder, outputFile, oformat="GPKG", epsg=25830, informat="ESRI Shapefile"):
    geometryType = ogr.wkbPolygon
    driver = ogr.GetDriverByName(oformat)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)

    if os.path.exists(outputFile):
        driver.DeleteDataSource(outputFile)
    out_ds = driver.CreateDataSource(outputFile)

    # Creating output layer name
    layer_name_tmp = outputFile.split(".")[0]
    layer_name_tmp2 = layer_name_tmp.split("/")
    layer_name = layer_name_tmp2[len(layer_name_tmp2) - 1]

    # Creating output layer
    out_layer = out_ds.CreateLayer(layer_name, srs, geom_type=geometryType)

    # Reading input layers
    list_layers = list(map(lambda x: f"{inputFolder}/{x}", os.listdir(inputFolder)))
    if informat == "ESRI Shapefile":
        list_shps = list(filter(lambda x: "shp" in x, list_layers))
    else:
        list_shps = list_layers

    for shp in list_shps:
        in_ds = ogr.Open(shp)
        if in_ds is None: return ("Flipao")
        in_layer = in_ds.GetLayer()
        in_layer_defn = in_layer.GetLayerDefn()

        # Add fields
        nfields = in_layer_defn.GetFieldCount()
        for i in range(0, nfields):
            fieldDefn = in_layer_defn.GetFieldDefn(i)
            out_layer.CreateField(fieldDefn)

        # Get output layer definition
        out_layer_defn = out_layer.GetLayerDefn()

        for infeat in in_layer:
            geom = infeat.GetGeometryRef().Clone()
            ofeat = ogr.Feature(out_layer_defn)
            ofeat.SetGeometry(geom)

            # Get in all fields
            for i in range(0, nfields):
                # print(i)
                ofield_name = out_layer_defn.GetFieldDefn(i).GetNameRef()
                ofield_value = infeat.GetField(i)
                ofeat.SetField(ofield_name, ofield_value)

            out_layer.CreateFeature(ofeat)

        del in_ds
    out_layer.SyncToDisk()
    del out_ds


def stacktodo(lidar_tifs, crop_stacks, crop_stackForstats):

    def get_bandName_fromMetadata(data_source):
        option_name = "TIFFTAG_IMAGEDESCRIPTION"
        md = band.GetMetadata()
        for k, v in md.items():
            if k == option_name:
                return (v)


    def get_bandName_fromName(name):
        out = name.split("_")[0]
        return (out)

    def same_coordinates(a, b):
        avariable, acoordinates, atile = a.split("_")
        bvariable, bcoordinates = b.split("_")
        bcoordinates = bcoordinates.split(".tif")[0]
        if acoordinates == bcoordinates:
            return (True)
        else:
            return (False)


    def get_rid_of_path(x):
        tmp = x.split("/")
        out = tmp[len(tmp)-1]
        return(out)

    def get_all_files_from_listoffolders(listfolders):
        tmp = list(map(lambda x:
                       list(map(lambda y:
                                f"{x}/{y}",

                                # Pass list of each folder inside listfolder (x)
                                os.listdir(x)
                                )),
                       listfolders
                       ))
        out = sum(tmp, [])
        return(out)


    list_lidartif_folders = list(map(lambda x: f"{lidar_tifs}/{x}", os.listdir(lidar_tifs)))
    list_lidar_files = get_all_files_from_listoffolders(list_lidartif_folders)
    list_temporal_stacks = list(map(lambda x: f"{crop_stacks}/{x}", os.listdir(crop_stacks)))

    lista2 = []
    for temporal in list_temporal_stacks:
        lista = [temporal]
        for lidar_raster in list_lidar_files:
            if same_coordinates(get_rid_of_path(temporal), get_rid_of_path(lidar_raster)):
                lista.append(lidar_raster)

        lista2.append(lista)

    lists_toStack = list(filter(lambda x: len(x)>1, lista2))


    for toStack in lists_toStack:
        listarrays = []
        listband_names = []
        for tif in toStack:
            tif_name = get_rid_of_path(tif)


            ds = gdal.Open(tif)
            nbands = ds.RasterCount
            diccionario = {}


            for bandi in range(1, nbands + 1):
                band = ds.GetRasterBand(bandi)
                if nbands > 1:
                    band_name = get_bandName_fromMetadata(band)
                else:

                    band_name = get_bandName_fromName(tif_name)


                arr = band.ReadAsArray()

                diccionario[band_name] = arr

            for k,v in diccionario.items():
                listband_names.append(k)
                listarrays.append(v)

        variable, coordinatestif = tif_name.split("_")
        out_path = f"{crop_stackForstats}/{coordinatestif}"

        arrays = np.stack(listarrays, axis = 2)
        create_raster(in_ds = ds,
                      fn = out_path,
                      data = arrays,
                      data_type = gdal.GDT_Float32,
                      band_names = listband_names
                      )
        ds = None


def create_layer(output, feature_list,
                 driver_name="GPKG", epsg=25830,
                 geom_type=ogr.wkbPolygon, data_type=ogr.OFTReal):
    """
    output_name         -  Name of the shapefile to craete with extension
    feature_dictionary  -  list with two elements, geometry and a list with a dictionary with the name of the field at the keys
                            and its values at de values.
    driver_name         -  driver to use. GPKG by default.
    epsg                -  epsg code to assign projection
    geom_type           -  geom type of the geometry suplied
    data_type           -  data_type of the values
    """

    # Getting name of the output without path and extension
    output_layer_tmp = output.split("/")
    output_layer_tmp2 = output_layer_tmp[len(output_layer_tmp) - 1]
    output_layer_name = output_layer_tmp2.split(".")[0]
    #     print(output_layer_name)

    # Getting srs
    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(epsg)
    #     print(out_srs)

    # create output layer
    driver = ogr.GetDriverByName(driver_name)
    if os.path.exists(output):
        driver.DeleteDataSource(output)
    out_ds = driver.CreateDataSource(output)
    out_layer = out_ds.CreateLayer(output_layer_name, geom_type=geom_type, srs=out_srs)

    # very important matter to reset Reading after define out layer
    out_layer.ResetReading()
    #     print(out_layer)

    # Iterate through list to get fields and create them
    for feature in feature_list:
        diccionario_tmp = feature[1]
        diccionario = diccionario_tmp[0]

        for field in diccionario.keys():
            outFieldDefn = ogr.FieldDefn(field, data_type)
            out_layer.CreateField(outFieldDefn)

    # Get Layer Definition
    out_layerDefn = out_layer.GetLayerDefn()
    #     print(out_layerDefn.GetGeomFieldDefn())
    #     print(out_layerDefn)

    # Iterate through list to get geometris, fields and values
    it = 0
    for feature in feature_list:
        geomwkt = feature[0]
        geom = ogr.CreateGeometryFromWkt(geomwkt)

        diccionario_tmp = feature[1]
        diccionario = diccionario_tmp[0]

        ofeat = ogr.Feature(out_layerDefn)
        ofeat.SetGeometry(geom)
        for field, value in diccionario.items():
            ofeat.SetField(field, value)

        #             print(field, value*1.0)

        out_layer.CreateFeature(ofeat)

    out_layer.SyncToDisk()
    out_ds = None


def getting_stats(src_filename, layer_filename, output):
    def get_band_name(band):
        option_name = "TIFFTAG_IMAGEDESCRIPTION"
        md = band.GetMetadata()
        for k, v in md.items():
            if k == option_name:
                return (v)

    #     src_filename = "tmp/image_tiles/crop_stacks/tStack_360000.0-4358000.0_T30SUJ.tif"
    #     layer_filename = "tmp/image_tiles/segmented_tiles/Segmented_360000.0-4358000.0_T30SUJ.shp"

    lds = ogr.Open(layer_filename)
    layer = lds.GetLayer()
    lista_diccionarios = []

    for feat in layer:
        lista_features = []
        geom = feat.GetGeometryRef()
        geomwkt = geom.ExportToWkt()
        lista_features.append(geomwkt)

        geomjson = geom.ExportToJson()

        src_ds = gdal.Open(src_filename)
        RasterCount = src_ds.RasterCount
        dictmp = {}
        for i in range(1, RasterCount + 1):
            banda = src_ds.GetRasterBand(i)
            band_name = get_band_name(banda)
            stats = rasterstats.zonal_stats(geomjson, src_filename, band=i)

            for k, v in stats[0].items():
                field_name = f"{band_name}_{k}"
                dictmp[field_name] = v

        tmp = [dictmp]
        lista_features.append(tmp)
        lista_diccionarios.append(lista_features)

    create_layer(output, lista_diccionarios)

    del lds
    del src_ds



def getting_stats_toFolder(crop_stackForStats, segmented_tiles, segmented_stats):
    diccionario = {}
    for segments in os.listdir(segmented_tiles):
        if "shp" in segments:
            name, coordinates, tileext = segments.split("_")
            tile = tileext.split(".")[0]
            for raster_stack in os.listdir(crop_stackForStats):
                #                     print(raster_stack)
                #                     if tile in raster_stack:
                if coordinates in raster_stack:
                    diccionario[segments] = raster_stack

    for k, v in diccionario.items():
        value, coordinates, tiletmp = k.split("_")
        tile = tiletmp.split(".")[0]
        out_name = f"segStats_{coordinates}_{tile}.gpkg"
        print(out_name)
        segStats_file = f"{segmented_stats}/{out_name}"
        raster_stack_file = f"{crop_stackForStats}/{v}"
        segmented_file = f"{segmented_tiles}/{k}"

        getting_stats(raster_stack_file, segmented_file, segStats_file)



def joinTraining_with_Stats(forTraining, output_training, segmented_stats, training_field = "CODETYPE"):
    ft_ds = ogr.Open(forTraining)
    ft_layer = ft_ds.GetLayer()

    listof_SegStats = list(map(lambda x: f"{segmented_stats}/{x}", os.listdir(segmented_stats)))
    lista_features = []
    for SegStats in listof_SegStats:
    #     print(SegStats)
        ss_ds = ogr.Open(SegStats)
        ss_layer = ss_ds.GetLayer()
        ss_layerDefn = ss_layer.GetLayerDefn()

        # print(1)


        for ft_feat in ft_layer:
            ft_geom = ft_feat.GetGeometryRef()
            ft_geom_wkt = ft_geom.ExportToWkt()

            for ss_feat in ss_layer:
                ss_geom = ss_feat.GetGeometryRef()

                lfeatures = []
                diccionario = {}
                if ft_geom.Contains(ss_geom):
    #                 print("ja")
                    lfeatures.append(ft_geom_wkt)

                    for i in range(ft_feat.GetFieldCount()):
                        field_name = ft_feat.GetFieldDefnRef(i).name
                        field_value = ft_feat.GetField(field_name)
                        if field_name == training_field:
                            diccionario[field_name] = field_value

                    for i in range(ss_feat.GetFieldCount()):
                        field_name = ss_feat.GetFieldDefnRef(i).name
                        field_value = ss_feat.GetField(field_name)
                        diccionario[field_name] = field_value


                    lfeatures.append([diccionario])

                if len(lfeatures) > 0:
                    lista_features.append(lfeatures)




    #     print("\n")

    del ft_ds
    del ss_ds

    create_layer(output_training, lista_features)


def modela(archivoEntrenado, training_field):
    df = gpd.read_file(archivoEntrenado)
    columns = list(filter(lambda x: x not in [training_field, "geometry"], df.columns))
    y = df[training_field]
    X = df[columns]

    clasificador = RandomForestClassifier(n_estimators = 30)
    modelo = clasificador.fit(X, y)
    return(modelo)

def predecir(modelo, tile_for_predict, output, epsg = 25830, training_field = "CODETYPE"):
    df_seg = gpd.read_file(tile_for_predict)
    columns = list(filter(lambda x: x not in [training_field, "geometry"], df_seg.columns))
    geometry = df_seg["geometry"]
    df_topredict = df_seg[columns]
    prediction = modelo.predict(df_topredict)
    df_topredict[training_field] = prediction
    df_topredict["geometry"] = geometry
    crs = {'init': f'epsg:{epsg}'}
    final = gpd.GeoDataFrame(df_topredict, geometry = geometry, crs = crs)
    final.to_file(output, driver = "GPKG")


def model_and_predict(archivoEntrenado, tile_for_predict, output,
                      pca=False, epsg=25830, training_field="CODETYPE"):
    # Modeling
    df = gpd.read_file(output_training)
    columns = list(filter(lambda x: x not in [training_field, "geometry"], df.columns))
    y = df[training_field]
    X = df[columns]
    if pca:
        pca.fit(X)
        X = pca.transform(X)

    clasificador = RandomForestClassifier(n_estimators=30)
    modelo = clasificador.fit(X, y)

    # Predicting
    df_seg = gpd.read_file(tile_for_predict)
    columns = list(filter(lambda x: x not in [training_field, "geometry"], df_seg.columns))
    geometry = df_seg["geometry"]
    df_topredict = df_seg[columns]
    if pca:
        df_topredict = pca.transform(df_topredict)

    prediction = modelo.predict(df_topredict)
    print(type(prediction))
    #     df_topredict[training_field] = prediction
    diccionario = {training_field: prediction}
    df_topredict = pd.DataFrame(data=diccionario)
    df_topredict["geometry"] = geometry
    crs = {'init': f'epsg:{epsg}'}
    final = gpd.GeoDataFrame(df_topredict, geometry=geometry, crs=crs)
    final.to_file(output, driver="GPKG")



def prediction_to_folder(modelo, tile_for_predict, output):

    def prediction_outputName(a):
        tmp = a.split("/")
        tmp2 = tmp[len(tmp)-1]
        variable, coordinates, tilegpkg = tmp2.split("_")
        out = f"Pred_{coordinates}_{tilegpkg}"
        return(out)


    lista_predOut = list(map(lambda x: f"{output}/{prediction_outputName(x)}", tile_for_predict))

    aaaaout = list(map(lambda x,y,z: predecir(x, y, z),
                       [modelo]*len(lista_SegStats),
                       lista_SegStats,
                       lista_predOut
                      ))


def forest_fuel(input_layer, output_layer, training_field="CODETYPE", driver_name="GPKG",
                epsg=25830, geom_type=ogr.wkbPolygon, data_type=ogr.OFTReal
                ):
    print(input_layer)
    print(output_layer)
    print("\n")

    # We must open the file
    in_ds = ogr.Open(input_layer)
    in_layer = in_ds.GetLayer()
    in_layerdefn = in_layer.GetLayerDefn()
    nfields = in_layerdefn.GetFieldCount()

    lista_diccionarios = []
    for feature in in_layer:

        geom = feature.GetGeometryRef().ExportToWkt()

        # Iterating through fields
        diccionario = {}
        diccionario[training_field] = None
        for i in range(nfields):
            field_name = in_layerdefn.GetFieldDefn(i).name
            #             print(field_name)
            if "mean" in field_name:
                if field_name in ["TCH_mean", "SCH_mean", "FCC_mean", "SCC_mean", "TCC_mean", training_field]:
                    field_value = feature.GetField(field_name)
                    diccionario[field_name] = field_value

        if diccionario[training_field] in [1, 2, 5, 9]:
            fuel_type = 0
        else:
            if diccionario["SCC_mean"] < 60:
                fuel_type = 1
            if diccionario["SCC_mean"] >= 60 and diccionario["TCC_mean"] < 50:
                meanHeight = np.mean([diccionario["TCH_mean"], diccionario["SCH_mean"]])
                if meanHeight >= 0.3 and meanHeight <= 0.6: fuel_type = 2
                if meanHeight > 0.6 and meanHeight <= 2.0: fuel_type = 3
                #                 if meanHeight > 2.0 and meanHeight <= 4.0: fuel_type = 4
                if meanHeight > 2.0:
                    fuel_type = 4
                else:
                    fuel_type = 1
            if diccionario["TCC_mean"] >= 50:
                if diccionario["SCC_mean"] < 30: fuel_type = 5
                if diccionario["SCC_mean"] >= 30:
                    diffHeight = diccionario["TCH_mean"] - diccionario["SCH_mean"]
                    if diffHeight >= 0.5: fuel_type = 6
                    if diffHeight < 0.5: fuel_type = 7

        diccionario["fuel_type"] = fuel_type
        lista_diccionarios.append([geom, [diccionario]])

    #         print("\n")

    in_ds = None

    create_layer(output_layer, lista_diccionarios, driver_name=driver_name,
                 epsg=epsg, geom_type=geom_type, data_type=data_type
                 )


def forest_fuel_to_folder(input_folder, output_folder):
    def ForestFuel_outputName(a):
        tmp = a.split("/")
        tmp2 = tmp[len(tmp) - 1]
        variable, coordinates, tilegpkg = tmp2.split("_")
        out = f"ForestFuel_{coordinates}_{tilegpkg}"
        return (out)

    listinputs = list(map(lambda x: f"{input_folder}/{x}", os.listdir(input_folder)))
    listoutputs = list(map(lambda x: f"{output_folder}/{ForestFuel_outputName(x)}", listinputs))
    #     print(listoutputs)

    jcintasr = list(map(lambda x, y: forest_fuel(x, y), listinputs, listoutputs))


def pendiente(src,dst=None):
    try:
        fichero=richdem.LoadGDAL(src)
    except:
        print("No existe el archivo")
        return False
    try:
        #se calcula la pendiente en radianes
        pendiente=richdem.TerrainAttribute(fichero, attrib='slope_radians')
    except:
        print("El archivo no tiene un formato válido")
        return False
    if not dst:
        dst=src.replace(".tif","_pendiente.tif")
    richdem.SaveGDAL(dst,pendiente)
    print("Se ha calculado la pendiente: "+dst)
    return True


def orientacion(src,dst=None):
    try:
        fichero=richdem.LoadGDAL(src)
    except:
        print("No existe el archivo")
        return False
    try:
        #se calcula la orientación en grados
        pendiente=richdem.TerrainAttribute(fichero, attrib='aspect')
    except:
        print("El archivo no tiene un formato válido")
        return False
    if not dst:
        dst=src.replace(".tif","_orientacion.tif")
    richdem.SaveGDAL(dst,pendiente)
    print("Se ha calculado la orientación: "+dst)
    return True

def slope_to_folders(las_tiles):
    variable = "slope"
    def getFile_FromPath(a):
        b = a.split("/")
        b = b[len(b)-1]
        return(b)

    list_coordinates_folder = list(map(lambda x: f"{las_tiles}/{x}", os.listdir(las_tiles)))
    for coordinates_folder in list_coordinates_folder:
        coordinates = getFile_FromPath(coordinates_folder)
        tmp = list(map(lambda x: f"{coordinates_folder}/{x}", os.listdir(coordinates_folder)))
        dem_folder = list(filter(lambda x: "DEM" in x, tmp)) # DEM folder
        demFolder_insides = list(map(lambda x: f"{dem_folder[0]}/{x}", os.listdir(dem_folder[0])))
        dem_tifs = list(filter(lambda x: ".tif" in x, demFolder_insides))

        output_folder = f"{las_tiles}/{coordinates}/{variable}"
        os.makedirs(output_folder, exist_ok = True)
        output_files = list(map(lambda x: f"{output_folder}/slope_{getFile_FromPath(x)[4:]}", dem_tifs))

        diccionario = list(map(lambda x,y: {x:y}, dem_tifs, output_files))
        for dicc in diccionario:
            for k,v in dicc.items():
                pendiente(k,v)

def aspect_to_folders(las_tiles):
    variable = "aspect"

    def getFile_FromPath(a):
        b = a.split("/")
        b = b[len(b) - 1]
        return (b)

    list_coordinates_folder = list(map(lambda x: f"{las_tiles}/{x}", os.listdir(las_tiles)))
    for coordinates_folder in list_coordinates_folder:
        coordinates = getFile_FromPath(coordinates_folder)
        tmp = list(map(lambda x: f"{coordinates_folder}/{x}", os.listdir(coordinates_folder)))
        dem_folder = list(filter(lambda x: "DEM" in x, tmp))  # DEM folder
        demFolder_insides = list(map(lambda x: f"{dem_folder[0]}/{x}", os.listdir(dem_folder[0])))
        dem_tifs = list(filter(lambda x: ".tif" in x, demFolder_insides))

        output_folder = f"{las_tiles}/{coordinates}/{variable}"
        #         print(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        output_files = list(map(lambda x: f"{output_folder}/aspect_{getFile_FromPath(x)[4:]}", dem_tifs))

        diccionario = list(map(lambda x, y: {x: y}, dem_tifs, output_files))
        for dicc in diccionario:
            for k, v in dicc.items():
                orientacion(k, v)

def reclass_aspect(inputfile, outputfile):
    ds = gdal.Open(inputfile)
    arr = ds.GetRasterBand(1).ReadAsArray()
    arr2 = np.select([(arr >= 45) & (arr < 135),
                      (arr >= 135) & (arr < 225),
                      (arr >= 225) & (arr < 315),
                      (arr >= 315) | (arr < 45)],
                     [1, 2, 3, 4] # 1 = W, 2 = S, 3 = E, 4 = N
                     )
    create_raster(ds, outputfile, arr2, data_type = gdal.GDT_Int16)
    ds = None


def reclass_aspect_to_folder(lidar_tifs, variable_folder = "aspect"):
    def getFile_FromPath(a):
        b = a.split("/")
        b = b[len(b) - 1]
        return (b)

    out = "aspect_reclass"
    input_folder = f"{lidar_tifs}/{variable_folder}"
    output_folder = f"{lidar_tifs}/aspect_reclass"
    os.makedirs(output_folder, exist_ok = True)

    list_diccionarios = list(map(lambda x: {f"{input_folder}/{x}":
                                            f"{output_folder}/aspectReclassed{getFile_FromPath(x)[6:]}"},
                                 os.listdir(input_folder)
                                ))

    for d in list_diccionarios:
        for k,v in d.items():
            reclass_aspect(k, v)

def get_layerextent(layer):
    longitud = len(layer.split("."))
    driver_name = layer.split(".")[longitud - 1]
    if driver_name == "gpkg":
        driver = ogr.GetDriverByName("GPKG")
    if driver_name == "shp":
        driver = ogr.GetDriverByName("ESRI Shapefile")

    ds = driver.Open(layer)
    xmin, xmax, ymin, ymax = ds.GetLayer().GetExtent()
    extent = f"{xmin}, {ymin}, {xmax}, {ymax}"

    del ds

    return (extent)

def read_temperaturas(database, table, periodo, out_file,
                      user="postgres", password="postgres",
                      layer=None, extent=None, srs=25830,
                      temperatura_variable="tm_mes", temperatura_name="tmed",
                      rango_de_años=15, outdriver="GPKG"):

    if layer is not None:
        extent = get_layerextent(layer)

    if extent is not None:
        sql_envelope = f"{extent}, {srs}"

    # Conexión con la base de datos
    connection = psycopg2.connect(database=database,
                                  user=user, password=password)
    cursor = connection.cursor()

    # Consulta sql para un periodo de tiempo
    p1 = periodo[0]
    p2 = periodo[1]
    sql_query = f"""WITH subst AS (
                        select
                                indicativo,
                                count(distinct(year)) as len
                        from
                                {table}
                        where
                                variable = '{temperatura_variable}'
                                and
                                year between {p1} and  {p2}
                        group by
                                indicativo
                        )
                    select
                        a.indicativo,
                        avg(a.values)/10 as {temperatura_name},
                        st_astext(a.geom) as geom
                    from
                        {table} as a
                    inner join
                        subst as b
                    using
                        (indicativo)
                    where
                        a.year between {p1} and {p2}
                        and
                        b.len >= {rango_de_años}
                        and
                        a.variable = '{temperatura_variable}'
                        and
                        st_intersects(a.geom, st_MakeEnvelope({sql_envelope}))
                    group by
                        a.indicativo,
                        a.geom;"""
    cursor.execute(sql_query)

    # Guardando puntos de temperatura en una capa.
    lista = []
    for indicativo, temperatura, geom in cursor:
        data = {'indicativo': indicativo, f'{temperatura_name}': temperatura, 'geometry': shp.wkt.loads(geom)}
        lista.append(data)

    gdf = gpd.GeoDataFrame(lista, crs=f'epsg:{srs}').set_index('indicativo')
    gdf[f'{temperatura_name}'] = gdf[f'{temperatura_name}'].astype("float")
    gdf.to_file(out_file, driver=outdriver)
    print("Archivo creado")



def get_extent_from_r(layer, dist):

    utils = importr("utils")
    utils.chooseCRANmirror(ind=1)
    base = importr("base")


    packages = ("sf")
    names_to_install = []
    for pkg in packages:
         if not rpackages.isinstalled(pkg):
             names_to_install.append(pkg)


    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

    robjects.r('''
        source("get_layerextent.R")
    ''')

    get_layeRextent = robjects.globalenv['get_extent']

    extent = get_layeRextent(layer = layer, dist = dist)
    out_extent = f"{extent[0]}, {extent[1]}, {extent[2]}, {extent[3]}"
    return(out_extent)


def TempRegKrig(points, var, formula, mdt, lat, dist, output, res = 1000):

    utils = importr("utils")
    utils.chooseCRANmirror(ind=1)
    base = importr("base")

    # Check r packages in use
    packages = ("sf", "raster", "automap")
    names_to_install = []
    for pkg in packages:
        if not rpackages.isinstalled(pkg):
            names_to_install.append(pkg)
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

    # Load R script
    robjects.r(f'''
        source("kriging.R")
    ''')

    # print(1)
    # print(type(mdt))
    rasterlist = base.list(mdt = mdt, lat = lat, dist = dist)
    # print(2)
    # print(rasterlist)
    doReKrig = robjects.globalenv['doRegKrig']
    rekriged = doReKrig(points = points, var = var, formula = formula, listrasters = rasterlist, res = res)
    raster = importr("raster")
    raster.writeRaster(rekriged, output, overwrite = True)

if __name__=="__main__":
    create_grid_from_name("tmp/test/tiles", spacing = 250, intile_length = 2000)

    # layer = "input/caba_limites.gpkg"
    # createGrid(path_to_layer = layer, spacing=500, epsg=25830, output = "tmp/layers/caba_500m_tiles.gpkg")

    # imagen = "tmp/images_test/S2A_MSIL1C_20180427T110621_N0206_R137_T30SUJ_20180427T133034/A014863/T30SUJ_20180427T110621_B04.jp2"
    # a = reproject(image = imagen, output_folder = "tmp/reprojected_images", epsg_to=25830)
