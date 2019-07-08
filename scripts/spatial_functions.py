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
    """
    :param tmp: temporal folder where the different and need folders will be created
    :return: dictionary with name of the folder and path
    """

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
        "forest_fuel" : f"{tmp}/image_tiles/forest_fuel",
        "T_tiles" : f"{tmp}/image_tiles/temp_tiles",
        "H_tiles" : f"{tmp}/image_tiles/reH_tiles",
        "PIG_tiles" : f"{tmp}/image_tiles/pig_tiles",
        "fireRisk" : f"{tmp}/image_tiles/fireRisk"
    }

    out = list(map(lambda x: os.makedirs(x[1], exist_ok = True), dict_paths.items()))
    return(dict_paths)

def get_layerextent(layer_path):
    """
    Return the extent of layer with GPKG or ESRI Shapefile formats
    :param layer_path: Path to the layer
    :return: String with xmin, ymin, xmax and ymax coordinates.
    """
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
    """
    Get the extent of a raster image
    :param raster: Path to the raster file
    :param dictionary: If True, a dictionary with the path of the raster as key and the extent of the raster as value is returned.
    :return: The extent with next order: xmin, xmax, ymin and ymax coordinates
    """

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
    """
    Creates a grid from a layer extent.
    :param path_to_layer: Path to layer
    :param spacing: Spacing of the grid cells (it applies to x and y)
    :param epsg: EPSG code of the coordinate
    :param buffer: Buffer to apply to the extent coordinates
    :param output: If defined, it creates an output.
    :return: Grid in geopandas format
    """

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
    """
    This function reprojects a raster image

    :param image: path to raster image
    :param output_folder: output folder where the output image will be saved
    :param epsg_to: coordinate epsg code to reproject into
    :param return_output_path: If True, it returns the output path
    :return: returns a virtual data source
    """
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
    """
    It creates tiles from a raster image based on a grid previously created
    :param layer_tiles: Path to grid
    :param raster_path: Path to raster
    :param output_folder: Path to output folder
    :param field: Field with cut tiles with
    :param naming: Apply naming rule
    :param extent: Cut with extent
    :param lesser_lextent: create an smaller extent
    :param reproyectar: If True, reprojection is applied
    :param epsg: EPSG code of the srs to reproject into
    :return:
    """
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
    """
    Iterate through Sentinel .SAFE format looking for images and mask them with masking_tiles function

    :param folder_images: .SAFE folder
    :param layer_tiles: Path to grid
    :param output_folder: Folder where save outputs
    :param field:  Field to cut with
    :param lesser_lextent: For cutting with a smaller exent
    :param reproyectar: If a apply a reprojection
    :param naming: If apply a naming rule
    :param bandas: list of bands selected bands
    :return:
    """
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
    """
    Mask out the buffers of rasters created with LiDAR tiles.

    :param tiles_folder: Folder with the different tiles
    :param variable: Variable to mask out
    :param field: Field of the grid to cut buffers out
    :param tile_spacing: Spacing of the tile
    :param lesser_lextent: If it will use a smaller extent or not.
    :return:
    """

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
    band_names    - optional. It gives a name to each band for easier identification. It has to have same length than data dimensions.
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
    """
    Create temporal band means from tiles base on their names

    :param path: Path to folder with image tiles
    :param out_path: path to output folder
    :param bandas: band to be considered
    :return:
    """
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
    """
    Create a temporal stack of the bands selected
    :param path: Path to folder with image tiles
    :param out_path: path to output folder
    :param bandas: band to be considered
    :return:
    """
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
    """
    Creates a grid based on the name of the folders
    :param folder_with_tiles: path to folder with tiles
    :param spacing: Spacing of the desired grid cells
    :param intile_length: an smallers spacing to be considered
    :param epsg: EPSG code to define projection
    :return:
    """
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
    """
    Creates naming based on the raster name and geometries: date_xmin-ymax_sentineltile_band
    :param raster_path: Path to raster file
    :param geometry: geom
    :return:
    """
    # xmin, xmax, ymin, ymax
    xmin, xmax, ymin, ymax = geometry.GetEnvelope()
    splitted = raster_path.split("/")
    len_splitted = len(splitted)
    name_tmp1 = splitted[len_splitted - 1]
    name = name_tmp1.split(".")[0]
    name_splitted = name.split("_")
    if len(name_splitted) < 3:
        outaname = f"{name}_{float(xmin)}-{float(ymax)}"
    else:
        sent_tile = name_splitted[0]
        band = name_splitted[2]
        date_tmp = name_splitted[1]
        date = date_tmp.split("T")[0]

        # outaname = f"{date}_{int(xmin)}_{int(ymax)}_{sent_tile}_{band}"
        outaname = f"{date}_{float(xmin)}-{float(ymax)}_{sent_tile}_{band}"
    return (outaname)


def layer_within_raster(raster_extent, layer_geom, lesser_lextent=False):
    """
    check if a layer is inside the raster
    :param raster_extent: extent of the raster
    :param layer_geom: layer geom
    :param lesser_lextent: If True a smaller extent is evaluated
    :return:
    """
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
    """
    Mosaic LiDAR images up to the image tiles spacing

    :param ImageTiles_layer: Grid created for tile Sentinel images
    :param rasterTiles_folder: LiDAR raster tiles
    :param output_folder: Path to output folder
    :param res: resolution desired for resampling)
    :param variable: Variable to be mosaiced
    :return:
    """
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
    """
    Prepare the stacks for segmentation
    :param crop_means: Desire means stack
    :param path_dsms: DSM raster files folder
    :param output: output
    :return:
    """
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



def merge_layers(inputFolder, outputFile, oformat="GPKG", epsg=25830, informat="ESRI Shapefile"):
    """
    Merge layers tiles together
    :param inputFolder: Input with layer tiles
    :param outputFile: Output to be created
    :param oformat: output file format
    :param epsg: epsg of output file
    :param informat: Input files format
    :return:
    """
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
    if type(inputFolder) is list:
        list_shps = inputFolder
    else:
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
    """
    Stack all the information needed for modelling
    :param lidar_tifs: Raster images derived from LiDAR pointclouds
    :param crop_stacks: Stacks with all the information wanted
    :param crop_stackForstats: Output stacks (tiles)
    :return:
    """

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
    if out_ds is None:
        print("output data source is None")
        return 1
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


def getting_stats(src_filename, layer_filename, output, statistics = "mean", name_band = None, all_touched = True,
                  all_features = False):
    """
    Generate spatial statistics of the segments

    :param src_filename:
    :param layer_filename:
    :param output:
    :param statistics: mean by default
    :param name_band:
    :param all_touched:
    :param all_features:
    :return:
    """

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
        if all_features:
            nField = feat.GetFieldCount()
            for fieldi in range(0, nField):
                fieldname = feat.GetFieldDefnRef(fieldi).GetName()
                # print(fieldname)
                fieldvalue = feat.GetField(fieldi)
                dictmp[fieldname] = fieldvalue

        for i in range(1, RasterCount + 1):
            banda = src_ds.GetRasterBand(i)
            if name_band is None:
                band_name = get_band_name(banda)
            else:
                band_name = name_band

            stats = rasterstats.zonal_stats(geomjson, src_filename, stats = statistics, band=i, all_touched = all_touched)

            for k, v in stats[0].items():
                field_name = f"{band_name}_{k}"
                dictmp[field_name] = v

        tmp = [dictmp]
        lista_features.append(tmp)
        lista_diccionarios.append(lista_features)

    create_layer(output, lista_diccionarios)

    del lds
    del src_ds



def getting_stats_toFolder(crop_stackForStats, segmented_tiles, segmented_stats, name_band = None,
                           statistics = "mean", all_touched = True, all_features = False):
    """
    Apply getting stats to a folder
    :param crop_stackForStats:
    :param segmented_tiles:
    :param segmented_stats:
    :param name_band:
    :param statistics:
    :param all_touched:
    :param all_features:
    :return:
    """
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

        if "gpkg" in segments:
            name, coordinates, tileext = segments.split("_")
            tile = tileext.split(".")[0]
            if ".tif" in crop_stackForStats:
                raster_stack = crop_stackForStats
                diccionario[segments] = raster_stack

            else:
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
        if ".tif" in crop_stackForStats:
            raster_stack_file = raster_stack
        else:
            raster_stack_file = f"{crop_stackForStats}/{v}"
        segmented_file = f"{segmented_tiles}/{k}"

        getting_stats(raster_stack_file, segmented_file, segStats_file, name_band = name_band, statistics = statistics,
                      all_touched = all_touched, all_features = all_features)



def joinTraining_with_Stats(forTraining, output_training, segmented_stats, training_field = "CODETYPE"):
    """
    Join training dataset with th Stats
    :param forTraining: Training dataset
    :param output_training:
    :param segmented_stats:
    :param training_field:
    :return:
    """

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

    # print(output_training)
    create_layer(output_training, lista_features)


def modela(archivoEntrenado, training_field):
    """
    Model creation
    :param archivoEntrenado: Trained file
    :param training_field:  Training target
    :return:
    """
    df = gpd.read_file(archivoEntrenado)
    columns = list(filter(lambda x: x not in [training_field, "geometry"], df.columns))
    y = df[training_field]
    X = df[columns]

    clasificador = RandomForestClassifier(n_estimators = 30)
    modelo = clasificador.fit(X, y)
    return(modelo)

def predecir(modelo, tile_for_predict, output, epsg = 25830, training_field = "CODETYPE"):
    """
    Prediction based on model
    :param modelo: Model
    :param tile_for_predict: Tile
    :param output:
    :param epsg:
    :param training_field:
    :return:
    """
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

    """
    Compute modelling and prediction at the same time
    :param archivoEntrenado:
    :param tile_for_predict:
    :param output:
    :param pca:
    :param epsg:
    :param training_field:
    :return:
    """

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
    """
    Apply prediction to folder

    :param modelo:
    :param tile_for_predict:
    :param output:
    :return:
    """
    def prediction_outputName(a):
        # print(a)
        tmp = a.split("/")
        tmp2 = tmp[len(tmp)-1]
        variable, coordinates, tilegpkg = tmp2.split("_")
        out = f"Pred_{coordinates}_{tilegpkg}"
        return(out)


    lista_SegStats = list(map(lambda x: f"{tile_for_predict}/{x}", os.listdir(tile_for_predict)))
    lista_predOut = list(map(lambda x: f"{output}/{prediction_outputName(x)}", lista_SegStats))

    aaaaout = list(map(lambda x,y,z: predecir(x, y, z),
                       [modelo]*len(lista_SegStats),
                       lista_SegStats,
                       lista_predOut
                      ))


def forest_fuel(input_layer, output_layer, training_field="CODETYPE", driver_name="GPKG",
                epsg=25830, geom_type=ogr.wkbPolygon, data_type=ogr.OFTReal
                ):

    """
    Forest fuel classification

    :param input_layer: Land cover map (with forests)
    :param output_layer:
    :param training_field:
    :param driver_name:
    :param epsg:
    :param geom_type:
    :param data_type:
    :return:
    """

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

        nfields = feature.GetFieldCount()
        for fieldi in range(nfields):
            nameField = feature.GetFieldDefnRef(fieldi).GetName()
            if nameField == training_field:
                valueField = feature.GetField(fieldi)
                diccionario[training_field] = valueField
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

    """
    Application of forest_fuel to folders
    :param input_folder:
    :param output_folder:
    :return:
    """

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
    """
    Pendiente computation from DEM
    :param src: DEM
    :param dst: output
    :return:
    """
    try:
        fichero=richdem.LoadGDAL(src)
    except:
        print("No existe el archivo")
        return False
    try:
        #se calcula la pendiente en radianes
        pendiente=richdem.TerrainAttribute(fichero, attrib='slope_percentage')
    except:
        print("El archivo no tiene un formato válido")
        return False
    if not dst:
        dst=src.replace(".tif","_pendiente.tif")
    richdem.SaveGDAL(dst,pendiente)
    print("Se ha calculado la pendiente: "+dst)
    return True


def orientacion(src,dst=None):
    """
    Aspect computation from DEM
    :param src: DEM
    :param dst: output
    :return:
    """
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
    """
    Apply slope to folder
    :param las_tiles:
    :return:
    """
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
    """
    Apply aspect to folders
    :param las_tiles:
    :return:
    """
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
    """
    Reclass aspect into North (4), East (3), South (2) and West (1)
    :param inputfile:
    :param outputfile:
    :return:
    """
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
    """
    Apply reclassification aspect to folder
    :param lidar_tifs:
    :param variable_folder:
    :return:
    """
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


def read_temperaturas(database, table, periodo, out_file,
                      user="postgres", password="postgres",
                      host = "localhost", port = "5432",
                      layer=None, extent=None, srs=25830,
                      temperatura_variable="tm_mes", temperatura_name="tmed",
                      rango_de_años=15, outdriver="GPKG"):

    """
    Read temperatures from a postgis database
    :param database: database name
    :param table: table name
    :param periodo: Span of years to create the mean
    :param out_file: Output file
    :param user: user
    :param password: password
    :param host: host
    :param port: port
    :param layer: subset with layer extent
    :param extent: subset with extent
    :param srs: Define an srs
    :param temperatura_variable: name of the variable
    :param temperatura_name: name of out variable
    :param rango_de_años: Minimum number of years to be considered
    :param outdriver: format of the output file
    :return:
    """

    if layer is not None:
        extent = get_layerextent(layer)

    if extent is not None:
        sql_envelope = f"{extent}, {srs}"

    # Conexión con la base de datos
    connection = psycopg2.connect(database=database,
                                  user=user, password=password,
                                  port = port, host = host
                                  )
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

    """
    Get extent from a layer with R
    :param layer:
    :param dist:
    :return:
    """

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


def TempRegKrig(points, var, formula, mdt, lat, dist, output, res = 1000, epsg = 25830):

    """
    Compute regression kriging with R

    :param points: Temperature points
    :param var: Temperature variable name
    :param formula: Formula for regression kiring as character
    :param mdt: mdt raster path
    :param lat: latitude raster path
    :param dist: distance raster path
    :param output: output
    :param res: resolution
    :param epsg: espeg code to define projection
    :return:
    """

    utils = importr("utils")
    utils.chooseCRANmirror(ind=1)
    base = importr("base")
    raster = importr("raster")

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
    rekriged = doReKrig(points = points, var = var, formula = formula, listrasters = rasterlist, res = res, epsg = epsg)

    raster.writeRaster(rekriged, output, overwrite = True)


def read_humedad(database, table, periodo, out_file,
                 user="postgres", password="postgres",
                 host = "localhost", port = "5432",
                 humidity_name = "rehumidity",
                 layer=None, extent=None, srs=25830,
                 rango_de_años=15, outdriver="GPKG"):
    """
    Read humidty points from a postgis database
    :param database:
    :param table:
    :param periodo:
    :param out_file:
    :param user:
    :param password:
    :param host:
    :param port:
    :param humidity_name: name of the variable
    :param layer:
    :param extent:
    :param srs:
    :param rango_de_años: Minimum span of years to be considered
    :param outdriver:
    :return:
    """

    if layer is not None:
        extent = get_layerextent(layer)

    if extent is not None:
        sql_envelope = f"{extent}, {srs}"

    # Conexión con la base de datos
    connection = psycopg2.connect(database=database,
                                  user=user, password=password,
                                  port = port, host = host
                                  )
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
                                year between {p1} and  {p2}
                        group by
                                indicativo
                        )
                    select
                        a.indicativo,
                        avg(a.humidity)/10 as {humidity_name},
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
                        st_intersects(a.geom, st_MakeEnvelope({sql_envelope}))
                    group by
                        a.indicativo,
                        a.geom;"""
    cursor.execute(sql_query)

    # Guardando puntos de temperatura en una capa.
    lista = []
    for indicativo, temperatura, geom in cursor:
        data = {'indicativo': indicativo, f'{humidity_name}': temperatura, 'geometry': shp.wkt.loads(geom)}
        lista.append(data)

    gdf = gpd.GeoDataFrame(lista, crs=f'epsg:{srs}').set_index('indicativo')
    gdf[f'{humidity_name}'] = gdf[f'{humidity_name}'].astype("float")
    gdf.to_file(out_file, driver=outdriver)
    print("Archivo creado")


def HumOKrig(points, var, output, res = 1000, epsg = 25830):

    """
    Compute simple kriging to humidity interpolation
    :param points: points layer path
    :param var: humidity variable name
    :param output: output path
    :param res: resolution
    :param epsg: EPSG code to define projection
    :return:
    """

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
    # rasterlist = base.list(mdt = mdt, lat = lat, dist = dist)
    # print(2)
    # print(rasterlist)
    doKrig = robjects.globalenv['doKrig']
    kriged = doKrig(points = points, var = var, res = res, epsg = epsg)
    raster = importr("raster")
    raster.writeRaster(kriged, output, overwrite = True)


# Reference fuel moisture
def rfm(temperature, reH):
    """
    Compute Reference Fuel Moisture for Ignition Probability Computation
    :param temperature:
    :param reH:
    :return:
    """
    if temperature <= -1.5:
        i = 0
    if temperature > -1.5 and temperature <= 10:
        i = 1
    if temperature > 10 and temperature <= 20:
        i = 2
    if temperature > 20 and temperature <= 31:
        i = 3
    if temperature > 31 and temperature <= 42:
        i = 4
    if temperature > 42:
        i = 5

    if reH < 5:
        j = 0
    if reH >= 5 and reH < 10:
        j = 1
    if reH >= 10 and reH < 15:
        j = 2
    if reH >= 15 and reH < 20:
        j = 3
    if reH >= 20 and reH < 25:
        j = 4
    if reH >= 25 and reH < 30:
        j = 5
    if reH >= 30 and reH < 35:
        j = 6
    if reH >= 35 and reH < 40:
        j = 7
    if reH >= 40 and reH < 45:
        j = 8
    if reH >= 45 and reH < 50:
        j = 9
    if reH >= 50 and reH < 55:
        j = 10
    if reH >= 55 and reH < 60:
        j = 11
    if reH >= 60 and reH < 65:
        j = 12
    if reH >= 65 and reH < 70:
        j = 13
    if reH >= 70 and reH < 75:
        j = 14
    if reH >= 75 and reH < 80:
        j = 15
    if reH >= 80 and reH < 85:
        j = 16
    if reH >= 85 and reH < 90:
        j = 17
    if reH >= 90 and reH < 95:
        j = 18
    if reH >= 95 and reH < 100:
        j = 19
    if reH >= 100:
        j = 20

    RFM = np.array([[1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 8, 9, 9, 10, 11, 12, 12, 13, 13, 14],
                    [1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 7, 8, 9, 9, 10, 10, 11, 12, 13, 13, 13],
                    [1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 11, 12, 12, 12, 13],
                    [1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 12, 12, 13],
                    [1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 12, 12, 13],
                    [1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 12, 12, 12]]
                   )

    out = RFM[i, j]
    return (out)


def dfm_corrections(slope, aspect, shading, month_number):
    """
    Compute correction for rfm

    :param slope:
    :param aspect:
    :param shading: Shading value (1 for tree forests, 0 for everything else)
    :param month_number: Number of the month
    :return:
    """

    # Slope is i
    # Aspect is j
    # Shading is z
    if month_number in [5, 6, 7]:
        tmp1 = np.array([[0, 0, 0, 0], [1, 0, 1, 1]])
        tmp2 = np.array([[3, 3, 3, 3], [3, 3, 3, 3]])
        DFMCorrections = np.stack([tmp1, tmp2], axis=2)

    if month_number in [2, 3, 4, 8, 9, 10]:
        tmp1 = np.array([[1, 1, 1, 1], [3, 1, 1, 2]])
        tmp2 = np.array([[4, 4, 4, 4], [4, 4, 4, 4]])
        DFMCorrections = np.stack([tmp1, tmp2], axis=2)

    if month_number in [11, 12, 1]:
        tmp1 = np.array([[3, 3, 3, 3], [5, 2, 1, 4]])
        tmp2 = np.array([[5, 5, 5, 5], [5, 5, 5, 5]])
        DFMCorrections = np.stack([tmp1, tmp2], axis=2)

    if slope <= 30: i = 0
    if slope > 30: i = 1

    if np.round(aspect) == 1.0: j = 3
    if np.round(aspect) == 2.0: j = 2
    if np.round(aspect) == 3.0: j = 1
    if np.round(aspect) == 4.0: j = 0

    if shading == 0: z = 0
    if shading == 1: z = 1

    out = DFMCorrections[i, j, z]
    return (out)


def pig(temperature, fdmp, shading):
    """
    Compute Ignition Probability from forest_fuel type
    :param temperature: temperature value
    :param fdmp: rfm + dfm_correction
    :param shading: Shading value (1 for tree forests, 0 for everything else)
    :return:
    """

    # Temperature == i
    if temperature >= 43:
        i = 0
    if temperature >= 37 and temperature < 43:
        i = 1
    if temperature >= 32 and temperature < 37:
        i = 2
    if temperature >= 26 and temperature < 32:
        i = 3
    if temperature >= 21 and temperature < 26:
        i = 4
    if temperature >= 15 and temperature < 21:
        i = 5
    if temperature >= 10 and temperature < 15:
        i = 6
    if temperature >= 4 and temperature < 10:
        i = 7
    if temperature < 4:
        i = 8

    # FDFM == j
    j = fdmp - 2  # It starts on 2 and go up to 17, but it will index 0 based.

    # shading == z
    if shading == 0: z = 0
    if shading == 1: z = 1

    PIG_0 = np.array([[100, 100, 80, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 20, 10],
                      [100, 90, 80, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10],
                      [100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 30, 20, 20, 20, 10, 10],
                      [100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10],
                      [100, 80, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10],
                      [90, 80, 70, 60, 50, 50, 40, 30, 30, 20, 20, 20, 20, 10, 10, 10],
                      [90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10],
                      [90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10],
                      [80, 70, 60, 50, 50, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10],
                      ])
    PIG_1 = np.array([[100, 90, 80, 70, 60, 50, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10],
                      [100, 90, 80, 70, 60, 50, 50, 40, 30, 30, 30, 20, 20, 20, 10, 10],
                      [100, 90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10],
                      [100, 80, 70, 60, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10],
                      [90, 80, 700, 60, 50, 50, 40, 30, 30, 30, 20, 20, 20, 10, 10, 10],
                      [90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10],
                      [90, 80, 70, 60, 50, 40, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10],
                      [90, 80, 60, 50, 50, 40, 30, 30, 30, 20, 20, 20, 10, 10, 10, 10],
                      [80, 80, 60, 50, 50, 40, 30, 30, 20, 20, 20, 10, 10, 10, 10, 10]
                      ])
    PIG = np.stack([PIG_0, PIG_1], axis=2)
    out = PIG[i, j, z]
    return (out)


def ignition_probability(file, output, month_number):
    """
    Compute ignition probability
    :param file: layer file
    :param output: output path
    :param month_number: Month number
    :return:
    """

    print(file)
    in_ds = ogr.Open(file)
    in_layer = in_ds.GetLayer()

    list_dictionaries = []
    for feat in in_layer:

        diccionario = {}

        geom = feat.GetGeometryRef().ExportToWkt()

        nfields = feat.GetFieldCount()
        for n in range(nfields):
            field_name = feat.GetFieldDefnRef(n).GetName()
            if field_name == "T_mean": temperatura = feat.GetField(n)
            if field_name == "reH_mean": reH = feat.GetField(n)
            if field_name == "fuel_type":
                value = feat.GetField(n)
                if value in [8, 9]:
                    shading = 1
                else:
                    shading = 0

            if field_name == "slope_mean":
                slope = feat.GetField(n)
                if slope is None:
                    # This has to be improved!
                    slope = 0
            if field_name == "aspect_mean": aspect = feat.GetField(n)

        tmp_rfm = rfm(temperatura, reH)
        tmp_dfm = dfm_corrections(slope, aspect, shading, month_number)
        fdmp = tmp_rfm + tmp_dfm
        ignition = pig(temperatura, fdmp, shading)
        if value == 0.0: ignition = 0

        diccionario["pig"] = int(ignition)
        list_dictionaries.append([geom, [diccionario]])

    in_ds = None

    # print(list_dictionaries)
    create_layer(output, list_dictionaries, driver_name = "GPKG", data_type = ogr.OFTInteger)


def ignition_probability_toFolders(folderin, folderout, month_number):
    """
    Apply ignition probability to folders
    :param folderin:
    :param folderout:
    :param month_number:
    :return:
    """
    def getouts(a):
        b = a.split("/")
        b = b[len(b) - 1]
        name, coordinates, tilegpkg = b.split("_")
        out = f"pig_{coordinates}_{tilegpkg}"
        return (out)

    listins = list(map(lambda x: f"{folderin}/{x}", os.listdir(folderin)))
    listouts = list(map(lambda x: f"{folderout}/{getouts(x)}", listins))
    ooout = list(map(lambda x, y, z: ignition_probability(x, y, z), listins, listouts, [month_number] * len(listouts)))
    return (ooout)


def fireRisk(in_to, in_from, output, buffer_distance=20000.0):
    """
    Compute fire risk
    :param in_to: Probability ignition layer
    :param in_from: Auxiliary data layer
    :param output: output
    :param buffer_distance: By default 20000.0
    :return:
    """
    pig_ds = ogr.Open(in_to)
    leisure_ds = ogr.Open(in_from)
    pig_layer = pig_ds.GetLayer()
    leisure_layer = leisure_ds.GetLayer()

    trigger = 0
    lista_features = []
    ultra_diccionario = {}
    for leisure_feat in leisure_layer:
        leisure_geom = leisure_feat.GetGeometryRef()

        for pig_feat in pig_layer:
            diccionario = {}
            #         print(diccionario)
            pig_geom = pig_feat.GetGeometryRef()
            pig_geomWkt = pig_geom.ExportToWkt()

            nfields = pig_feat.GetFieldCount()
            for i in range(nfields):
                field_name = pig_feat.GetFieldDefnRef(i).GetName()
                field_value = pig_feat.GetField(i)
                if field_name == "pig":
                    pig = field_value

                diccionario[field_name] = field_value

            if pig_geomWkt in ultra_diccionario:
                if "fireRisk" in ultra_diccionario[pig_geomWkt][0]:
                    trigger = 1
                    old_value = float(ultra_diccionario[pig_geomWkt][0]["fireRisk"])

            if leisure_geom.Intersects(pig_geom.Centroid().Buffer(buffer_distance)):
                if leisure_geom.Contains(pig_geom.Centroid()):
                    distancia = 0.0
                else:
                    distancia = float(leisure_geom.Distance(pig_geom.Centroid()))

            else:

                distancia = buffer_distance

            fireRisk = ((buffer_distance - distancia) / buffer_distance) * pig
            #             print(distancia, pig, fireRisk)
            if trigger == 1:
                if old_value > fireRisk:
                    fireRisk = old_value
                trigger = 0

            diccionario["fireRisk"] = fireRisk
            #         print(old_value, distancia, trigger, old_value < distancia)
            del distancia

            ultra_diccionario[pig_geomWkt] = [diccionario]

    leisure_ds = None
    pig_ds = None

    for k, v in ultra_diccionario.items():
        lista_features.append([k, v])

    create_layer(output, lista_features)


def fire_risk_toFolders(folderin, folderout, auxiliary_layer, buffer_distance = 20000.0):
    """
    Compute fire risk to folders
    :param folderin:
    :param folderout:
    :param auxiliary_layer:
    :param buffer_distance:
    :return:
    """

    def getouts(a):
        b = a.split("/")
        b = b[len(b) - 1]
        name, coordinates, tilegpkg = b.split("_")
        out = f"fireRisk_{coordinates}_{tilegpkg}"
        return (out)

    listins = list(map(lambda x: f"{folderin}/{x}", os.listdir(folderin)))
    listouts = list(map(lambda x: f"{folderout}/{getouts(x)}", listins))
    ooout = list(map(lambda x, y, z, a: fireRisk(x, z, y, a),
                     listins,
                     listouts,
                     [auxiliary_layer] * len(listouts),
                     [buffer_distance] * len(listouts)
                    ))
    return (ooout)


if __name__=="__main__":
    create_grid_from_name("tmp/test/tiles", spacing = 250, intile_length = 2000)

    # layer = "input/caba_limites.gpkg"
    # createGrid(path_to_layer = layer, spacing=500, epsg=25830, output = "tmp/layers/caba_500m_tiles.gpkg")

    # imagen = "tmp/images_test/S2A_MSIL1C_20180427T110621_N0206_R137_T30SUJ_20180427T133034/A014863/T30SUJ_20180427T110621_B04.jp2"
    # a = reproject(image = imagen, output_folder = "tmp/reprojected_images", epsg_to=25830)
