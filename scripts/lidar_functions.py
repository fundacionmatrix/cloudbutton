import pdal
import json
import os
from osgeo import gdal
from osgeo import osr
from scipy import ndimage
import numpy as np

def get_lidar_subset_byextent(path_to_lidar_folder, layerextent, output_folder):
    """
    Subset lidar tiles by extent
    :param path_to_lidar_folder: Folder with Lidar tiles from PNOA
    :param layerextent: Layer extent to subset with
    :param output_folder: output folder
    :return:
    """

    lxmin, lxmax, lymin, lymax = layerextent.split(",")
    lxmin = float(lxmin) - 100
    lxmax = float(lxmax) + 100
    lymax = float(lymax) + 100
    lymin = float(lymin) - 100


    listafiles = []
    listlaz = os.listdir(path_to_lidar_folder)
    for laz in listlaz:
        if "laz" in laz or "las" in laz:

            pnoa, year, lote, comunidad, coordinates, options = laz.split("_")
            if "CIR" in options[8:11]:
                xmin, ymax = coordinates.split("-")
                xmin = int(xmin) * 1000.0
                ymax = int(ymax) * 1000.0
                xmax = xmin + 2000
                ymin = ymax - 2000
                # print(xmin, ymin, xmax, ymax)

                i = 0
                if lxmin <= xmin:  # 1. upper left corner
                    i += 1
                if lymax >= ymax:  # 2. upper right corner
                    i += 1
                if lxmax >= xmax:  # 3. lower right corner
                    i += 1
                if lymin <= ymin:  # 4. lower left corner
                    i += 1

                if i == 4:
                    listafiles.append(laz)

    print(len(listafiles))
    for file in listafiles:
        inputfile = f"{path_to_lidar_folder}/{file}"
        outputfile = f"{output_folder}/{file}"
        os.rename(inputfile, outputfile)
    # next step is erase useless lidar


def tile_lidar(folder_with_las, spacing, output_folder, buffer_distance):

    """
Autor = Juanma Cintas
Fecha = 07/03/2019
email = juanmanuel.cintas@fundacionmatrix.es

Descripción:
Tile using filters.splitter.  ERROR: It can create las files iteratively.

La documentación de la libería pdal se puede encontrar en la siguiente
dirección: https://pdal.io/index.html


    """

    creating_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": f"{folder_with_las}/*",
                "spatialreference": "EPSG:25830"
            },
            {
                "type": "filters.splitter",
                "origin_x": "348500",
                "origin_y": "4391500",
                "length": f"{spacing}",
                "buffer": f"{buffer_distance}"
            },
            {
                "type": "writers.las",
                "a_srs": "EPSG:25830",
                "compression": "laszip",
                "filename": f"{output_folder}/test_#.laz"
            }
        ]
    }

    consulta = json.dumps(creating_json, indent=4)
    print(consulta)

    pipeline = pdal.Pipeline(consulta)
    pipeline.validate()  # Check if json options are good
    pipeline.loglevel = 8
    count = pipeline.execute()
    print(count)


def tile_lidar_bash(folder_with_las, output_folder, spacing, buffer=None):
    """
    Tile lidar tiles in smaller tiles with bash
    :param folder_with_las: folder with Lidar tiles
    :param output_folder: output folder
    :param spacing: spacing of the new tiles
    :param buffer: extra distance to be considered
    :return:
    """

    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)

    for las in os.listdir(folder_with_las):
        if ".laz" in las or ".las" in las:
            pnoa, year, lote, comunidad, coordinates, options = las.split("_")
            xmin, ymax = coordinates.split("-")
            xmin = int(xmin) * 1000.0
            ymax = int(ymax) * 1000.0
            coordinates = f"{xmin}-{ymax}"

            if os.path.exists(f"{output_folder}/{coordinates}") is False:
                os.mkdir(f"{output_folder}/{coordinates}")
            print(f"Doing {coordinates}")
            if buffer is None:
                #                 print(f"pdal tile {folder_with_las}/{las} {output_folder}/{folder_with_las}/{coordinates}/out_#.laz --length {spacing} --origin_x {xmin} --origin_y {ymax}")
                os.system(
                    f"pdal tile {folder_with_las}/{las} {output_folder}/{coordinates}/out_#.laz --length {spacing} --origin_x {xmin} --origin_y {ymax}")

            else:
                #                 print(f"pdal tile {folder_with_las}/{las} {output_folder}/{folder_with_las}/{coordinates}/out_#.laz --length {spacing} --buffer {buffer} --origin_x {xmin} --origin_y {ymax}")
                os.system(
                    f"pdal tile {folder_with_las}/{las} {output_folder}/{coordinates}/out_#.laz --length {spacing} --buffer {buffer} --origin_x {xmin} --origin_y {ymax}")


def decimate_points(lasfile, outputlas):
    """
Autor = Juanma Cintas
Fecha = 21/02/2019
email = juanmanuel.cintas@fundacionmatrix.es

Descripción:
Clasifica los puntos de solape como ruido, de esta forma es más sencillo de ignorarlos en los posteriores procesos.

La documentación de la libería pdal se puede encontrar en la siguiente
dirección: https://pdal.io/index.html
"""
    creating_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": f"{lasfile}"
            },
            {
                # Filter assigning points of class overlay (12) to class noise (7)
                "type": "filters.decimation",
                "step": "10"
            },
            {
                "type": "writers.las",
                "compression": "laszip",
                "filename": f"{outputlas}"
            }
        ]
    }

    consulta = json.dumps(creating_json, indent=4)
    #     print(consulta)
    try:
        pipeline = pdal.Pipeline(consulta)
        pipeline.validate()  # Check if json options are good
        pipeline.loglevel = 8
        count = pipeline.execute()

        return(0)

    except ValueError:
        print(ValueError)
        return(1)

#     print(count)

def decimate_folder(input_folder, output_folder):
    """
    Decimate lidar tiles inside a folder
    :param input_folder:
    :param output_folder:
    :return:
    """
    os.makedirs(output_folder, exist_ok=True)
    lasinfile = os.listdir(input_folder)
    dict_inout = list(map(lambda x: {f"{input_folder}/{x}": f"{output_folder}/dec_{x[7:]}"},
                          lasinfile))

    try:
        tmp = list(map(lambda x: decimate_points(lasfile=list(x.keys())[0],
                                                 outputlas=list(x.values())[0]
                                                 ),
                       dict_inout
                       ))
        if tmp[0] == 0:
            return (0)
        else:
            return (2)
    except ValueError:
        print(ValueError)
        return (3)


def decimate_all_folders(las_tiles, output_folder, input_folder):
    """
    Decimate all the folders with tiles
    :param las_tiles:
    :param output_folder:
    :param input_folder:
    :return:
    """
    coordinates_folders = os.listdir(las_tiles)
    dict_inout = list(map(lambda x, y, z: {f"{las_tiles}/{x}/{y}": f"{las_tiles}/{x}/{z}"},
                          coordinates_folders,
                          [input_folder] * len(coordinates_folders),
                          [output_folder] * len(coordinates_folders)
                          ))

    try:
        tmp = list(map(lambda x: decimate_folder(input_folder=list(x.keys())[0],
                                                 output_folder=list(x.values())[0]),
                       dict_inout
                       ))

        if tmp[0] == 0:
            return (0)
        else:
            return (4)
    except ValueError:
        print(ValueError)
    except Exception:
        print(Exception.name)
        return (5)


def remove_overlay_points(lasfile, outputlas):
    """
Autor = Juanma Cintas
Fecha = 21/02/2019
email = juanmanuel.cintas@fundacionmatrix.es

Descripción:
Clasifica los puntos de solape como ruido, de esta forma es más sencillo de ignorarlos en los posteriores procesos.

La documentación de la libería pdal se puede encontrar en la siguiente
dirección: https://pdal.io/index.html
"""
    creating_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": f"{lasfile}"
            },
            {
                # Filter assigning points of class overlay (12) to class noise (7)
                "type": "filters.assign",
                "assignment": "Classification[12:12]=7"
            },
            {
                "type": "writers.las",
                "compression": "laszip",
                "filename": f"{outputlas}"
            }
        ]
    }

    consulta = json.dumps(creating_json, indent=4)
    #     print(consulta)

    pipeline = pdal.Pipeline(consulta)
    pipeline.validate()  # Check if json options are good
    pipeline.loglevel = 8
    count = pipeline.execute()


#     print(count)

def remove_overlay_points_folder(tiles_folder, output_folder):
    """
    Remove overlay points from Lidar tiles inside a folder
    :param tiles_folder:
    :param output_folder:
    :return:
    """
    for folder in os.listdir(tiles_folder):
        lasfolder = f"{tiles_folder}/{folder}"
        if os.path.exists(f"{lasfolder}/{output_folder}") is False:
            os.mkdir(f"{lasfolder}/{output_folder}")

        print(lasfolder)
        for las in os.listdir(lasfolder):

            ## AQUI LA PARALELIZACIÓN ##
            if "las" in las or "laz" in las:
                remove_overlay_points(f"{lasfolder}/{las}", f"{lasfolder}/{output_folder}/noOver_{las}")


def remove_noise(lasfile, outputlas):
    """
    Remove noise points
    :param lasfile:
    :param outputlas:
    :return:
    """
    creating_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": f"{lasfile}"
            },
            {
                # Creates a window to find outliers. If they are found they are classified as noise (7).
                "type": "filters.outlier",
                "method": "statistical",
                "multiplier": 1.5,
                "mean_k": 8
            },
            {
                "type": "writers.las",
                "compression": "laszip",
                "filename": f"{outputlas}"
            }
        ]
    }

    consulta = json.dumps(creating_json, indent=4)
    #     print(consulta)

    pipeline = pdal.Pipeline(consulta)
    pipeline.validate()  # Check if json options are good
    pipeline.loglevel = 8
    count = pipeline.execute()


#     print(count)

def remove_noise_folder(tiles_folder, output_folder):
    """
    remove noise points from Lidar tiles inside a folder
    :param tiles_folder:
    :param output_folder:
    :return:
    """
    for folder in os.listdir(tiles_folder):
        lasfolder = f"{tiles_folder}/{folder}" # in coordinate tiles folder
        if os.path.exists(f"{lasfolder}/{output_folder}") is False:
            os.mkdir(f"{lasfolder}/{output_folder}")

        clean_folder = f"{lasfolder}/{output_folder}"
        print(lasfolder)
        for f in os.listdir(lasfolder):
            path = f"{lasfolder}/{f}"

            # It should be careful, since it is checking for folder. If more than one folder exists... then it could replicate values.
            if os.path.isdir(path):

                ## AQUI LA PARALELIZACIÓN ##
                for las in os.listdir(path):
                    tmp = las.split("_")[0]
                    out_name = f"clean_{las[len(tmp) + 1:]}"
                    remove_noise(f"{path}/{las}", f"{clean_folder}/{out_name}")


def DEMonizator(lasfile, outputfile, resolution=1000):
    """
    Creates a DEM from a las file
    :param lasfile:
    :param outputfile:
    :param resolution:
    :return:
    """
    creating_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": f"{lasfile}",
                "spatialreference": "EPSG:25830"
            },
            {
                "type": "filters.range",
                "limits": "Classification[2:2]"
            },
            {
                "type": "writers.gdal",
                "gdaldriver": "GTiff",
                "nodata": "-9999",
                "output_type": "idw",
                "resolution": f"{resolution}",
                "filename": f"{outputfile}"
            }

        ]
    }

    consulta = json.dumps(creating_json, indent=4)
    #         print(consulta)

    pipeline = pdal.Pipeline(consulta)
    pipeline.validate()  # Check if json options are good
    pipeline.loglevel = 8
    count = pipeline.execute()


#         print(count)

def CHMonizator(lasfile, outputfile, resolution=5.0, type=None):
    """
Autor = Juanma Cintas Rodríguez
Fecha = 21/03/2019
email = juanmanuel.cintas@fundacionmatrix.es

Descripción:
Crea un Digital Height Model (por defecto), un Canopy Height Model (CHM), un Tree Canopy Height (TCH),
un Shrub Canopy Height (SCH) a partir de una nube de`puntos LiDAR.

Argumentos:
@lasfile = fichero de la nube de puntos LiDAR con extension las o laz.
@outputfile = fichero tif de devuelto.
@resolution = Resolución del fichero de salida en la unidades del sistema de referencia.
@type = Tipo de raster a producir. Puede dejarse en blanco o tomar los valores TCH, SCH y CHM. En el caso de
dejarse en blanco, un DHM será producido.


La documentación de la libería pdal se puede encontrar en la siguiente
dirección: https://pdal.io/index.html
    """

    if type == "TCH":
        kindoftype = {"type": "filters.range", "limits": "Classification[2:2],Classification[5:5]"}
    elif type == "SCH":
        kindoftype = {"type": "filters.range", "limits": "Classification[2:2],Classification[4:4]"}
    elif type == "CHM":
        kindoftype = {"type": "filters.range", "limits": "Classification[2:2],Classification[3:5]"}
    else:
        kindoftype = {"type": "filters.info"}

    creating_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": f"{lasfile}",
                "spatialreference": "EPSG:25830"
            },
            {
                "type": "filters.hag"
            },
            kindoftype,
            {
                "type": "filters.ferry",
                "dimensions": "HeightAboveGround=Z"
            },
            {
                "type": "writers.gdal",
                "gdaldriver": "GTiff",
                "nodata": "-9999",
                "output_type": "idw",
                "resolution": f"{resolution}",
                "filename": f"{outputfile}"
            }
        ]
    }

    consulta = json.dumps(creating_json, indent=4)
    #         print(consulta)

    pipeline = pdal.Pipeline(consulta)
    pipeline.validate()  # Check if json options are good
    pipeline.loglevel = 8
    count = pipeline.execute()


#         print(count)

def DSMonizator(lasfile, outputfile, resolution=5):
    """
Autor = Juanma Cintas Rodríguez
Fecha = 21/03/2019
email = juanmanuel.cintas@fundacionmatrix.es

Descripción:
Produce un Modelo Digital de Superficie (DSM por sus siglas en inglés).

Argumentos:
@lasfile = fichero de la nube de puntos LiDAR con extension las o laz.
@outputfile = fichero tif de devuelto.
@resolution = Resolución del fichero de salida en la unidades del sistema de referencia.


La documentación de la libería pdal se puede encontrar en la siguiente
dirección: https://pdal.io/index.html

    """

    creating_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": f"{lasfile}",
                "spatialreference": "EPSG:25830"
            },
            {
                "type": "filters.returns",
                "groups": "first, only"
            },
            {
                "type": "writers.gdal",
                "gdaldriver": "GTiff",
                "nodata": "-9999",
                "output_type": "idw",
                "resolution": f"{resolution}",
                "filename": f"{outputfile}"
            }

        ]
    }
    consulta = json.dumps(creating_json, indent=4)
    #         print(consulta)
    pipeline = pdal.Pipeline(consulta)
    pipeline.validate()  # Check if json options are good
    pipeline.loglevel = 8
    count = pipeline.execute()
    #         print(count)


def surfaces_to_folder(folder, surface="DEM", name_clean="clean", resolution=5):
    """
    Creates different surface raster from LiDAR tiles inside a folder
    :param folder:
    :param surface:
    :param name_clean:
    :param resolution:
    :return:
    """
    for f in os.listdir(folder):
        lasfolder = f"{folder}/{f}/{name_clean}"
        output_folder = f"{folder}/{f}/{surface}"

        # Creating output folder
        if os.path.exists(output_folder) is False:
            os.mkdir(output_folder)

        ## Aplying functions to las
        for las in os.listdir(lasfolder):
            inlas = f"{lasfolder}/{las}"
            tmp = las.split(".")[0]
            out_tmp = f"{tmp[len(name_clean) + 1:]}.tif"
            out_tif = f"{output_folder}/{surface}_{out_tmp}"


            if surface == "DEM":
                try:
                    DEMonizator(inlas, out_tif, resolution=resolution)
                except:
                    print("Something happened")

            if surface == "CHM":
                try:
                    CHMonizator(inlas, out_tif, resolution=resolution, type= surface)
                except:
                    print("Something happened")

            if surface == "TCH":
                try:
                    CHMonizator(inlas, out_tif, resolution=resolution, type= surface)
                except:
                    print("Something happened")

            if surface == "SCH":
                try:
                    CHMonizator(inlas, out_tif, resolution=resolution, type= surface)
                except:
                    print("Something happened")

            if surface == "DSM":
                try:
                    DSMonizator(inlas, out_tif, resolution=resolution)
                except:
                    print("Something happened")



def FCC(input, output, window = 20, breakpoint = 0.01):
    """
    Autor = Juanma Cintas Rodríguez
    Fecha = 21/03/2019
    email = juanmanuel.cintas@fundacionmatrix.es

    Descripción:
    Crea una proporción de una clase respecto al total de puntos, basado en el cálculo de Fracción Cabida Cubierta (FCC)
    presentado por García et al (2011) (DOI:10.10016/j.jag.2011.03.006). Dependerá del raster introducido en la función
    si es calculada la FCC, la TCC o la SCC.

    Argumentos:
    @input = Raster del cual será computada la relación (FCC, TCC o SCC)
    @output = Nombre del raster a generar.
    @winodw = Tamaño de la ventana móvil usada para calcular la relación. A cosiderar que, cuanto menor sea la resolución,
    mayor deberá ser la ventana móvil para conseguir buenos resultados. Una medida puede ser 4 veces la resolución del raster.
    @breakpoint = Valor a partir del cual se consideraran las celdas del raster como vegetación.


    La documentación de la libería pdal se puede encontrar en la siguiente
    dirección: https://pdal.io/index.html

    Documentación acerca de gdal/ogr y su API de python puede ser encontrada en el siguiente enlace:
    https://gdal.org

    Documentación acerca de la librería scipy puede ser encontrada en el siguiente enlace:
    https://www.scipy.org/

    Documentación acerca de la libreri numpy puede ser encontrada en el siguiente enlace:
    http://www.numpy.org/
    """

    def compute_fraction(array):
        nveg = np.sum(array == 1)
        total = len(array)
        out = (nveg / total) * 100
        return (out)

    # Reading data needed
    tch = input
    in_ds = gdal.Open(tch)
    rows = in_ds.RasterYSize
    cols = in_ds.RasterXSize
    in_band = in_ds.GetRasterBand(1)
    data = in_band.ReadAsArray(0, 0, cols, rows).astype(np.float)

    # Reclassifying data
    data[data > breakpoint] = 1
    data[data <= breakpoint] = 0

    # Computing fraction on the whole raster through a moving window.
    TCC = ndimage.generic_filter(data, compute_fraction, size = window)

    # Setting output
    gtiff_driver = gdal.GetDriverByName("GTiff")
    out_ds = gtiff_driver.Create(output, cols, rows, 1, in_band.DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())

    # Writing data
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(TCC)
    # out_ds.BuildOverviews("Average", [2, 4, 8, 16, 32])

    out_ds.FlushCache()

    del in_ds, out_ds



def FCC_to_folders(las_tiles, covers):
    """
    Cover Fracition Percentage from LiDAR tiles inside a folder
    :param las_tiles:
    :param covers:
    :return:
    """
    def assigning_input_folder(a):
        if a == "TCC":
            out = "TCH"
        if a == "SCC":
            out = "SCH"
        if a == "FCC":
            out = "CHM"
        return (out)

    def FCC_to_folder(coordinates_folder, cover):
        infol = assigning_input_folder(cover)
        input_folder = f"{coordinates_folder}/{infol}"
        output_folder = f"{coordinates_folder}/{cover}"

        input_file = list(map(lambda x: x, os.listdir(input_folder)))
        dict_files = list(map(lambda x: {f"{input_folder}/{x}": f"{output_folder}/{cover}_{x[4:]}"},
                              input_file))

        test = list(map(lambda x: FCC(list(x.keys())[0], list(x.values())[0]), dict_files))

        return (0)

    list_coordinates = list(map(lambda x: f"{las_tiles}/{x}", os.listdir(las_tiles)))
    folder_covers = list(map(lambda x, y: f"{x}/{y}",
                             list_coordinates * len(covers),
                             covers * len(list_coordinates)))
    tmp = list(map(lambda x: os.makedirs(x, exist_ok=True), folder_covers))

    # input_folders = list(map(lambda x,y: f"{x}/{assigning_input_folder(y)}",
    #                          list_coordinates*len(covers),
    #                          covers*len(list_coordinates)
    #                         ))
    # z = FCC_to_folder('tmp/las_tiles/360000.0-4370000.0', "TCC")

    tmp = list(map(lambda x, y: FCC_to_folder(x, y),
                   list_coordinates * len(covers),
                   covers * len(list_coordinates)))

    return (0)

def metrics(lasfile, outputfile, metric, returns="veg4+veg5", position=None, resolution=5, radius=56.41896):
    """
    Creates metrics images from las files.
    :param lasfile:
    :param outputfile:
    :param metric: Metric desired
    :param returns: Specify the returns to take into account ("veg4", "veg5", "veg4+veg5" or "veg3+veg4+veg5")
    :param position: Specifiy a postition (last, first, etc)
    :param resolution: resolution of the output raster image
    :param radius: search radius
    :return:
    """
    if metric not in ["min", "max", "count", "stdev"]:
        return(1)

    if returns == "veg4+veg5":
        variante = {
            "type": "filters.range",
            "limits": "Classification[2:2],Classification[4:5]"
        }
    elif returns == "veg4":
        variante = {
            "type": "filters.range",
            "limits": "Classification[2:2],Classification[4:4]"
        }
    elif returns == "veg5":
        variante = {
            "type": "filters.range",
            "limits": "Classification[2:2],Classification[5:5]"
        }
    elif returns == "veg3+veg4+veg5":
        variante = {
            "type": "filters.range",
            "limits": "Classification[2:2],Classification[3:5]"
        }
    if position != None:
        creating_json = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": f"{lasfile}",
                    "spatialreference": "EPSG:25830"
                },
                {
                    "type": "filters.returns",
                    "groups": f"{position}"
                },
                variante,
                {
                    "type": "writers.gdal",
                    "gdaldriver": "GTiff",
                    "nodata": "-9999",
                    "radius": f"{radius}",
                    "output_type": f"{metric}",
                    "resolution": f"{resolution}",
                    "filename": f"{outputfile}"
                }

            ]
        }
    else:

        creating_json = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": f"{lasfile}",
                    "spatialreference": "EPSG:25830"
                },
                variante,
                {
                    "type": "writers.gdal",
                    "gdaldriver": "GTiff",
                    "nodata": "-9999",
                    "radius": f"{radius}",
                    "output_type": f"{metric}",
                    "resolution": f"{resolution}",
                    "filename": f"{outputfile}"
                }

            ]
        }

    consulta = json.dumps(creating_json, indent=4)
    # print(consulta)

    pipeline = pdal.Pipeline(consulta)
    pipeline.validate()  # Check if json options are good
    pipeline.loglevel = 8
    count = pipeline.execute()
    print(count)


def metric_to_folder(input_folder, output_folder):
    """
    Apply metric folders
    :param input_folder:
    :param output_folder:
    :return:
    """
    def out_name(name):
        a = name.split("/")
        b = a[len(a) - 1]
        return (b)

    def get_metric(name):
        a = name.split("_")
        b = a[0]
        return (b)

    outname = out_name(output_folder)

    metrica = get_metric(outname)

    os.makedirs(output_folder, exist_ok=True)
    listlas = os.listdir(input_folder)
    dict_inout = list(map(lambda x: {f"{input_folder}/{x}": f"{output_folder}/{outname}_{x[6:14]}.tif"},
                          listlas
                          ))
    try:
        #         for d in dict_inout:
        #             print(d)
        tmp = list(map(lambda x, y: metrics(lasfile=list(x.keys())[0],
                                            outputfile=list(x.values())[0],
                                            metric=y
                                            ),
                       dict_inout,
                       [metrica] * len(dict_inout)
                       ))
        print(tmp)
        return (0)
    except:
        return (1)


def metrics_all_folders(las_tiles, metricas, input_folder="clean"):
    """
    Apply metrics to all the folders
    :param las_tiles:
    :param metricas:
    :param input_folder:
    :return:
    """
    coordinates_folder = os.listdir(las_tiles)

    dict_inout = []
    for metrica in metricas:
        dict_inout.append(list(map(lambda x, y, z: {f"{las_tiles}/{x}/{z}": f"{las_tiles}/{x}/{y}"},
                                   coordinates_folder,
                                   [metrica] * len(coordinates_folder),
                                   [input_folder] * len(coordinates_folder)
                                   ))
                          )
    # it has created a list of lists. So we need to simplify
    # that into one list of dictionaries.
    dict_inout = sum(dict_inout, [])
    #     print(dict_inout)
    tmp = list(map(lambda x: metric_to_folder(list(x.keys())[0], list(x.values())[0]),
                   dict_inout))

    print(tmp)


