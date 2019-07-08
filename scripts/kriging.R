library(raster)
library(sf)
library(automap)

#####################################################
create_PixelsDF <- function(points, cellsize = 1000, seed = 1312){
  # https://gis.stackexchange.com/questions/158021/plotting-map-resulted-from-kriging-in-r/164421
  if("sf" %in% class(points)) points <- as(points, "Spatial")
  if(class(points) != "SpatialPointsDataFrame") stop("Input points have to be SpatialPointsDataFrame")
  if(dim(points)[2] > 1) stop("Input points just have to have one variable")
  
  set.seed(1312)
  srs <- proj4string(points)
  bbox <- as.vector(points@bbox)
  minx <- bbox[1]
  miny <- bbox[2]
  maxx <- bbox[3]
  maxy <- bbox[4]
  xlength <- maxx - minx
  ylength <- maxy - miny
  ncol <- round(xlength/cellsize, 0)
  nrow <- round(ylength/cellsize, 0)
  grid <- GridTopology(cellcentre.offset = c(minx, miny),
                       cellsize = c(cellsize, cellsize),
                       cells.dim = c(ncol, nrow))
  grid <- SpatialPixelsDataFrame(grid,
                                 data = data.frame(id = 1:prod(ncol, nrow)),
                                 proj4string = CRS(srs))
  return(grid)
  
}  

resampling <- function(rast, to, ...){
        a <- raster::crop(rast, to)
        b <- resample(a, to, ...)
        return(b)
}

get_DependentVar <- function(formula){
        t1 <- strsplit(formula, "~")[[1]][2]
        t2 <- strsplit(t1, "\\+")[[1]]
        t3 <- strsplit(t2, " ")
        out <- c()
        for(var in t3){
                out <- c(out, var[2])
        }
        return(out)
} 

doRegKrig <- function(points, var, formula, listrasters, res = 1000, epsg = 25830){
        env <- environment()
        
        nombres <- names(listrasters)
        dependentvar <- get_DependentVar(formula)
        if(sum(dependentvar %in% nombres == FALSE) > 0) {
                stop("Raster layers doesn't include all the dependent variables. \n
                     Rasters list has to have the following format: \n
                     listarasters <- list(mdt = 'somepath', lat = 'somepath', etc)")
        }
        
        punticos <- sf::st_read(points)
        punticos <- punticos[!is.na(st_dimension(punticos)), ]
        punticos <- as(punticos, "Spatial")
        # tmp <- punticos
        
        for(i in 1:length(listrasters)){
          assign(x = nombres[i], raster(listrasters[[i]]), envir = env)
        }
        
        punticos <- raster::extract(mdt, punticos, sp = T)
        punticos <- raster::extract(lat, punticos, sp = T)
        punticos <- raster::extract(dist, punticos, sp = T)
        
        nmax <- ceiling(nrow(punticos) * 0.7)
        nmin <- ceiling(nrow(punticos) * 0.5)
        
        names(punticos) <- c(var, "mdt", "lat", "dist")
        
        grid <- create_PixelsDF(punticos[, var], cellsize = res)
        tmp <- raster(grid)
        
        
        dist_crop <- resampling(dist, tmp, "ngb")
        mdt_crop <- resampling(mdt, tmp, "ngb")
        lat_crop <- resampling(lat, tmp, "ngb")
        
        stacka <- stack(dist_crop, mdt_crop, lat_crop)
        stacka <- as(stacka, "SpatialPixelsDataFrame")
        

        formula <- as.formula(formula)
        krig <- autoKrige(formula, input_data = punticos, new_data = stacka, nmax = nmax, nmin = nmin)
        pred <- krig$krige_output["var1.pred"]
        pred <- raster(pred)
        crs(pred) <- CRS(paste("+init=epsg:", epsg, sep=""))
        print(crs(pred))
        return(pred)
        
}

doKrig <- function(points, var, res = 1000, epsg = 25830){
        punticos <- sf::st_read(points)
        punticos <- punticos[!is.na(st_dimension(punticos)), ]
        punticos <- as(punticos, "Spatial")
        
        nmax <- ceiling(nrow(punticos) * 0.7)
        nmin <- ceiling(nrow(punticos) * 0.5)
        grid <- create_PixelsDF(punticos[, var], cellsize = res)
        
        formula <- sprintf("%s ~ 1", var)
        formula <- as.formula(formula)
        krig <- autoKrige(formula, input_data = punticos, new_data = grid, nmax = nmax, nmin = nmin)
        pred <- krig$krige_output["var1.pred"]
        pred <- raster(pred)
        crs(pred) <- CRS(paste("+init=epsg:", epsg, sep=""))
        return(pred)
}


#####################################################

# rasterslist <- list(mdt = "../input/mdt.tif",
#                     lat  = "../input/lat.tif",
#                     dist = "../input/dist.tif"
#                     )
# a <- doRegKrig(points = "../output/temperatura_murcia.gpkg", var = "tmed",
#                formula = "tmed ~ mdt + lat + dist", listrasters = rasterslist)
# writeRaster(a, "../output/testrk_temp.tif")


