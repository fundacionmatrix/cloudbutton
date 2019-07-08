library(sf)


#####################################################

get_extent <- function(layer, dist, s_srs = 25830){
        a <- sf::st_read(layer)
        a <- sf::st_set_crs(a, s_srs)
        b <- sf::st_buffer(a, dist)
        e <- sf::st_bbox(b)
        e <- as.vector(e)
        return(e)
}

#####################################################

#extent <- get_extent(layer = "../input/murcia.gpkg", dist = 500000)
#print(extent)
