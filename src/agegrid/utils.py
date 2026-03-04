import pygplates
import numpy as np
import xarray as xr
from ptt.utils import points_in_polygons


##################################################################
# Coordinate utilities
##################################################################
def lonlat_to_xyz(lons, lats):
    """
    Convert longitude/latitude arrays to 3D Cartesian coordinates on the unit sphere.
    :param lons: array-like of longitudes in degrees
    :param lats: array-like of latitudes in degrees
    :returns: (N, 3) numpy array of [x, y, z] unit-sphere coordinates
    """
    lons_rad = np.radians(lons)
    lats_rad = np.radians(lats)
    return np.column_stack([np.cos(lats_rad) * np.cos(lons_rad),
                            np.cos(lats_rad) * np.sin(lons_rad),
                            np.sin(lats_rad)])


##################################################################
# Continent grid utilities
##################################################################
def load_continent_grid(filename):
    """
    Load a continent mask grid from a NetCDF file and return a RegularGridInterpolator.
    :param filename: path to NetCDF file containing continent mask
    :returns: scipy.interpolate.RegularGridInterpolator (nearest-neighbour)
    """
    from scipy.interpolate import RegularGridInterpolator
    gridX, gridY, gridZ = load_netcdf(filename)
    return RegularGridInterpolator((gridX, gridY), gridZ.T, method='nearest')


def point_on_continent(interpolator, point):
    """
    Test whether a pygplates point lies on a continent using a pre-loaded interpolator.
    :param interpolator: RegularGridInterpolator from load_continent_grid()
    :param point: pygplates.PointOnSphere
    :returns: True if the point is on a continent (interpolated value > 0.5)
    """
    return bool(interpolator([point.to_lat_lon()[1], point.to_lat_lon()[0]]) > 0.5)


##################################################################
# Feature creation helpers
##################################################################
def create_point_feature(lat, lon, begin_time, end_time=-999):
    """
    Create a pygplates point feature with geometry and valid time.
    :param lat: latitude in degrees
    :param lon: longitude in degrees
    :param begin_time: appearance time (Ma)
    :param end_time: disappearance time (Ma), default -999
    :returns: pygplates.Feature
    """
    feature = pygplates.Feature()
    feature.set_geometry(pygplates.PointOnSphere(lat, lon))
    feature.set_valid_time(begin_time, end_time)
    return feature


##################################################################
# Gridding utilities
##################################################################
def block_median_2d(lons, lats, vals, grdspace, region):
    """
    Bin scattered points into cells of size grdspace and return the median
    longitude, latitude, and value for each non-empty cell.
    :param lons: array of longitudes
    :param lats: array of latitudes
    :param vals: array of values to bin
    :param grdspace: cell size in degrees
    :param region: (xmin, xmax, ymin, ymax) bounding box
    :returns: (bm_lons, bm_lats, bm_vals) arrays of block-median results
    """
    from scipy.stats import binned_statistic_2d

    xmin, xmax, ymin, ymax = region
    lon_edges = np.arange(xmin, xmax + 2 * grdspace, grdspace)
    lat_edges = np.arange(ymin, ymax + 2 * grdspace, grdspace)
    stat_val, _, _, _ = binned_statistic_2d(
        lons, lats, vals, statistic='median', bins=[lon_edges, lat_edges])
    stat_lon, _, _, _ = binned_statistic_2d(
        lons, lats, lons, statistic='median', bins=[lon_edges, lat_edges])
    stat_lat, _, _, _ = binned_statistic_2d(
        lons, lats, lats, statistic='median', bins=[lon_edges, lat_edges])
    valid = ~np.isnan(stat_val)
    return stat_lon[valid], stat_lat[valid], stat_val[valid]


##################################################################
# File I/O
##################################################################
def write_xyz_file(output_filename, output_data):
    """
    write data arrays to an xyz file
    :param filename: (string) name of output ascii file
    :param output_data: name of array containing data to be written
    """
    with open(output_filename, 'w') as output_file:
        for output_line in output_data:
            output_file.write(' '.join(str(item) for item in output_line) + '\n')


def load_netcdf(grdfile,z_field_name='z'):

    ds_disk = xr.open_dataset(grdfile)

    data_array = ds_disk[z_field_name]
    coord_keys = [key for key in data_array.coords.keys()]  # updated for python3 compatibility

    if 'lon' in coord_keys[0].lower():
        latitude_key=1; longitude_key=0
    elif 'x' in coord_keys[0].lower():
        latitude_key=1; longitude_key=0
    else:
        latitude_key=0; longitude_key=1

    try:
        gridX = data_array.coords[coord_keys[longitude_key]].data
        gridY = data_array.coords[coord_keys[latitude_key]].data
        gridZ = data_array.data
    except:
        # attempt to handle old-school GMT netcdfs (e.g. produced by grdconvert)
        gridX = np.linspace(ds_disk.data_vars['x_range'].data[0],
                            ds_disk.data_vars['x_range'].data[1],
                            ds_disk.data_vars['dimension'].data[0])
        gridY = np.linspace(ds_disk.data_vars['y_range'].data[0],
                            ds_disk.data_vars['y_range'].data[1],
                            ds_disk.data_vars['dimension'].data[1])
        gridZ = np.flipud(ds_disk.data_vars[z_field_name].data.reshape(ds_disk.data_vars['dimension'].data[1],
                                                                       ds_disk.data_vars['dimension'].data[0]))

    ds_disk.close()

    if gridZ.shape[0]==gridX.shape[0]:
        gridZ = gridZ.T

    return gridX,gridY,gridZ


##################################################################
# Sphere utilities
##################################################################
def healpix_mesh(nSide):
    """
    create a set of healpix points, returned as numpy arrays of the longitudes and latitudes
    """
    #import healpy as hp
    from astropy_healpix import healpy as hp
    othetas,ophis = hp.pix2ang(nSide,np.arange(12*nSide**2))
    othetas = np.pi/2-othetas
    ophis[ophis>np.pi] -= np.pi*2

    # ophis -> longitude, othetas -> latitude
    return np.degrees(ophis), np.degrees(othetas)


##################################################################
# Feature creation
##################################################################
def create_gpml_velocity_feature(longitude_array,latitude_array,filename=None,feature_type=None):
# function to make a velocity mesh nodes at an arbitrary set of points defined in Lat
# Long and Lat are assumed to be 1d arrays.

    multi_point = pygplates.MultiPointOnSphere(zip(latitude_array,longitude_array))

    # Create a feature containing the multipoint feature.
    # optionally, define as 'MeshNode' type, so that GPlates will recognise it as a velocity layer
    if feature_type=='MeshNode':
        meshnode_feature = pygplates.Feature(pygplates.FeatureType.create_from_qualified_string('gpml:MeshNode'))
        meshnode_feature.set_name('Velocity Mesh Nodes')
    else:
        meshnode_feature = pygplates.Feature()
        meshnode_feature.set_name('Multipoint Feature')

    meshnode_feature.set_geometry(multi_point)

    output_feature_collection = pygplates.FeatureCollection(meshnode_feature)

    if filename is not None:
        output_feature_collection.write(filename)
    else:
        return output_feature_collection


def create_gpml_healpix_mesh(nSide,filename=None,feature_type=None):

    # call the function to create a healpix array
    longitude_array,latitude_array = healpix_mesh(nSide)

    # call the function to create a multipoint feature, with user-defined type
    output_feature_collection = create_gpml_velocity_feature(longitude_array,latitude_array,filename,feature_type)

    if filename is not None:  # This is superfluous, since file has already been written in previous line???
        output_feature_collection.write(filename)
    else:
        return output_feature_collection


def create_gpml_regular_long_lat_mesh(Sampling=1,filename=None,feature_type=None):

    # call the function to create a healpix array
    longitude_array,latitude_array = np.meshgrid(np.arange(-180.,180.001,Sampling),np.arange(-90.,90.001,Sampling))
    longitude_array = longitude_array.flatten()
    latitude_array = latitude_array.flatten()

    # call the function to create a multipoint feature, with user-defined type
    output_feature_collection = create_gpml_velocity_feature(longitude_array,latitude_array,filename,feature_type)

    if filename is not None:
        output_feature_collection.write(filename)
    else:
        return output_feature_collection


##################################################################
# Spatial utilities
##################################################################
def force_polygon_geometries(input_features):
# given any pygplates feature collection, creates an output feature collection
# where all geometries are polygons based on the input geometries
# intended for use in forcing features that are strictly polylines to close

    polygons = []
    for feature in input_features: 
        for geom in feature.get_all_geometries():
            polygon = pygplates.Feature(feature.get_feature_type())
            polygon.set_geometry(pygplates.PolygonOnSphere(geom))
            polygon.set_reconstruction_plate_id(feature.get_reconstruction_plate_id())
            # some features in COBTerranes had invalid time ranges - these with throw an error if 
            # we try to create a new feature with same times
            if feature.get_valid_time()[0]>=feature.get_valid_time()[1]:
                polygon.set_valid_time(feature.get_valid_time()[0],feature.get_valid_time()[1])
                polygons.append(polygon)
    polygon_features = pygplates.FeatureCollection(polygons)

    return polygon_features


def polygon_area_threshold(polygons,area_threshold):
    
    polygons_larger_than_threshold = []
    for polygon in polygons:
        if polygon.get_geometry() is not None:
            if polygon.get_geometry().get_area()>area_threshold:
                polygons_larger_than_threshold.append(polygon)

    return polygons_larger_than_threshold


#This is a function to do fast point in polygon text
def run_grid_pip(time,points,polygons,rotation_model,grid_dims):

    reconstructed_polygons = []
    pygplates.reconstruct(polygons,rotation_model,reconstructed_polygons,time)

    rpolygons = []
    for polygon in reconstructed_polygons:
        if polygon.get_reconstructed_geometry():
            rpolygons.append(polygon.get_reconstructed_geometry())

    polygons_containing_points = points_in_polygons.find_polygons(points, rpolygons)

    lat = []
    lon = []
    zval = []
    for pcp,point in zip(polygons_containing_points,points):
        lat.append(point.get_latitude())
        lon.append(point.get_longitude())
        if pcp is not None:
            zval.append(1)
        else:
            zval.append(0)

    bi = np.array(zval).reshape(grid_dims[0],grid_dims[1])

    return bi


def merge_polygons(polygons,rotation_model,
                   reconstruction_time=0,sampling=1.,area_threshold=None,filename=None,
                   return_raster=False):

    from skimage import measure

    multipoints = create_gpml_regular_long_lat_mesh(sampling)
    grid_dims = (int(180/sampling)+1,int(360/sampling)+1)

    for multipoint in multipoints:
        for mp in multipoint.get_all_geometries():
            points = mp.to_lat_lon_point_list()

    bi = run_grid_pip(reconstruction_time,points,polygons,rotation_model,grid_dims)
    
    if return_raster:
        return bi
    
    else:
        # To handle edge effects, pad grid before making contour polygons  
        ## --- start
        pad_hor = np.zeros((1,bi.shape[1]))
        pad_ver = np.zeros((bi.shape[0]+2,1))
        pad1 = np.vstack((pad_hor,bi,pad_hor))      # add row of zeros to top and bottom
        pad2 = np.hstack((pad_ver,pad1,pad_ver))    # add row of zeros to left and right
        #pad3 = np.hstack((pad2,pad_ver))
        contours = measure.find_contours(pad2, 0.5, fully_connected='low')
        ## --- end
    
        contour_polygons = []
        contour_features = []
    
        for n,cp in enumerate(contours):
        
            # To handle edge effects again - strip off parts of polygon
            # due to padding, and adjust from image coordinates to long/lat
            # --- start
            cp[:,1] = (cp[:,1]*sampling)-sampling
            cp[:,0] = (cp[:,0]*sampling)-sampling
            cp[np.where(cp[:,0]<0.),0] = 0
            cp[np.where(cp[:,0]>180.),0] = 180
            cp[np.where(cp[:,1]<0.),1] = 0
            cp[np.where(cp[:,1]>360.),1] = 360
            ## --- end
        
            cpf = pygplates.PolygonOnSphere(zip(cp[:,0]-90,cp[:,1]-180))
            contour_polygons.append(cpf)
        
            feature = pygplates.Feature()
            feature.set_geometry(cpf)
            contour_features.append(feature)

        if filename is not None:
            pygplates.FeatureCollection(contour_features).write(filename)

        else:
            return contour_features


# TODO merge the next two function, since they are largely duplicative
#  
# This cell uses COB Terranes to make a masking polygon
# (which is called 'seive_polygons')
def get_merged_cob_terrane_polygons(COBterrane_file, rotation_model, reconstruction_time,
                                    sampling, area_threshold=None, return_raster=False):

    polygon_features = pygplates.FeatureCollection(COBterrane_file)

    cobter = force_polygon_geometries(polygon_features)

    cf = merge_polygons(cobter, rotation_model, reconstruction_time=reconstruction_time, sampling=sampling)
    
    if area_threshold is not None:
        sieve_polygons = polygon_area_threshold(cf, area_threshold)
        return sieve_polygons

    else:
        return cf

# This cell uses COB Terranes to make a masking polygon
# (which is called 'seive_polygons')
def get_merged_cob_terrane_raster(COBterrane_file, rotation_model, reconstruction_time,
                                  sampling, method='pygplates'):

    if method == 'pygplates':
        polygon_features = pygplates.FeatureCollection(COBterrane_file)

        cobter = force_polygon_geometries(polygon_features)

        mask = merge_polygons(cobter, rotation_model, reconstruction_time=reconstruction_time,
                                sampling=sampling, return_raster=True)

    elif method=='rasterio':
        import tempfile
        import geopandas as gpd
        from rasterio.features import rasterize, Affine

        polygon_features = pygplates.FeatureCollection(COBterrane_file)

        polygon_features = force_polygon_geometries(polygon_features)

        with tempfile.TemporaryDirectory() as temporary_directory:
            pygplates.reconstruct(polygon_features, rotation_model, '{:s}/masking_temp.shp'.format(temporary_directory), reconstruction_time)

            gdf = gpd.read_file('{:s}/masking_temp.shp'.format(temporary_directory))

        dims = (int(180./sampling)+1, int(360./sampling)+1)
        transform = Affine(sampling, 0.0, -180.-sampling/2., 0.0, sampling, -90.-sampling/2.)
    
        geometry_zval_tuples = [(x.geometry, 1) for i, x in gdf.iterrows()]
        
        mask = rasterize(
            geometry_zval_tuples,
            transform=transform,
            out_shape=dims)

        # the first and last columns should match, but may not due to the imposed dateline
        mask[:,0] = mask[:,-1]

    return mask


##################################################################
# Paleogeography
##################################################################
def rasterise_paleogeography(pg_features,rotation_model,time,
							 sampling=0.5,env_list=None,meshtype='LongLatGrid',
							 masking=None):
    # takes paleogeography polygons like those from Cao++ 2017 and converts them
    # into a raster
    # if meshtype is set to 'healpix', sampling should be set to an integer defining nSide

    #pg_features = load_paleogeography(pg_dir,env_list)
    if meshtype=='healpix':
        raster_domain = create_gpml_healpix_mesh(sampling,filename=None,feature_type='MeshNode')
    else:
        raster_domain = create_gpml_regular_long_lat_mesh(sampling,filename=None,feature_type='MeshNode')

    plate_partitioner = pygplates.PlatePartitioner(pg_features, rotation_model, reconstruction_time=time)

    if masking is not None:
        pg_points = plate_partitioner.partition_features(raster_domain,
														 partition_return = pygplates.PartitionReturn.separate_partitioned_and_unpartitioned,
                                                         properties_to_copy=[pygplates.PropertyName.gpml_shapefile_attributes])
        if masking == 'Outside':
            pg_points = pg_points[0]
        elif masking == 'Inside':
            pg_points = pg_points[1]

    else:
        pg_points = plate_partitioner.partition_features(raster_domain,
                                                         properties_to_copy=[pygplates.PropertyName.gpml_shapefile_attributes])

    return pg_points


########################
# PALEOBATHYMETRY
def find_distance_to_nearest_ridge(resolved_topologies,shared_boundary_sections,
                                   point_features,fill_value=5000.):

    all_point_distance_to_ridge = []
    all_point_lats = []
    all_point_lons = []

    for topology in resolved_topologies:
        plate_id = topology.get_resolved_feature().get_reconstruction_plate_id()
        print('Generating distances for Plate %d ...' % plate_id)

        # Section to isolate the mid-ocean ridge segments that bound the current plate
        mid_ocean_ridges_on_plate = []
        for shared_boundary_section in shared_boundary_sections:

            if shared_boundary_section.get_feature().get_feature_type() == pygplates.FeatureType.create_gpml('MidOceanRidge'):
                for shared_subsegment in shared_boundary_section.get_shared_sub_segments():
                    sharing_resolved_topologies = shared_subsegment.get_sharing_resolved_topologies()
                    for resolved_polygon in sharing_resolved_topologies:
                        if resolved_polygon.get_feature().get_reconstruction_plate_id() == plate_id:
                            mid_ocean_ridges_on_plate.append(shared_subsegment.get_resolved_geometry())

        point_distance_to_ridge = []
        point_lats = []
        point_lons = []

        for point_feature in point_features:

            for points in point_feature.get_geometries():
                for point in points:

                    if topology.get_resolved_geometry().is_point_in_polygon(point):

                        if len(mid_ocean_ridges_on_plate)>0:

                            min_distance_to_ridge = None

                            for ridge in mid_ocean_ridges_on_plate:
                                distance_to_ridge = pygplates.GeometryOnSphere.distance(point,ridge,min_distance_to_ridge)

                                if distance_to_ridge is not None:
                                    min_distance_to_ridge = distance_to_ridge

                            point_distance_to_ridge.append(min_distance_to_ridge*pygplates.Earth.mean_radius_in_kms)
                            point_lats.append(point.to_lat_lon()[0])
                            point_lons.append(point.to_lat_lon()[1])

                        else:

                            point_distance_to_ridge.append(fill_value)
                            point_lats.append(point.to_lat_lon()[0])
                            point_lons.append(point.to_lat_lon()[1])

        all_point_distance_to_ridge.extend(point_distance_to_ridge)
        all_point_lats.extend(point_lats)
        all_point_lons.extend(point_lons)


    return all_point_lons,all_point_lats,all_point_distance_to_ridge
