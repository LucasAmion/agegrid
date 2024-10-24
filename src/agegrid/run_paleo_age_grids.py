import os
from . import automatic_age_grid_seeding as aags

def run_paleo_age_grids(model_name, model_dir, temp_dir, logger, max_time, min_time, time_step,
                        sampling, xmin, xmax, ymin, ymax, num_cpus, spreading_rate):
    ##########################################################
    # Set the input parameters 

    # Input files
    input_rotation_filenames = get_layer_files(model_dir, model_name, 'Rotations')
    topology_features = get_layer_files(model_dir, model_name, 'Topologies')
    COBterrane_file = get_layer_files(model_dir, model_name, 'COBs')[0]
    
    # Output files
    output_gridfile_template = f'{model_name}_seafloor_age_'
    grd_output_dir = f'{temp_dir}/grid_files/'
    
    continent_mask_file_pattern = '%s/masks/mask_{:0.1f}Ma.nc' % grd_output_dir
    seedpoints_output_dir = '{:s}/seedpoints/'.format(grd_output_dir)
    initial_ocean_seedpoint_filename = '{:s}/seedpoints/age_from_distance_to_mor_{:0.2f}Ma.gmt'.format(grd_output_dir, max_time)
    mor_seedpoint_filename = '{:s}/seedpoints/MOR_plus_one_merge_{:0.2f}_{:0.2f}.gmt'.format(grd_output_dir, min_time, max_time)

    if not os.path.isdir(grd_output_dir):
        os.mkdir(grd_output_dir)
    if not os.path.isdir('{0}/unmasked/'.format(grd_output_dir)):
        os.mkdir('{0}/unmasked/'.format(grd_output_dir))
    if not os.path.isdir('{0}/masked/'.format(grd_output_dir)):
        os.mkdir('{0}/masked/'.format(grd_output_dir))
    if not os.path.isdir('{0}/masks/'.format(grd_output_dir)):
        os.mkdir('{0}/masks/'.format(grd_output_dir))
    if not os.path.isdir('{0}/gridding_input/'.format(grd_output_dir)):
        os.mkdir('{0}/gridding_input/'.format(grd_output_dir))
    if not os.path.isdir('{0}/seedpoints/'.format(grd_output_dir)):
        os.mkdir('{0}/seedpoints/'.format(grd_output_dir))
    
    # Time parameters
    max_time = float(max_time)
    min_time = float(min_time)
    gridding_time_step = mor_time_step = time_step
    
    # Spatial parameters
    grdspace = ridge_sampling = sampling
    region = [xmin, xmax, ymin, ymax]
    initial_ocean_healpix_sampling = 32
    area_threshold = 0.0001
    
    # Other
    backend = 'v2'
    initial_ocean_mean_spreading_rate = spreading_rate
    subduction_collision_parameters = (5.0, 10.0)
    ###################################################
    # Run the algorithm
    
    logger.info("Making masks...")
    aags.make_masking_grids(COBterrane_file, input_rotation_filenames, max_time, min_time, gridding_time_step,
                            grdspace, grd_output_dir, output_gridfile_template, num_cpus)
    logger.progress += 10
    
    logger.info("Creating seed points for initial ocean at reconstruction start time...")
    aags.get_initial_ocean_seeds(topology_features, input_rotation_filenames, COBterrane_file, seedpoints_output_dir,
                                max_time, initial_ocean_mean_spreading_rate, initial_ocean_healpix_sampling,
                                area_threshold, mask_sampling=grdspace)
    logger.progress += 10

    logger.info("Generating seed points along mid ocean ridges...")
    aags.get_isochrons_from_topologies(topology_features, input_rotation_filenames, seedpoints_output_dir,
                                    max_time, min_time, mor_time_step, ridge_sampling, num_cpus=num_cpus)
    logger.progress += 10
    
    logger.info("Assembling seed points and reconstructing by topologies...")
    aags.reconstruct_seeds(input_rotation_filenames, topology_features, seedpoints_output_dir,
                        mor_seedpoint_filename, initial_ocean_seedpoint_filename,
                        max_time, min_time, gridding_time_step, grd_output_dir,
                        subduction_collision_parameters=subduction_collision_parameters,
                        continent_mask_file_pattern=continent_mask_file_pattern, backend=backend)
    logger.progress += 10

    logger.info("Exporting and masking grids...")
    aags.make_grids_from_reconstructed_seeds(input_rotation_filenames, max_time, min_time, gridding_time_step,
                                            grdspace, region, grd_output_dir, output_gridfile_template,
                                            num_cpus=num_cpus, COBterrane_file=COBterrane_file)
    logger.progress += 10

def get_layer_files(model_dir, model_name, layer_name):
    dir =  f'{model_dir}/{model_name}/{layer_name}'
    files = os.listdir(dir)
    files.remove('.metadata.json')
    return [f'{dir}/{filename}' for filename in files]
