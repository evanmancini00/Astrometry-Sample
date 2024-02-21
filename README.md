# Astrometry-Sample

Astrometric Calibration Algorithm

Overview
The astrometer.py script implements an astrometric calibration algorithm, which involves several key steps for accurate celestial coordinate determination. This README provides a detailed explanation of the algorithm's workflow.
Step 1: Source Detection and Gaia Coordinates Retrieval
The get_image_ij function is responsible for detecting image sources using a relaxed set of parameters. These sources' pixel coordinates are determined based on the pointing position of the telescope. Subsequently, the get_gaia_xy function retrieves corresponding Gaia source coordinates in degrees, along with flux data.
Step 2: Image Subdivision and Astroalign Affine Transformation
The image is divided into 24 regions, and the astroalign library is employed to match image sources to their Gaia counterparts in each subregion. Astroalign determines the affine transformation in each region, and the mean affine transformation is calculated across the full field.
Step 3: Refinement of Matched Source Lists
The mean transformation parameters obtained in Step 2 are used to fit the lists of matched sources. Pairs with root mean square (rms) match distances greater than a defined threshold (CLIPPING_THRESHOLD) are clipped, and the refined lists are used to calculate the Third-order Polynomial Variables (TPV) distortion parameters.
Step 4: Depth-Match Algorithm for Full Image
The get_image_ij function is applied to detect sources across the entire image using relaxed sextractor (SEP) parameters. Each source is matched to a Gaia target in the original list. The primary matching method is the 'depth-match' algorithm, ranking both source and target lists by flux_adu. The algorithm works from the brightest to the faintest source, removing source and target pairs as they are matched. The depth-match algorithm handles scenarios where there are more image sources than Gaia targets.
In cases where depth_match fails due to insufficient Gaia targets, the script reverts to a standard KD tree method. This method generates a KD tree from the entire target list, matching each source to its nearest neighbor as a group.
Step 5: RMS Clipping and Error Model Refinement
Lists from Step 3 go through rounds of root mean square (rms) clipping before fitting the error model. Rounds of clipping are performed for points with residual errors greater than a defined threshold (ERROR_THRESHOLD). With each subsequent round of residual error (sigma) sigma clipping, the transformation parameters are refitted to refine the error model.
Step 6: Final Chi-Square Fitting
Using the final refined transformation parameters, specifically a and sigma_0, the entire list of points (from Step 4) is used to fit chi-square using the predefined error model. This step ensures an accurate astrometric calibration.
Execution
The script can be executed either as part of the main image processing pipeline or through a Docker command. The Docker command provides flexibility in specifying the task and filename for astrometric calibration.
Docker Command
bash

Copy code
docker run -it -v /local/scratch:/local/scratch --network host --rm process_test '{"task": "astrometer", "filename": "SAMPLEFILE.fits.fz" }'
Dependencies
The script relies on various Python libraries, including numpy, matplotlib, sep (python sextractor), scikit-learn, astroalign, and a custom package wcsfit. Ensure these dependencies are installed before running the script.
Notes
* The script accommodates flips in telescope pointing prior to a specific timestamp.
* Image retrieval, division, and matching parameters are configured within the script.
* The algorithm provides detailed astrometric calibration information, including field size, distortion parameters, and chi-square statistics.
