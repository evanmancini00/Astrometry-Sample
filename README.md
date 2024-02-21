# Astrometric Calibration Algorithm and Packages

This repository contains Python scripts implementing an intricate astrometric calibration algorithm that matches sources detected from an image to their coordinates pulled from the Gaia telescope database. The main script, `astrometry.py`, orchestrates the calibration process, while the supporting functions are defined in `wcsfit.py`.

## Calibration Process Overview

### Step 1: Source Detection and Gaia Coordinates Retrieval

The `get_image_ij` function detects image sources and their pixel coordinates using relaxed Sextractor (SEP) parameters. Using the telescope's pointing position, the `get_gaia_xy` function retrieves Gaia source coordinates in degrees along with corresponding flux data from the region encapsulating the exposure's field of view.

### Step 2: Region-based Matching and Affine Transformation

- **Divide the Image**: The image is divided into 24 regions.
- **Matching**: Image sources are matched to their Gaia counterparts in each region using the `astroalign` package. This process also determines the affine transformation in each region.
- **Calculate Mean Affine Transformation**: The mean affine transformation across the full field is calculated based on the transformations obtained from individual regions.

### Step 3: Refinement and TPV Distortion Parameters Calculation

- **Refinement**: Lists of matched sources are refined by clipping pairs with root mean square (rms) match distances greater than a defined `CLIPPING_THRESHOLD`.
- **TPV Distortion Parameters**: The refined lists of matched sources are used to calculate the TPV distortion parameters.

### Step 4: Depth-Match Algorithm and Source-Gaia Matching

- **Depth-Match Algorithm**: Sexctractor parameters are relaxed to increase the number of sources dectected across the entire image. Each source is matched with a Gaia target from the original. The primary matching method is referred to as the `depth-match` algorithm, which ranks both source and target list by flux_adu. Working down from the brightest to the faintest source, the algorithm removes source and target pairs as they are matched one-by-one via a KD tree method. The key is that the remaining target list is sliced, and a new kd tree is generated.
- **Matching Logic**: Source and target lists are ranked by flux, and matches are made one-by-one via a KD tree method, with successive rounds of clipping and fitting to weed out incorrectly matched points.
- **Failsafe**: In rare cases where there are more image sources than gaia targets available, depth_match fails. The code will then revert back to a standard kd tree method that generates a kd tree from the entire target list, and matches each source to its nearest neighbor as a group. This will work regardless of either list size, but it allows for the possibility of multiple source points being matched
to the same target, which is obviously incorrect. 

### Step 5: RMS Clipping and Error Model Fitting

- **RMS Clipping**: Lists undergo rounds of RMS clipping before being used to fit the error model.
- **Error Model Determination**: The `like_fit` function within the WCSFit class inputs two coordinate lists and fits the Max-likilihood function to a the error model
relating each image sources centroiding error to match residual distance for each matched set of points. The sources list x,y are the image sources found with sep and the target list xx,yy are their gaia coordinates, both in degrees. Var_i,var_j, and var_ij (covariance) are SEP's centroid errors that must be transformed to degree units. We take the gaia points to have negligible positional errors.
- **Error Model Fitting**: Rounds of clipping for points with residual errors greater than a defined `ERROR_THRESHOLD` are followed by refitting of the error model.

### Step 6: Final Chi-Square Fitting

- **Final Chi-Square Fitting**: Using the final refined error model parameters, the entire list of points is used to fit for an accurate determination of chi-square.

## Files

### astrometry.py

This script orchestrates the calibration process and contains the following key functions:

- `get_image_ij`: Detects image sources and their pixel coordinates using SEP parameters.
- `get_gaia_xy`: Retrieves Gaia source coordinates along with flux data.
- `depth_match`: Implements the depth-match algorithm for source-Gaia matching.
- `wcssolve`: Performs affine transformation, source-Gaia matching, and error model fitting.

### wcsfit.py

This file contains additional functions used for WCS fitting, including:

- Classes and methods for WCS fitting.
- Utility functions for coordinate transformation and plotting.

## Optimization and Usage

- **Parallel Processing**: The code can be optimized for parallel processing to improve performance, especially during source-Gaia matching.
- **Parameter Tuning**: Adjusting parameters such as clipping thresholds and control points can enhance calibration accuracy.
- **Debugging**: Debugging flags are available to aid in troubleshooting and optimization.

## Usage

The calibration routine can be initiated as part of the main image processing pipeline or via Docker. Below is the Docker command for invoking the calibration process:

```
docker run -it -v /local/scratch:/local/scratch --network host --rm process_test '{"task": "astrometer", "filename": "SAMPLEFILE.fits.fz" }'
```

Replace `SAMPLEFILE.fits.fz` with the filename of the FITS image to be calibrated.

## Notes

- Certain portions of the code that involve interaction with Condor's Linux and Cloud databases have been redacted from this sample for security purposes.
- The script and design of these software packages are the property of the Condor Array Telescope Project. They are being shared temporarily with select members of STScI's SCSB for interview and personal evaluation of Evan Mancini for the position of Senior Scientific Software Engineer. 

## Contributors

- Evan Mancini
- Kenneth Lanzetta
  

## License

GNU GPLv3
