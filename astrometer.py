#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import io
from sys import argv
import numpy
import matplotlib.pyplot as pyplot
from pyplot import ZScaleInterval
import sep #python sextractor
from numpy import arcsin as asin,arctan2 as atan2,cos,pi,sin,sqrt,square,tan, degrees as deg, radians as rad
from sklearn.neighbors import KDTree
import astroalign

import wcsfit #additional package included in repository

"""
Some portions of the larger image processing infrastructure that require host, server, and user credentails are redacted from this sample:

'retrieve' is a larger function that pulls a specificied exposure file and associated FITS header information

'patch_api' writes information related to exposure to db

'get_api' function pulls associated db information related to exposure

the 'FetchSources' class pulls db information according to the postgresql schema of the Condor ubuntu/linux server

'put_minio' arranges and uploads exposure and image data to Condor's AWS S3 format cloud server

"""

def get_image_ij(image, deep=False):
  BW = 64
  BH = 64
  FW = 9
  FH = 9

  DEBLEND_CONT = 0.005
  DEBLEND_NTHRESH = 20
  FILTER_KERNEL = numpy.array(
    [
      [1, 2, 1],
      [2, 4, 2],
      [1, 2, 1]
    ]
  )
  N_RETRY_SEP = 10
  if deep:
    MINAREA = 20
    THRESH = 1.2
  else:
    MINAREA = 30
    THRESH = 1.4

  print('getting image sources ij...', flush=True)
  background = sep.Background(image.data, bw=BW, bh=BH, fw=FW, fh=FH)
  data0 = image.data - background
  thresh = THRESH
  for _ in range(N_RETRY_SEP):
    try:
      objects = sep.extract(data0, thresh, err=image.sigma, minarea=MINAREA, deblend_nthresh=DEBLEND_NTHRESH, deblend_cont=DEBLEND_CONT, filter_kernel=FILTER_KERNEL)
      break
    except:
      print('sep extraction failed but trying again...')
#      thresh += 0.5
      thresh *= 2.0
  else:
    print('sep extraction failed and giving up...')
    raise
  (flux, fluxerr, flag) = sep.sum_circle(data0, objects['x'], objects['y'], 3.0, err=image.sigma, gain=1.0)

  (r, flag) = sep.flux_radius(data0, objects['x'], objects['y'], 6.0*objects['a'], 1.0, normflux=flux, subpix=4)
  (i, j, flag) = sep.winpos(data0, objects['x'], objects['y'], r, subpix=0)
  var_i = objects['errx2']
  var_j = objects['erry2']
  var_ij = objects['errxy']

  desc = flux.argsort()[::-1]
  i = i[desc]
  j = j[desc]
  flux = flux[desc]
  var_i = var_i[desc]
  var_j = var_j[desc]
  var_ij = var_ij[desc]

  print(f'{len(i)} image sources detected...', flush=True)

  return (i, j, flux, var_i, var_j, var_ij)

def get_gaia_xy(CRVAL1, CRVAL2, pm=False, epoch=2015.5):
  print('getting gaia xy...', flush=True)
  LONPOLE = 180.0
  MAX_G_MAG = 18.0
  QUERY_RA_DEG_WIDTH = 1.6 # half width
  QUERY_DEC_DEG_HEIGHT = 1.1 # half width
  SOURCE_CATALOG = 'gaia'

  fetchsources = sourcestable.FetchSources(postgres_host, postgres_database, postgres_user, postgres_password)
  sources_table = fetchsources(CRVAL1, QUERY_RA_DEG_WIDTH, CRVAL2, QUERY_DEC_DEG_HEIGHT, max_g_mag=MAX_G_MAG)
  print(f'{len(sources_table)} gaia sources found...', flush=True)
  source_id = sources_table['source_id']
  alpha = sources_table['ra_deg']
  delta = sources_table['dec_deg']
  pmra = sources_table['pmra']
  pmra_error = sources_table['pmra_error']
  pmdec = sources_table['pmdec']
  pmdec_error = sources_table['pmdec_error']
  ref_epoch = sources_table['ref_epoch']
  g_mag = sources_table['phot_g_mean_mag']

  if pm:
    delta_year = epoch - ref_epoch
    pmra[pmra == None] = 0.0
    pmdec[pmdec == None] = 0.0
    pmra_error[pmra_error == None] = 9999.9
    pmdec_error[pmdec_error == None] = 9999.9
    numpy.add(alpha , pmra/cos(rad(delta))/1000/3600*delta_year, out=alpha, casting='unsafe')
    numpy.add(delta , pmdec/1000/3600*delta_year, out=delta, casting='unsafe')

  theta = deg(asin(sin(rad(delta))*sin(rad(CRVAL2)) + cos(rad(delta))*cos(rad(CRVAL2))*cos(rad(alpha - CRVAL1))))
  phi = LONPOLE + deg(atan2(sin(rad(delta))*cos(rad(CRVAL2)) - cos(rad(delta))*sin(rad(CRVAL2))*cos(rad(alpha - CRVAL1)), -cos(rad(delta))*sin(rad(alpha - CRVAL1))))
  R = 180.0/pi/tan(rad(theta))
  x = R*cos(rad(phi))
  y = -R*sin(rad(phi))

  asc = g_mag.argsort()
  x = x[asc]
  y = y[asc]
  source_id = source_id[asc]
  alpha = alpha[asc]
  delta = delta[asc]
  pmra = pmra[asc]
  pmra_error = pmra_error[asc]
  pmdec = pmdec[asc]
  pmdec_error = pmdec_error[asc]
  ref_epoch = ref_epoch[asc]
  g_mag = g_mag[asc]

  return (x, y, source_id, alpha, delta, pmra, pmra_error, pmdec, pmdec_error, ref_epoch, g_mag)

def kdt_match(x1, y1, x2, y2, id2=None):
  LEAF_SIZE = 10
  K = 1
  source = numpy.array(join(x1, y1))
  target = numpy.array(join(x2, y2))
  kdtree = KDTree(target, leaf_size=LEAF_SIZE, metric= 'euclidean')
  (distances, indices) = kdtree.query(source, k=K)
  index_m = indices.flatten()
  if id2 is None:
    return (x2[index_m], y2[index_m], index_m)
  return (x2[index_m], y2[index_m], id2[index_m], index_m)

def depth_match(xs, ys, xt, yt, mag_depth=None):
  s= time.time()
  if mag_depth is not None:
    MAG_DEPTH = mag_depth
  else:
    MAG_DEPTH = round(0.3*len(xt))
  print(f"running depth algorithm, searching {MAG_DEPTH} brightest targets...",flush = True)
  LEAF_SIZE = 10
  K = 1
  index_m0 = []
  x0 = xt
  y0 = yt
  x_m = []
  y_m = []
  for xs_,ys_ in zip(xs,ys):
    source = numpy.column_stack((xs_,ys_))
    x0 = numpy.delete(x0,index_m0)
    y0 = numpy.delete(y0,index_m0)
    target = numpy.column_stack((x0,y0))
    kdtree = KDTree(target[:min(MAG_DEPTH, len(target))], leaf_size=LEAF_SIZE, metric= 'euclidean')
    (distances, index_m0) = kdtree.query(source, k=K)
    x_m.append(x0[index_m0])
    y_m.append(y0[index_m0])
  x_m = numpy.concatenate(x_m,axis = 0).flatten()
  y_m = numpy.concatenate(y_m,axis = 0).flatten()
  print(f"{len(xs)} sources matched to {len(x_m)} targets")
  print(f'Duration of Depth Match Algorithm: {time.time() - s} seconds')
  return (x_m,y_m)

def join(a, b): # join two numpy arrays into one coordinate list
  c = []
  for (a_, b_) in zip(a, b):
    c.append((a_, b_))
  return c

def rend(c): # rend one numpy array into two coordinate lists
  a = []
  b = []
  for c_ in c:
    a.append(c_[0])
    b.append(c_[1])
  return (a, b)

def plot_ij(i, j, image, filename):
  (ny, nx) = image.shape
  pyplot.rcParams.update({'font.size': 4})
  axes = pyplot.gca()
  axes.set_aspect(1)
  pyplot.scatter(i, j, s=7, alpha=0.3, facecolors='none', edgecolors='red')
  pyplot.xlim((0, nx))
  pyplot.ylim((0, ny))
  z = ZScaleInterval()
  (z1, z2) = z.get_limits(image)
  pyplot.imshow(image, cmap='gray_r', vmin=z1, vmax=z2)
  pyplot.savefig(filename, bbox_inches='tight')
  pyplot.clf()

def plot_resid(i, j, ii, jj, image, filename):
  SCALE = 300

  (ny, nx) = image.shape
  pyplot.rcParams.update({'font.size': 4})
  axes = pyplot.gca()
  axes.set_aspect(1)
  di = ii - i
  dj = jj - j
  r = sqrt(square(di) + square(dj))*SCALE
  theta = atan2(dj, di)
  di = r*cos(theta)
  dj = r*sin(theta)
  for (i_, j_, di_, dj_) in zip(i, j, di, dj):
    pyplot.plot((i_, i_+di_), (j_, j_+dj_), color='green', linewidth=0.35)
#  pyplot.plot(i, j, marker='o', linestyle='None', color='red', alpha=0.3, markersize=0.75)
  pyplot.scatter(i, j, s=7, alpha=0.3, facecolors='none', edgecolors='red', linewidths=0.75)
  pyplot.xlim((0, nx))
  pyplot.ylim((0, ny))
  z = ZScaleInterval()
  (z1, z2) = z.get_limits(image)
  pyplot.imshow(image, cmap='gray_r', vmin=z1, vmax=z2)
  pyplot.savefig(filename, bbox_inches='tight')
  pyplot.clf()



def wcssolve(image, filename, ra_deg0, dec_deg0, epoch, flip=False, debug=False):
  WCS_VERSION = '1.3'

  WCS_FILEBUCKET = 'wcs'
  WCS_FILEHOST = DATA_HOST_NAME

  MAX_CONTROL_POINTS = 100 # maximum number of points used by astroalign
  SCALE_FACTOR = 1000 # need to scale our platescale for astroalign efficacy

  TARGET_TO_SOURCE_RATIO = 1.2

  DEG_PER_PIX = 0.86/3600
  PADDING_DEG = 0.05

  M = 6 #divide image into 24 regions
  N = 4

  NSOURCES_MIN = 300

  (ny, nx) = image.data.shape

  def set_status_uncalibrated():
    print('error:  cannot determine wcs calibration...', flush=True)
    data = {'system_wcs_status': 'uncalibrated', 'system_wcs_version': WCS_VERSION}
    patch_api(f'exposures/{filename}', data)

  try:
    (i, j, flux, var_i, var_j, var_ij) = get_image_ij(image, deep=False)
  except Exception as e:
    print(e, flush=True)
    set_status_uncalibrated()
    return

  if debug:
    plot_ij(i, j, image.data, '/local/scratch/wcs_ij_in.pdf')

  (x, y, source_id, alpha, delta, pmra, pmra_error, pmdec, pmdec_error, ref_epoch, g_mag) = get_gaia_xy(ra_deg0, dec_deg0, pm=True, epoch=epoch)

  print('finding affine transformations...', flush=True)
  source_m_i = []
  source_m_j = []
  target_m_x = []
  target_m_y = []
  xlim = int(len(i)*TARGET_TO_SOURCE_RATIO)
  ylim = int(len(j)*TARGET_TO_SOURCE_RATIO)
  for m in range(M):
    imin = m*nx//M
    imax = imin + nx//M
    if flip:
      xmin = (imin - nx//2)*DEG_PER_PIX + PADDING_DEG
      xmax = (imax - nx//2)*DEG_PER_PIX + PADDING_DEG
    else:
      xmin = (nx - imax - nx//2)*DEG_PER_PIX - PADDING_DEG
      xmax = (nx - imin - nx//2)*DEG_PER_PIX + PADDING_DEG
    for n in range(N):
      jmin = n*ny//N
      jmax = jmin + ny//N
      indices = numpy.where((i >= imin) & (i <= imax) & (j >= jmin) & (j <= jmax), 1, 0).nonzero()
      source = join(i[indices], j[indices])
      if flip:
        ymin = (jmin - ny//2)*DEG_PER_PIX + PADDING_DEG
        ymax = (jmax - ny//2)*DEG_PER_PIX + PADDING_DEG
      else:
        ymin = (ny - jmax - ny//2)*DEG_PER_PIX - PADDING_DEG
        ymax = (ny - jmin - ny//2)*DEG_PER_PIX + PADDING_DEG
      indices = numpy.where((x[:xlim] >= xmin) & (x[:xlim] <= xmax) & (y[:ylim] >= ymin) & (y[:ylim] <= ymax), 1, 0).nonzero()
      target = join(x[indices]*SCALE_FACTOR, y[indices]*SCALE_FACTOR)
      print(f'imin: {imin} imax: {imax} jmin: {jmin} jmax: {jmax}', flush=True)
      print(f'finding transformation and source matches from {len(source)} sources and {len(target)} targets...', flush=True)
      try:
        (transform, (source_m, target_m)) = astroalign.find_transform(source, target, max_control_points=MAX_CONTROL_POINTS)
        print(f'matched {len(source_m)} sources...', flush=True)
      except:
        print('matched 0 sources...', flush=True)
        continue
      transform.params /= SCALE_FACTOR
      transform.params[2, 2] = 1
      try:
        transform0.params += transform.params
        n0 += 1
      except:
        transform0 = transform
        n0 = 1
      target_m /= SCALE_FACTOR
      (source_m_i_, source_m_j_) = rend(source_m)
      source_m_i += source_m_i_
      source_m_j += source_m_j_
      (target_m_x_, target_m_y_) = rend(target_m)
      target_m_x += target_m_x_
      target_m_y += target_m_y_
  try: # mean transformation
    transform0.params /= n0
  except: # no transformation found
    set_status_uncalibrated()
    return

  print('minimizing rms...', flush=True)
  wcs = wcsfit.WCSFit(nx, ny, ra_deg0, dec_deg0, transform0.params)
  wcs.wcsfit(numpy.array(source_m_i), numpy.array(source_m_j), numpy.array(target_m_x), numpy.array(target_m_y))

  (i, j, flux, var_i, var_j, var_ij) = get_image_ij(image, deep=True)
  if debug:
    plot_ij(i, j, image.data, '/local/scratch/wcs_ij_in.pdf')

  print('matching all sources to all targets...', flush=True)
  (xx, yy) = wcs.i_j_to_xx_yy(i, j)
  try:
    (xx_m,yy_m) = depth_match(xx, yy, x, y, mag_depth=15000)
  except:
    print('error with brightness-limited KDT, running standard KDT...', flush = True)
    (xx_m, yy_m, index_m) = kdt_match(xx, yy, x, y)

  print('minimizing rms of full kdt list...', flush=True)
  wcs.wcsfit(i, j, xx_m, yy_m, 1, var_i, var_j, var_ij)

  print('using clipped kdt list to fit a and sigma_0...', flush=True)
  # print(f"Sanity check: values entering ML_FIT should be length of clipped lists: {len(wcs.i)}, {len(wcs.var_i)}")
  (chisqr0, a2, var_0) = wcs.wcsfit(wcs.i, wcs.j, wcs.x, wcs.y, 0.0, wcs.var_i, wcs.var_j, wcs.var_ij, ML_FIT =True)
  # using previously fitted params,fit full lists of matches using only chi square procedure
  print('full list chisqr fitting...', flush=True)
  wcs.wcsfit(i, j, xx_m, yy_m, 0.0, var_i, var_j, var_ij, a2=a2, var_0=var_0)



  if debug:
    (x_m, y_m) = wcs.xx_yy_to_x_y(wcs.x, wcs.y)
    (i_m, j_m) = wcs.x_y_to_i_j(x_m, y_m)
    plot_ij(i_m, j_m, image.data, '/local/scratch/wcs_ij_out.pdf')
    plot_resid(wcs.i, wcs.j, i_m, j_m, image.data, '/local/scratch/wcs_resid.pdf')

  if wcs.nsources_used < NSOURCES_MIN:
    set_status_uncalibrated()
    return

  wcs_filename = f'{filename.split(".fits")[0]}.wcs.fits'
  with io.BytesIO() as f:
    wcs.hdul.writeto(f)
    raw = f.getvalue()
#  put(minio, FILEBUCKET, f'{FILEOBJECT_PREFIX}{filename_wcs}', raw)
  put_minio(WCS_FILEHOST, WCS_FILEBUCKET, wcs_filename, raw)
  data = {
    'system_wcs_dec_deg': wcs.dec_deg0,
    'system_wcs_field_size_x_deg': wcs.field_size_x_deg,
    'system_wcs_field_size_y_deg': wcs.field_size_y_deg,
    'system_wcs_filebucket': WCS_FILEBUCKET,
    'system_wcs_filehost': WCS_FILEHOST,
    'system_wcs_filename': wcs_filename,
    'system_wcs_fileobject': wcs_filename,
    'system_wcs_filepath': f'{WCS_FILEBUCKET}/{wcs_filename}',
    'system_wcs_nstars_total': wcs.nsources_total,
    'system_wcs_nstars_used': wcs.nsources_used,
    'system_wcs_ra_deg': wcs.ra_deg0,
    'system_wcs_rms_arcsec': wcs.rms_arcsec,
    'system_wcs_rotation_angle_deg': wcs.rotation_angle_deg,
    'system_wcs_sigma_sys_arcsec': sigma_sys_arcsec,
    'system_wcs_status': 'calibrated',
    'system_wcs_version': WCS_VERSION
  }
  patch_api(f'exposures/{filename}', data)





def astrometer(filename, debug=False):
  print('running astrometer...', flush=True)
  try:
    image = retrieve(filename, dispatch=None, wcs=False, bias=True, sigma=True, mask=False , bias_mask=False , rate=False , flux=False, flat=True, background=False, flat_method='twilight')

  except Exception as e:
    print(e)
    print(f'error:  cannot retrieve image {filename}...')
    data = {'system_wcs_status': 'uncalibrated'}
    patch_api(f'exposures/{filename}', data)
    return

  if image.data is None:
    print(f'error:  cannot retrieve image {filename}...')
    data = {'system_wcs_status': 'uncalibrated'}
    patch_api(f'exposures/{filename}', data)

    return
  response0 = get_api(f'exposures/{filename}')
  ra_deg = response0['mount_ra_j2000_hours']/24.0*360.0
  dec_deg = response0['mount_dec_j2000_degs']
  epoch = response0['system_exposure_start_timestamp_microsecond']/1E6/3600/24/365 + 1970
  timestamp_microsecond = response0['system_exposure_start_timestamp_microsecond']
  telescope_id = int(response0['system_telescope_id'])
  # handle camera flip prior to 2021-05-24T11:35:53Z
  if timestamp_microsecond < 1621856153948600 and telescope_id > 2:
    flip = True
  else:
    flip = False
  wcssolve(image, filename, ra_deg, dec_deg, epoch, flip, debug)



# initiate routine as portion of main image processing pipeline. Automatically called as image is admitted to database, can also be called with
# docker as follows:

data = json.loads(argv[1])

task = data.get('task')
if task == 'astrometer':
  filename = data.get('filename')
  debug = data.get('debug', False)
  astrometer(filename, debug)


"""

docker command:

docker run -it -v /local/scratch:/local/scratch --network host --rm process_test '{"task": "astrometer", "filename": "SAMPLEFILE.fits.fz" }'



"""
