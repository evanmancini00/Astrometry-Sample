#!/home/ubuntu/home/wcsfit/venv/bin/python3

import math

import numpy
from astropy.io import fits
from numba import njit
from numpy import arcsin as asin
from numpy import arctan as atan
from numpy import arctan2 as atan2
from numpy import cos
from numpy import degrees as deg
from numpy import pi
from numpy import radians as rad
from numpy import sin
from numpy import sqrt
from numpy import square
from numpy import tan
from scipy.optimize import minimize

@njit
def do_i_j_to_x_y(i, j, affine_transform):
  x = affine_transform[0, 0]*i + affine_transform[0, 1]*j + affine_transform[0, 2]
  y = affine_transform[1, 0]*i + affine_transform[1, 1]*j + affine_transform[1, 2]
  return (x, y)

@njit
def do_x_y_to_xx_yy(x, y, params):
  [PV1_0, PV1_1, PV1_2, PV1_3, PV1_4, PV1_5, PV1_6, PV1_7, PV1_8, PV1_9, PV1_10, PV1_11, PV1_12, PV1_13, PV1_14, PV1_15, PV1_16, PV1_17, PV1_18, PV1_19, PV1_20, PV1_21, PV1_22, PV1_23, PV1_24, PV1_25, PV1_26, PV1_27, PV1_28, PV1_29, PV1_30, PV1_31, PV1_32, PV1_33, PV1_34, PV1_35, PV1_36, PV1_37, PV1_38, PV1_39, PV2_0, PV2_1, PV2_2, PV2_3, PV2_4, PV2_5, PV2_6, PV2_7, PV2_8, PV2_9, PV2_10, PV2_11, PV2_12, PV2_13, PV2_14, PV2_15, PV2_16, PV2_17, PV2_18, PV2_19, PV2_20, PV2_21, PV2_22, PV2_23, PV2_24, PV2_25, PV2_26, PV2_27, PV2_28, PV2_29, PV2_30, PV2_31, PV2_32, PV2_33, PV2_34, PV2_35, PV2_36, PV2_37, PV2_38, PV2_39] = params
  xx = x*x
  yy = y*y
  xxx = x*xx
  yyy = y*yy
  xxxx = x*xxx
  yyyy = y*yyy
  xxxxx = x*xxxx
  yyyyy = y*yyyy
  xxxxxx = x*xxxxx
  yyyyyy = y*yyyyy
  xxxxxxx = x*xxxxxx
  yyyyyyy = y*yyyyyy
  r = sqrt(xx + yy)
  rr = r*r
  rrr = r*rr
  rrrrr = rr*rrr
  rrrrrrr = rr*rrrrr
  xy = yx = x*y
  xxy = yxx = xx*y
  xyy = yyx = x*yy
  xxxy = yxxx = xxx*y
  xxyy = yyxx = xx*yy
  xyyy = yyyx = x*yyy
  xxxxy = yxxxx = xxxx*y
  xxxyy = yyxxx = xxx*yy
  xxyyy = yyyxx = xx*yyy
  xyyyy = yyyyx = x*yyyy
  xxxxxy = yxxxxx = xxxxx*y
  xxxxyy = yyxxxx = xxxx*yy
  xxxyyy = yyyxxx = xxx*yyy
  xxyyyy = yyyyxx = xx*yyyy
  xyyyyy = yyyyyx = x*yyyyy
  xxxxxxy = yxxxxxx = xxxxxx*y
  xxxxxyy = yyxxxxx = xxxxx*yy
  xxxxyyy = yyyxxxx = xxxx*yyy
  xxxyyyy = yyyyxxx = xxx*yyyy
  xxyyyyy = yyyyyxx = xx*yyyyy
  xyyyyyy = yyyyyyx = x*yyyyyy
  xx_ = PV1_0 + PV1_1*x + PV1_2*y + PV1_3*r + PV1_4*xx + PV1_5*xy + PV1_6*yy + PV1_7*xxx + PV1_8*xxy + PV1_9*xyy + PV1_10*yyy + PV1_11*rrr + PV1_12*xxxx + PV1_13*xxxy + PV1_14*xxyy + PV1_15*xyyy + PV1_16*yyyy + PV1_17*xxxxx + PV1_18*xxxxy + PV1_19*xxxyy + PV1_20*xxyyy + PV1_21*xyyyy + PV1_22*yyyyy + PV1_23*rrrrr + PV1_24*xxxxxx + PV1_25*xxxxxy + PV1_26*xxxxyy + PV1_27*xxxyyy + PV1_28*xxyyyy + PV1_29*xyyyyy + PV1_30*yyyyyy + PV1_31*xxxxxxx + PV1_32*xxxxxxy + PV1_33*xxxxxyy + PV1_34*xxxxyyy + PV1_35*xxxyyyy + PV1_36*xxyyyyy + PV1_37*xyyyyyy + PV1_38*yyyyyyy + PV1_39*rrrrrrr
  yy_ = PV2_0 + PV2_1*y + PV2_2*x + PV2_3*r + PV2_4*yy + PV2_5*yx + PV2_6*xx + PV2_7*yyy + PV2_8*yyx + PV2_9*yxx + PV2_10*xxx + PV2_11*rrr + PV2_12*yyyy + PV2_13*yyyx + PV2_14*yyxx + PV2_15*yxxx + PV2_16*xxxx + PV2_17*yyyyy + PV2_18*yyyyx + PV2_19*yyyxx + PV2_20*yyxxx + PV2_21*yxxxx + PV2_22*xxxxx + PV2_23*rrrrr + PV2_24*yyyyyy + PV2_25*yyyyyx + PV2_26*yyyyxx + PV2_27*yyyxxx + PV2_28*yyxxxx + PV2_29*yxxxxx + PV2_30*xxxxxx + PV2_31*yyyyyyy + PV2_32*yyyyyyx + PV2_33*yyyyyxx + PV2_34*yyyyxxx + PV2_35*yyyxxxx + PV2_36*yyxxxxx + PV2_37*yxxxxxx + PV2_38*xxxxxxx + PV2_39*rrrrrrr
  return (xx_, yy_)

class WCSFit():

  NPARAM = 84

  def __init__(self, nx, ny, CRVAL1, CRVAL2, affine_transform, distortion_params=None):
    self.nx = nx
    self.ny = ny
    self.CRVAL1 = CRVAL1
    self.CRVAL2 = CRVAL2
    self.affine_transform = affine_transform
    self.distortion_params = distortion_params

    self.nparam = self.NPARAM

    self.iteration = 0
    self.i = None
    self.j = None
    self.x = None
    self.y = None

    self.ra_deg0 = None
    self.dec_deg0 = None
    self.field_size_x_deg = None
    self.field_size_y_deg = None
    self.rotation_angle_deg = None
    self.rms_arcsec = None
    self.chisqr_per_ndof = None
    self.nsources_total = None
    self.nsources_used= None

  @property
  def affine_transform(self):
    return numpy.array(
      [
        [self.CD1_1, self.CD1_2, -self.CD1_1*self.CRPIX1 - self.CD1_2*self.CRPIX2],
        [self.CD2_1, self.CD2_2, -self.CD2_1*self.CRPIX1 - self.CD2_2*self.CRPIX2],
        [0, 0, 1]
      ]
    )

  @affine_transform.setter
  def affine_transform(self, value):
    self.CD1_1 = value[0, 0]
    self.CD1_2 = value[0, 1]
    self.CD2_1 = value[1, 0]
    self.CD2_2 = value[1, 1]
    denom = self.CD1_2*self.CD2_1 - self.CD1_1*self.CD2_2
    self.CRPIX1 = (self.CD2_2*value[0, 2] - self.CD1_2*value[1, 2])/denom
    self.CRPIX2 = (self.CD1_1*value[1, 2] - self.CD2_1*value[0, 2])/denom

  @property
  def distortion_params(self):
    if self.distortion_params_are_set:
      return numpy.array([self.PV1_0, self.PV1_1, self.PV1_2, self.PV1_3, self.PV1_4, self.PV1_5, self.PV1_6, self.PV1_7, self.PV1_8, self.PV1_9, self.PV1_10, self.PV1_11, self.PV1_12, self.PV1_13, self.PV1_14, self.PV1_15, self.PV1_16, self.PV1_17, self.PV1_18, self.PV1_19, self.PV1_20, self.PV1_21, self.PV1_22, self.PV1_23, self.PV1_24, self.PV1_25, self.PV1_26, self.PV1_27, self.PV1_28, self.PV1_29, self.PV1_30, self.PV1_31, self.PV1_32, self.PV1_33, self.PV1_34, self.PV1_35, self.PV1_36, self.PV1_37, self.PV1_38, self.PV1_39, self.PV2_0, self.PV2_1, self.PV2_2, self.PV2_3, self.PV2_4, self.PV2_5, self.PV2_6, self.PV2_7, self.PV2_8, self.PV2_9, self.PV2_10, self.PV2_11, self.PV2_12, self.PV2_13, self.PV2_14, self.PV2_15, self.PV2_16, self.PV2_17, self.PV2_18, self.PV2_19, self.PV2_20, self.PV2_21, self.PV2_22, self.PV2_23, self.PV2_24, self.PV2_25, self.PV2_26, self.PV2_27, self.PV2_28, self.PV2_29, self.PV2_30, self.PV2_31, self.PV2_32, self.PV2_33, self.PV2_34, self.PV2_35, self.PV2_36, self.PV2_37, self.PV2_38, self.PV2_39])
    else:
      return None

  @distortion_params.setter
  def distortion_params(self, value):
    if value is None:
      self.PV1_0 = self.PV2_0 = 0.0
      self.PV1_1 = self.PV2_1 = 1.0
      self.PV1_2 = self.PV1_3 = self.PV1_4 = self.PV1_5 = self.PV1_6 = self.PV1_7 = self.PV1_8 = self.PV1_9 = self.PV1_10 = self.PV1_11 = self.PV1_12 = self.PV1_13 = self.PV1_14 = self.PV1_15 = self.PV1_16 = self.PV1_17 = self.PV1_18 = self.PV1_19 = self.PV1_20 = self.PV1_21 = self.PV1_22 = self.PV1_23 = self.PV1_24 = self.PV1_25 = self.PV1_26 = self.PV1_27 = self.PV1_28 = self.PV1_29 = self.PV1_30 = self.PV1_31 = self.PV1_32 = self.PV1_33 = self.PV1_34 = self.PV1_35 = self.PV1_36 = self.PV1_37 = self.PV1_38 = self.PV1_39 = self.PV2_2 = self.PV2_3 = self.PV2_4 = self.PV2_5 = self.PV2_6 = self.PV2_7 = self.PV2_8 = self.PV2_9 = self.PV2_10 = self.PV2_11 = self.PV2_12 = self.PV2_13 = self.PV2_14 = self.PV2_15 = self.PV2_16 = self.PV2_17 = self.PV2_18 = self.PV2_19 = self.PV2_20 = self.PV2_21 = self.PV2_22 = self.PV2_23 = self.PV2_24 = self.PV2_25 = self.PV2_26 = self.PV2_27 = self.PV2_28 = self.PV2_29 = self.PV2_30 = self.PV2_31 = self.PV2_32 = self.PV2_33 = self.PV2_34 = self.PV2_35 = self.PV2_36 = self.PV2_37 = self.PV2_38 = self.PV2_39 = 0.0
      self.distortion_params_are_set = False
    else:
      [self.PV1_0, self.PV1_1, self.PV1_2, self.PV1_3, self.PV1_4, self.PV1_5, self.PV1_6, self.PV1_7, self.PV1_8, self.PV1_9, self.PV1_10, self.PV1_11, self.PV1_12, self.PV1_13, self.PV1_14, self.PV1_15, self.PV1_16, self.PV1_17, self.PV1_18, self.PV1_19, self.PV1_20, self.PV1_21, self.PV1_22, self.PV1_23, self.PV1_24, self.PV1_25, self.PV1_26, self.PV1_27, self.PV1_28, self.PV1_29, self.PV1_30, self.PV1_31, self.PV1_32, self.PV1_33, self.PV1_34, self.PV1_35, self.PV1_36, self.PV1_37, self.PV1_38, self.PV1_39, self.PV2_0, self.PV2_1, self.PV2_2, self.PV2_3, self.PV2_4, self.PV2_5, self.PV2_6, self.PV2_7, self.PV2_8, self.PV2_9, self.PV2_10, self.PV2_11, self.PV2_12, self.PV2_13, self.PV2_14, self.PV2_15, self.PV2_16, self.PV2_17, self.PV2_18, self.PV2_19, self.PV2_20, self.PV2_21, self.PV2_22, self.PV2_23, self.PV2_24, self.PV2_25, self.PV2_26, self.PV2_27, self.PV2_28, self.PV2_29, self.PV2_30, self.PV2_31, self.PV2_32, self.PV2_33, self.PV2_34, self.PV2_35, self.PV2_36, self.PV2_37, self.PV2_38, self.PV2_39] = value
      self.distortion_params_are_set = True

  @property
  def hdul(self):
    if not self.distortion_params_are_set:
      return None
    header = fits.Header()
    header['WCSDIM'] = 2
    header['CTYPE1'] = 'RA---TPV'
    header['CTYPE2'] = 'DEC--TPV'
    header['CRVAL1'] = self.CRVAL1
    header['CRVAL2'] = self.CRVAL2
    header['CRPIX1'] = self.CRPIX1
    header['CRPIX2'] = self.CRPIX2
    header['CD1_1'] = self.CD1_1
    header['CD1_2'] = self.CD1_2
    header['CD2_1'] = self.CD2_1
    header['CD2_2'] = self.CD2_2
    header['CUNIT1'] = 'deg     '
    header['CUNIT2'] = 'deg     '
    header['PV1_0'] = self.PV1_0
    header['PV1_1'] = self.PV1_1
    header['PV1_2'] = self.PV1_2
    header['PV1_3'] = self.PV1_3
    header['PV1_4'] = self.PV1_4
    header['PV1_5'] = self.PV1_5
    header['PV1_6'] = self.PV1_6
    header['PV1_7'] = self.PV1_7
    header['PV1_8'] = self.PV1_8
    header['PV1_9'] = self.PV1_9
    header['PV1_10'] = self.PV1_10
    header['PV1_11'] = self.PV1_11
    header['PV1_12'] = self.PV1_12
    header['PV1_13'] = self.PV1_13
    header['PV1_14'] = self.PV1_14
    header['PV1_15'] = self.PV1_15
    header['PV1_16'] = self.PV1_16
    header['PV1_17'] = self.PV1_17
    header['PV1_18'] = self.PV1_18
    header['PV1_19'] = self.PV1_19
    header['PV1_20'] = self.PV1_20
    header['PV1_21'] = self.PV1_21
    header['PV1_22'] = self.PV1_22
    header['PV1_23'] = self.PV1_23
    header['PV1_24'] = self.PV1_24
    header['PV1_25'] = self.PV1_25
    header['PV1_26'] = self.PV1_26
    header['PV1_27'] = self.PV1_27
    header['PV1_28'] = self.PV1_28
    header['PV1_29'] = self.PV1_29
    header['PV1_30'] = self.PV1_30
    header['PV1_31'] = self.PV1_31
    header['PV1_32'] = self.PV1_32
    header['PV1_33'] = self.PV1_33
    header['PV1_34'] = self.PV1_34
    header['PV1_35'] = self.PV1_35
    header['PV1_36'] = self.PV1_36
    header['PV1_37'] = self.PV1_37
    header['PV1_38'] = self.PV1_38
    header['PV1_39'] = self.PV1_39
    header['PV2_0'] = self.PV2_0
    header['PV2_1'] = self.PV2_1
    header['PV2_2'] = self.PV2_2
    header['PV2_3'] = self.PV2_3
    header['PV2_4'] = self.PV2_4
    header['PV2_5'] = self.PV2_5
    header['PV2_6'] = self.PV2_6
    header['PV2_7'] = self.PV2_7
    header['PV2_8'] = self.PV2_8
    header['PV2_9'] = self.PV2_9
    header['PV2_10'] = self.PV2_10
    header['PV2_11'] = self.PV2_11
    header['PV2_12'] = self.PV2_12
    header['PV2_13'] = self.PV2_13
    header['PV2_14'] = self.PV2_14
    header['PV2_15'] = self.PV2_15
    header['PV2_16'] = self.PV2_16
    header['PV2_17'] = self.PV2_17
    header['PV2_18'] = self.PV2_18
    header['PV2_19'] = self.PV2_19
    header['PV2_20'] = self.PV2_20
    header['PV2_21'] = self.PV2_21
    header['PV2_22'] = self.PV2_22
    header['PV2_23'] = self.PV2_23
    header['PV2_24'] = self.PV2_24
    header['PV2_25'] = self.PV2_25
    header['PV2_26'] = self.PV2_26
    header['PV2_27'] = self.PV2_27
    header['PV2_28'] = self.PV2_28
    header['PV2_29'] = self.PV2_29
    header['PV2_30'] = self.PV2_30
    header['PV2_31'] = self.PV2_31
    header['PV2_32'] = self.PV2_32
    header['PV2_33'] = self.PV2_33
    header['PV2_34'] = self.PV2_34
    header['PV2_35'] = self.PV2_35
    header['PV2_36'] = self.PV2_36
    header['PV2_37'] = self.PV2_37
    header['PV2_38'] = self.PV2_38
    header['PV2_39'] = self.PV2_39
    primary_hdu = fits.PrimaryHDU(header=header)
    return fits.HDUList([primary_hdu])

  def i_j_to_x_y(self, i, j):
    return do_i_j_to_x_y(i, j, self.affine_transform)

  def x_y_to_xx_yy(self, x, y):
    return do_x_y_to_xx_yy(x, y, self.distortion_params)

  def i_j_to_xx_yy(self, i, j):
    (x, y) = self.i_j_to_x_y(i, j)
    (xx, yy) = self.x_y_to_xx_yy(x, y)
    return (xx, yy)

  def x_y_to_i_j(self, x, y):
    denom = self.CD1_1*self.CD2_2 - self.CD1_2*self.CD2_1
    i = (self.CD2_2*x - self.CD1_2*y)/denom + self.CRPIX1
    j = (self.CD1_1*y - self.CD2_1*x)/denom + self.CRPIX2
    return (i, j)

  def xx_yy_to_x_y(self, xx, yy):
    ITERATIONS = 3

    def dxxdx_inv(x, y):
      xx = x*x
      yy = y*y
      xxx = x*xx
      yyy = y*yy
      xxxx = x*xxx
      yyyy = y*yyy
      xxxxx = x*xxxx
      yyyyy = y*yyyy
      xxxxxx = x*xxxxx
      yyyyyy = y*yyyyy
      r = sqrt(xx + yy)
      rr = r*r
      rrr = r*rr
      rrrrr = rr*rrr
      xy = yx = x*y
      xxy = yxx = xx*y
      xyy = yyx = x*yy
      xxxy = yxxx = xxx*y
      xxyy = yyxx = xx*yy
      xyyy = yyyx = x*yyy
      xxxxy = yxxxx = xxxx*y
      xxxyy = yyxxx = xxx*yy
      xxyyy = yyyxx = xx*yyy
      xyyyy = yyyyx = x*yyyy
      xxxxxy = yxxxxx = xxxxx*y
      xxxxyy = yyxxxx = xxxx*yy
      xxxyyy = yyyxxx = xxx*yyy
      xxyyyy = yyyyxx = xx*yyyy
      xyyyyy = yyyyyx = x*yyyyy
      x_r = x/r
      xr = x*r
      xrrr = x*rrr
      xrrrrr = x*rrrrr
      dxxdx = self.PV1_1 + self.PV1_3*x_r + 2*self.PV1_4*x + self.PV1_5*y + 3*self.PV1_7*xx + 2*self.PV1_8*xy + self.PV1_9*yy + 3*self.PV1_11*xr + 4*self.PV1_12*xxx + 3*self.PV1_13*xxy + 2*self.PV1_14*xyy + self.PV1_15*yyy + 5*self.PV1_17*xxxx + 4*self.PV1_18*xxxy + 3*self.PV1_19*xxyy + 2*self.PV1_20*xyyy + self.PV1_21*yyyy + 5*self.PV1_23*xrrr + 6*self.PV1_24*xxxxx + 5*self.PV1_25*xxxxy + 4*self.PV1_26*xxxyy + 3*self.PV1_27*xxyyy + 2*self.PV1_28*xyyyy + self.PV1_29*yyyyy + 7*self.PV1_31*xxxxxx + 6*self.PV1_32*xxxxxy + 5*self.PV1_33*xxxxyy + 4*self.PV1_34*xxxyyy + 3*self.PV1_35*xxyyyy + 2*self.PV1_36*xyyyyy + self.PV1_37*yyyyyy + 7*self.PV1_39*xrrrrr
      return 1/dxxdx

    def dyydy_inv(x, y):
      xx = x*x
      yy = y*y
      xxx = x*xx
      yyy = y*yy
      xxxx = x*xxx
      yyyy = y*yyy
      xxxxx = x*xxxx
      yyyyy = y*yyyy
      xxxxxx = x*xxxxx
      yyyyyy = y*yyyyy
      r = sqrt(xx + yy)
      rr = r*r
      rrr = r*rr
      rrrrr = rr*rrr
      xy = yx = x*y
      xxy = yxx = xx*y
      xyy = yyx = x*yy
      xxxy = yxxx = xxx*y
      xxyy = yyxx = xx*yy
      xyyy = yyyx = x*yyy
      xxxxy = yxxxx = xxxx*y
      xxxyy = yyxxx = xxx*yy
      xxyyy = yyyxx = xx*yyy
      xyyyy = yyyyx = x*yyyy
      xxxxxy = yxxxxx = xxxxx*y
      xxxxyy = yyxxxx = xxxx*yy
      xxxyyy = yyyxxx = xxx*yyy
      xxyyyy = yyyyxx = xx*yyyy
      xyyyyy = yyyyyx = x*yyyyy
      y_r = y/r
      yr = y*r
      yrrr = y*rrr
      yrrrrr = y*rrrrr
      dyydy = self.PV2_1 + self.PV2_3*y_r + 2*self.PV2_4*y + self.PV2_5*x + 3*self.PV2_7*yy + 2*self.PV2_8*yx + self.PV2_9*xx + 3*self.PV2_11*yr + 4*self.PV2_12*yyy + 3*self.PV2_13*yyx + 2*self.PV2_14*yxx + self.PV2_15*xxx + 5*self.PV2_17*yyyy + 4*self.PV2_18*yyyx + 3*self.PV2_19*yyxx + 2*self.PV2_20*yxxx + self.PV2_21*xxxx + 5*self.PV2_23*yrrr + 6*self.PV2_24*yyyyy + 5*self.PV2_25*yyyyx + 4*self.PV2_26*yyyxx + 3*self.PV2_27*yyxxx + 2*self.PV2_28*yxxxx + self.PV2_29*xxxxx + 7*self.PV2_31*yyyyyy + 6*self.PV2_32*yyyyyx + 5*self.PV2_33*yyyyxx + 4*self.PV2_34*yyyxxx + 3*self.PV2_35*yyxxxx + 2*self.PV2_36*yxxxxx + self.PV2_37*xxxxxx + 7*self.PV2_39*yrrrrr
      return 1/dyydy

    (xx_, yy_) = self.x_y_to_xx_yy(xx, yy)
    x = xx + dxxdx_inv(xx, yy)*(xx - xx_)
    y = yy + dyydy_inv(xx, yy)*(yy - yy_)
    for _ in range(ITERATIONS):
      (xx_, yy_) = self.x_y_to_xx_yy(x, y)
      x = x + dxxdx_inv(x, y)*(xx - xx_)
      y = y + dyydy_inv(x, y)*(yy - yy_)
    return(x, y)

  def chisqr(self, params, i, j, x, y, var_0ij=1, var_i=None, var_j=None, var_ij=None, a2=None, var_0=None):
    [self.CD1_1, self.CD1_2, self.CD2_1, self.CD2_2, self.PV1_0, self.PV1_1, self.PV1_2, self.PV1_3, self.PV1_4, self.PV1_5, self.PV1_6, self.PV1_7, self.PV1_8, self.PV1_9, self.PV1_10, self.PV1_11, self.PV1_12, self.PV1_13, self.PV1_14, self.PV1_15, self.PV1_16, self.PV1_17, self.PV1_18, self.PV1_19, self.PV1_20, self.PV1_21, self.PV1_22, self.PV1_23, self.PV1_24, self.PV1_25, self.PV1_26, self.PV1_27, self.PV1_28, self.PV1_29, self.PV1_30, self.PV1_31, self.PV1_32, self.PV1_33, self.PV1_34, self.PV1_35, self.PV1_36, self.PV1_37, self.PV1_38, self.PV1_39, self.PV2_0, self.PV2_1, self.PV2_2, self.PV2_3, self.PV2_4, self.PV2_5, self.PV2_6, self.PV2_7, self.PV2_8, self.PV2_9, self.PV2_10, self.PV2_11, self.PV2_12, self.PV2_13, self.PV2_14, self.PV2_15, self.PV2_16, self.PV2_17, self.PV2_18, self.PV2_19, self.PV2_20, self.PV2_21, self.PV2_22, self.PV2_23, self.PV2_24, self.PV2_25, self.PV2_26, self.PV2_27, self.PV2_28, self.PV2_29, self.PV2_30, self.PV2_31, self.PV2_32, self.PV2_33, self.PV2_34, self.PV2_35, self.PV2_36, self.PV2_37, self.PV2_38, self.PV2_39] = params
    (xx, yy) = self.i_j_to_xx_yy(i, j)
    if a2 is None:
      result = numpy.sum((x - xx)**2 + (y - yy)**2)
      if not var_0ij == 1:
        result /= var_0ij
    else:
      #print(f" Model Params a:{a} , sigma_0:{sigma_0}")
      dx = x - xx
      dx2 = dx*dx
      dy = y - yy
      dy2 = dy*dy
      c11_11 = self.CD1_1*self.CD1_1
      c12_12 = self.CD1_2*self.CD1_2
      c11_12 = self.CD1_1*self.CD1_2
      var_x = c11_11*var_i + c12_12*var_j + 2*c11_12*var_ij
      var_y = c12_12*var_i + c11_11*var_j + 2*c11_12*var_ij

      var = (var_x + var_y)/2
      var_r = var_0+a2*var

      result = numpy.sum((dx2+dy2)/var_r)

    self.iteration  += 1
    if self.iteration % 1000 == 0:
      if var_0ij == 1:
        print(f"{self.iteration} rms (arcsec): {sqrt(result/len(i))*3600}", flush=True)
      else:
        print(f"{self.iteration} chisqr: {result}", flush=True)
    return result

  def resid_rms(self, x, y, xx, yy): # residuals in units of "sigma"
    return numpy.sqrt(numpy.square(x - xx) + numpy.square(y - yy))*3600/self.rms_arcsec

  def sigma_model(self, x, y, xx, yy, var_i, var_j, var_ij, a2, var_0, var_r=None):
      #print("Residual Model Fitting")
    if a2 is None:
       print("a, sigma_0 not successfully carried over")
    dx = x - xx
    dx2 = dx*dx
    dy = y - yy
    dy2 = dy*dy
    c11_11 = self.CD1_1*self.CD1_1
    c12_12 = self.CD1_2*self.CD1_2
    c11_12 = self.CD1_1*self.CD1_2
    var_0xy = (c11_11 + c12_12)*self.var_0ij
    var_x = c11_11*self.var_i + c12_12*self.var_j + 2*c11_12*self.var_ij
    var_y = c12_12*self.var_i + c11_11*self.var_j + 2*c11_12*self.var_ij
    var_xy = c11_12*self.var_i + c11_12*self.var_j + (c11_11 + c12_12)*self.var_ij

    if var_r is None:
      var = (var_x + var_y)/2
      var_r = var_0 + a2*var

    residual = (dx2+dy2)/var_r
    return residual

  def wcsfit(self, i, j, x, y, var_0ij=1, var_i=None, var_j=None, var_ij=None, r=None, flux_image=None, ML_FIT = None, a2=None, var_0=None):
    CLIPPING_THRESHOLD = 3.0
    ERROR_THRESHOLD = 3.0
    ITERATIONS = 7
    LONPOLE = 180
    MAXFEV = 200000
    MAXITER = 200000
    RMS_ARCSEC_INIT = 20

    self.i = i
    self.j = j
    self.x = x
    self.y = y
    self.var_0ij = var_0ij
    self.var_i = var_i
    self.var_j = var_j
    self.var_ij = var_ij

    if self.affine_transform is None:
      raise Exception('error:  no affine transformation is set...')

    if self.distortion_params is None:
      PV1_0 = PV2_0 = 0.0
      PV1_1 = PV2_1 = 1.0
      PV1_2 = PV1_3 = PV1_4 = PV1_5 = PV1_6 = PV1_7 = PV1_8 = PV1_9 = PV1_10 = PV1_11 = PV1_12 = PV1_13 = PV1_14 = PV1_15 = PV1_16 = PV1_17 = PV1_18 = PV1_19 = PV1_20 = PV1_21 = PV1_22 = PV1_23 = PV1_24 = PV1_25 = PV1_26 = PV1_27 = PV1_28 = PV1_29 = PV1_30 = PV1_31 = PV1_32 = PV1_33 = PV1_34 = PV1_35 = PV1_36 = PV1_37 = PV1_38 = PV1_39 = PV2_2 = PV2_3 = PV2_4 = PV2_5 = PV2_6 = PV2_7 = PV2_8 = PV2_9 = PV2_10 = PV2_11 = PV2_12 = PV2_13 = PV2_14 = PV2_15 = PV2_16 = PV2_17 = PV2_18 = PV2_19 = PV2_20 = PV2_21 = PV2_22 = PV2_23 = PV2_24 = PV2_25 = PV2_26 = PV2_27 = PV2_28 = PV2_29 = PV2_30 = PV2_31 = PV2_32 = PV2_33 = PV2_34 = PV2_35 = PV2_36 = PV2_37 = PV2_38 = PV2_39 = 0.0001
      self.distortion_params = [PV1_0, PV1_1, PV1_2, PV1_3, PV1_4, PV1_5, PV1_6, PV1_7, PV1_8, PV1_9, PV1_10, PV1_11, PV1_12, PV1_13, PV1_14, PV1_15, PV1_16, PV1_17, PV1_18, PV1_19, PV1_20, PV1_21, PV1_22, PV1_23, PV1_24, PV1_25, PV1_26, PV1_27, PV1_28, PV1_29, PV1_30, PV1_31, PV1_32, PV1_33, PV1_34, PV1_35, PV1_36, PV1_37, PV1_38, PV1_39, PV2_0, PV2_1, PV2_2, PV2_3, PV2_4, PV2_5, PV2_6, PV2_7, PV2_8, PV2_9, PV2_10, PV2_11, PV2_12, PV2_13, PV2_14, PV2_15, PV2_16, PV2_17, PV2_18, PV2_19, PV2_20, PV2_21, PV2_22, PV2_23, PV2_24, PV2_25, PV2_26, PV2_27, PV2_28, PV2_29, PV2_30, PV2_31, PV2_32, PV2_33, PV2_34, PV2_35, PV2_36, PV2_37, PV2_38, PV2_39]

    params = [self.CD1_1, self.CD1_2, self.CD2_1, self.CD2_2, self.PV1_0, self.PV1_1, self.PV1_2, self.PV1_3, self.PV1_4, self.PV1_5, self.PV1_6, self.PV1_7, self.PV1_8, self.PV1_9, self.PV1_10, self.PV1_11, self.PV1_12, self.PV1_13, self.PV1_14, self.PV1_15, self.PV1_16, self.PV1_17, self.PV1_18, self.PV1_19, self.PV1_20, self.PV1_21, self.PV1_22, self.PV1_23, self.PV1_24, self.PV1_25, self.PV1_26, self.PV1_27, self.PV1_28, self.PV1_29, self.PV1_30, self.PV1_31, self.PV1_32, self.PV1_33, self.PV1_34, self.PV1_35, self.PV1_36, self.PV1_37, self.PV1_38, self.PV1_39, self.PV2_0, self.PV2_1, self.PV2_2, self.PV2_3, self.PV2_4, self.PV2_5, self.PV2_6, self.PV2_7, self.PV2_8, self.PV2_9, self.PV2_10, self.PV2_11, self.PV2_12, self.PV2_13, self.PV2_14, self.PV2_15, self.PV2_16, self.PV2_17, self.PV2_18, self.PV2_19, self.PV2_20, self.PV2_21, self.PV2_22, self.PV2_23, self.PV2_24, self.PV2_25, self.PV2_26, self.PV2_27, self.PV2_28, self.PV2_29, self.PV2_30, self.PV2_31, self.PV2_32, self.PV2_33, self.PV2_34, self.PV2_35, self.PV2_36, self.PV2_37, self.PV2_38, self.PV2_39]
    self.nsources_total = len(self.i)

    for iteration in range(ITERATIONS):
      n_sources = len(self.i)
      print('number of sources before clipping:', n_sources, flush=True)
      if self.rms_arcsec is None:
        self.rms_arcsec = RMS_ARCSEC_INIT
        (xx, yy) = self.i_j_to_x_y(self.i, self.j)
      else:
        (xx, yy) = self.i_j_to_xx_yy(self.i, self.j)

      if ML_FIT or a2 is None:
        rms = self.resid_rms(self.x, self.y, xx, yy)
        print(f" Average rms(x,y): {numpy.average(rms*self.rms_arcsec)}")
        self.i = self.i[rms < CLIPPING_THRESHOLD]
        self.j = self.j[rms < CLIPPING_THRESHOLD]
        self.x = self.x[rms < CLIPPING_THRESHOLD]
        self.y = self.y[rms < CLIPPING_THRESHOLD]
        xx = xx[rms < CLIPPING_THRESHOLD]
        yy = yy[rms < CLIPPING_THRESHOLD]
        if self.var_i is not None:
          self.var_i = self.var_i[rms < CLIPPING_THRESHOLD]
          self.var_j = self.var_j[rms < CLIPPING_THRESHOLD]
          self.var_ij = self.var_ij[rms < CLIPPING_THRESHOLD]

      if ML_FIT or a2 is not None:
        if ML_FIT:
          (a2,var_0,var_r) = self.like_fit(self.x,self.y,xx,yy,self.var_i,self.var_j,self.var_ij)
          resid = self.sigma_model(self.x, self.y, xx, yy,self.var_i,self.var_j,self.var_ij,a2,var_0,var_r)
        if ML_FIT is None and a2 is not None:
          if iteration == 0:
            print(f"Params passed through TAG_B to sigma_model a:{sqrt(a2)} sigma_0:{sqrt(var_0)}")
            #a, sigma_0 are passed on for rounds of residual model rejections, and should be same value for use in chi square fitting of the whole list
          resid = self.sigma_model(self.x, self.y, xx, yy, self.var_i, self.var_j, self.var_ij,a2,var_0)
        print(f" Mean Resid(var model): {numpy.average(resid)}")
        self.i = self.i[resid < ERROR_THRESHOLD]
        self.j = self.j[resid < ERROR_THRESHOLD]
        self.x = self.x[resid < ERROR_THRESHOLD]
        self.y = self.y[resid < ERROR_THRESHOLD]
        xx = xx[resid < ERROR_THRESHOLD]
        yy = yy[resid < ERROR_THRESHOLD]
        self.var_i = self.var_i[resid < ERROR_THRESHOLD]
        self.var_j = self.var_j[resid < ERROR_THRESHOLD]
        self.var_ij = self.var_ij[resid < ERROR_THRESHOLD]

      if len(self.i) == n_sources:
        print('no sources to be clipped...', flush=True)
        if iteration > 0:
          break
      else:
        print('number of sources after clipping:', len(self.i), flush=True)

      result = minimize(self.chisqr, params, args=(self.i, self.j, self.x, self.y, self.var_0ij, self.var_i, self.var_j, self.var_ij,a2,var_0), method='powell', options={'maxfev': MAXFEV, 'maxiter': MAXITER})
      chisqr0 = self.chisqr(result.x, self.i, self.j, self.x, self.y)
      self.rms_arcsec = sqrt(chisqr0/len(self.i))*3600
      print('rms (arcsec):', self.rms_arcsec, flush=True)
      if a2 is not None:
        chisqr0 = self.chisqr(result.x, self.i, self.j, self.x, self.y, self.var_0ij, self.var_i, self.var_j, self.var_ij,a2=a2,var_0=var_0)
        self.chisqr_per_ndof = chisqr0/(len(self.i) - self.nparam)
        print('chisqr:', chisqr0, flush=True)
        print('chisqr/ndof:', self.chisqr_per_ndof, flush=True)
      self.nsources_used = len(self.i)
      params = result.x

    x0 = self.CD1_1*(self.nx//2 - self.CRPIX1) + self.CD1_2*(self.ny//2 - self.CRPIX2)
    y0 = self.CD2_1*(self.nx//2 - self.CRPIX1) + self.CD2_2*(self.ny//2 - self.CRPIX2)
    R0 = sqrt(x0*x0 + y0*y0)
    theta0 = deg(atan(180.0/pi/R0))
    phi0 = deg(atan2(x0, -y0))
    ra_deg0 = self.CRVAL1 + deg(atan2(-cos(rad(theta0))*sin(rad(phi0 - LONPOLE)), sin(rad(theta0))*cos(rad(self.CRVAL2)) - cos(rad(theta0))*sin(rad(self.CRVAL2))*cos(rad(phi0 - LONPOLE))))
    dec_deg0 = deg(asin(sin(rad(theta0))*sin(rad(self.CRVAL2)) + cos(rad(theta0))*cos(rad(self.CRVAL2))*cos(rad(phi0 - LONPOLE))))

    s = sqrt(self.CD1_1*self.CD1_1 + self.CD1_2*self.CD1_2)
    alpha = deg(asin(self.CD1_2/s))

    field_size_x_deg = s*self.nx
    field_size_y_deg = s*self.ny
    if alpha > 0:
      rotation_angle_deg = 180 - alpha
    else:
      rotation_angle_deg = -180 - alpha

    print('CRVAL1:', self.CRVAL1, flush=True)
    print('CRVAL2:', self.CRVAL2, flush=True)
    print('CRPIX1:', self.CRPIX1, flush=True)
    print('CRPIX2:', self.CRPIX2, flush=True)
    print('field size x (deg):', field_size_x_deg, flush=True)
    print('field size y (deg):', field_size_y_deg, flush=True)
    print('rotation angle (deg):', rotation_angle_deg, flush=True)
    print('rms (arcsec):', self.rms_arcsec, flush=True)
    if self.chisqr_per_ndof is not None:
      print('chisqr:', chisqr0, flush=True)
      print('chisqr/ndof:', self.chisqr_per_ndof, flush=True)
    print('number of sources total:', self.nsources_total, flush=True)
    print('number of sources used:', self.nsources_used, flush=True)
    print()

    self.ra_deg0 = ra_deg0
    self.dec_deg0 = dec_deg0
    self.field_size_x_deg = field_size_x_deg
    self.field_size_y_deg = field_size_y_deg
    self.rotation_angle_deg = rotation_angle_deg

    if ML_FIT:
      return (chisqr0, a2, var_0)
    else:
      return chisqr0

  def like_fit(self, x, y, xx, yy, var_i, var_j, var_ij):
    SCALE = 1
    INIT_a = 0
    INIT_sigma0 = 1

    print('Running ML fitting procedure...', flush = True)
    di = xx-x
    dj = yy-y
    c11_11 = self.CD1_1*self.CD1_1
    c12_12 = self.CD1_2*self.CD1_2
    c11_12 = self.CD1_1*self.CD1_2
    var_0xy = (c11_11 + c12_12)*self.var_0ij
    var_x = c11_11*self.var_i + c12_12*self.var_j + 2*c11_12*self.var_ij
    var_y = c12_12*self.var_i + c11_11*self.var_j + 2*c11_12*self.var_ij
    var_xy = c11_12*self.var_i + c11_12*self.var_j + (c11_11 + c12_12)*self.var_ij
    #print(f"Sanity Check: sep sources:{len(ii),len(jj),len(var_x),len(var_y)} gaia targs:{len(i),len(j)} ")
    res = numpy.array(sqrt(square(di)+square(dj))*SCALE)
    sigma = numpy.array(sqrt((var_x+var_y)/2)*SCALE**2)
    res2 = res**2
    var = sigma**2

    def log_likely(parameters, res2, var):
      sigma_0 = parameters[0]
      a = parameters[1]
      s = 0
      for res2_,var_ in zip(res2,var):
        s += math.log(sigma_0**2+a**2*var_,math.e) + (res2_)/(sigma_0**2+a**2*var_)
      return(s)

    parameters = [INIT_a, INIT_sigma0]
    min_estimates = minimize(log_likely, parameters, args=(res2, var), method='Nelder-Mead')
    sigma_0 = min_estimates.x[0]
    a = min_estimates.x[1]
    print('Min. Likelihood est using Nelder-Mead (x,y): a = %s , sigma_0 = %s' % (a,sigma_0))
    a2 = a**2
    var_0 = sigma_0**2
    #sigma_r = sqrt(var_0+a2*var)
    var_r = var_0 + a2*var

    return (a2, var_0, var_r)

