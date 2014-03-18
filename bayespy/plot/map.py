######################################################################
# Copyright (C) 2014 Jaakko Luttinen
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
######################################################################

######################################################################
# This file is part of BayesPy.
#
# BayesPy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# BayesPy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BayesPy.  If not, see <http://www.gnu.org/licenses/>.
######################################################################

"""
A module for plotting on maps.

The core functionality is from Basemap package so you need to install that in
order to use this module.
"""

import numpy as np
import matplotlib.pyplot as plt

from bayespy.utils import utils

from mpl_toolkits.basemap import Basemap

def _draw_map(m, fill=True, parallels=20, meridians=20):
    # draw coasts
    m.drawcoastlines(linewidth=0.5)
    
    # fill continents and background
    if fill:
        m.drawmapboundary(fill_color='aqua')
        m.fillcontinents(color='coral',lake_color='aqua', zorder=0)
    
    if parallels:
        m.drawparallels(np.arange(-80, 81, parallels))
    if meridians:
        m.drawmeridians(np.arange(-180, 180, meridians))
    
    return m

def draw_region(lats, lons,
                projection='gall',
                resolution='c',
                **kwargs):

    plt.cla()
    m = Basemap(llcrnrlon=lons[0],
                llcrnrlat=lats[0],
                urcrnrlon=lons[1],
                urcrnrlat=lats[1],
                projection=projection,
                resolution=resolution)
    return _draw_map(m, **kwargs)
                

def draw_area(lat, lon, width, height,
              projection='aea', 
              resolution='c',
              **kwargs):
    
    """
    Consider these projections: 
      cass = equidistant "Cassini Projection"
      aeqd = "Azimuthal Equidistant Projection"
      aea = equal area "Albers Equal Area Projection"
      lcc = conformal "Lambert Conformal Projection"
    """

    plt.cla()
    m = Basemap(width=width,
                height=height,
                lat_0=lat,
                lon_0=lon,
                projection=projection,
                resolution=resolution)
    return _draw_map(m, **kwargs)
    

def draw_globe(projection='robin', # cea
               resolution='c',
               lon=0,
               **kwargs):
    
    plt.cla()
    m = Basemap(lon_0=lon,
                projection=projection,
                resolution=resolution)
    return _draw_map(m, **kwargs)


def plot(m, lat, lon, *args, **kwargs):

    (x, y) = m(lon, lat)
    return m.plot(x, y, *args, **kwargs)

def pcolormesh(m, lats, lons, y, shading='flat', cmap=plt.cm.RdBu_r, 
               **kwargs):

    if np.ndim(lons) < 2 or np.ndim(lats) < 2:
        (lons, lats) = np.meshgrid(lons,lats)
        
    return m.pcolormesh(np.asanyarray(lons), 
                        np.asanyarray(lats),
                        np.asanyarray(y),
                        shading=shading,
                        cmap=cmap,
                        latlon=True,
                        **kwargs)

def _interpolate(D, Y, Dh, method='linear'):
    if method == 'linear':
        mu = np.mean(Y, axis=-1, keepdims=True)
        Y = Y - mu
        return np.dot(Dh, np.linalg.solve(D, Y)) + mu
    elif method == 'nearest':
        ind = np.argmin(Dh, axis=-1)
        return Y[...,ind]
    else:
        raise ValueError("Unknown interpolation method %s requested" % (method))

def interpolate(m, lat, lon, y, lats, lons, method='linear', **kwargs):

    (lons, lats) = np.meshgrid(lons,lats)

    D = utils.dist_haversine([lat, lon], [lat, lon])
    Dh = utils.dist_haversine([lats.flatten(), lons.flatten()],
                              [lat, lon])

    yh = _interpolate(D, y, Dh, method=method)
    yh = np.reshape(yh, np.shape(lons))

    return pcolormesh(m, lats, lons, yh, **kwargs)
