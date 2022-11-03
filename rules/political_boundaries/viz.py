import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point


def draw_counties(tracts):
    counties = gpd.GeoSeries(tracts.groupby('COUNTYFP').apply(lambda x: x.geometry.unary_union))
    ax = counties.plot(figsize=(10, 10), color='none', edgecolor='black')
    tracts.plot(ax=ax, color='none', edgecolor='red', lw=.2)
    return counties


def tract_geometry_translation(tracts, scaled_counties, seed):
    def scale_county(tract_series, scaled_county_centroids):
        centroid = scaled_county_centroids.loc[tract_series.name]
        return tract_series.translate(xoff=centroid.x, yoff=centroid.y)

    county_fips_to_ix = {fips: ix for ix, fips
                         in enumerate(sorted(tracts.COUNTYFP.unique()))}

    center = tracts.loc[seed].geometry.centroid

    x_offset = -center.x
    y_offset = -center.y

    centered_tracts = gpd.GeoDataFrame({
        'geometry': tracts.translate(x_offset, y_offset),
        'COUNTYFP': tracts.COUNTYFP.apply(lambda x: county_fips_to_ix[x]),
    }, index=tracts.index)

    scaled_tracts = centered_tracts.groupby('COUNTYFP').apply(
        lambda x: scale_county(x, scaled_counties)).reset_index(level=0, drop=True)

    return scaled_tracts