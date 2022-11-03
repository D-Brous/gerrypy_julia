"""This script downloads shapefiles from all 50 states and American territories
 from the TIGER census geospatial database. Shapefiles are saved in a folder
based on their respective state codes as defined by:
https://www2.census.gov/geo/docs/reference/state.txt?#

Simply run <python get_shapefile.py> in the directory you wish to store
these shapefiles and change the year as appropriate.

Saves data in shapes/STATE_year."""

import sys
sys.path.append('../gerrypy')

import os
from io import BytesIO
from ftplib import FTP
from zipfile import ZipFile
from urllib.request import urlopen
import constants

def download_vtds(states=None, year=constants.ACS_BASE_YEAR):
    """Download census VTDs for states.
    
    states: str list - state abbreviations to download
    year : str - year for data to be collected"""
    vtds_dir = constants.CENSUS_SHAPE_PATH_2020

    year=str(year)
    TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER%sPL/LAYER/VTD/%s/" %(year,year)
        #NOTE: this url will only work from 2020 onward- the data is stored differently for prev years

    if not states:
        states = [abbr for _, abbr, _ in constants.STATE_IDS]

    states_fips = [constants.ABBREV_DICT[abbrev][constants.FIPS_IX]
                   for abbrev in states]

    try:
        os.mkdir(vtds_dir)
    except FileExistsError:
        pass

    # Check if already have
    downloaded_files = os.listdir(vtds_dir)
    cached = [state for state, year in downloaded_files if year == year]
    states = [state for state in states if state not in cached]
    if not states:
        return 

    # Login to census FTP server
    ftp = FTP("ftp.census.gov")
    ftp.login()
    ftp.cwd("/geo/tiger/TIGER%sPL/LAYER/VTD/%s/" % (year, year))
    print("Successfully made FTP connection")

    # Download and extract all files for states in states
    for file_name in ftp.nlst():
        state_code = file_name.split("_")[2] #this includes state and county codes, but county codes will automatically be ignored
        try:
            state_abbr = constants.FIPS_DICT[state_code][constants.ABBREV_IX]
        except KeyError:
            continue

        if state_code not in states_fips:
            print("%s cached" % state_abbr)

        if state_code in states_fips:
            dir_name = os.path.join(vtds_dir, "_".join([state_abbr]))
            try:
                os.mkdir(dir_name)
            except FileExistsError:
                continue

            resp = urlopen(TIGER_URL + file_name)
            zipfile = ZipFile(BytesIO(resp.read()))

            zipfile.extractall(dir_name)
            print("Successfully downloaded and extracted state", state_abbr) 

def download_state_shapes(states=None, year=constants.ACS_BASE_YEAR):
    """Download census shapefiles for states.
    
    states: str list - state abbreviations to download
    year : str - year for data to be collected"""
    shapes_dir = constants.CENSUS_SHAPE_PATH_2020
    year = str(year)
    TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER%s/TRACT/" % year

    if not states:
        states = [abbr for _, abbr, _ in constants.STATE_IDS]

    states_fips = [constants.ABBREV_DICT[abbrev][constants.FIPS_IX]
                   for abbrev in states]

    year = str(year)

    try:
        os.mkdir(shapes_dir)
    except FileExistsError:
        pass

    # Check if already have
    downloaded_files = os.listdir(shapes_dir)
    downloaded_files = [f.split("_") for f in downloaded_files]
    cached = [state for state, year in downloaded_files if year == year]
    states = [state for state in states if state not in cached]
    if not states:
        return

    # Login to census FTP server
    ftp = FTP("ftp.census.gov")
    ftp.login()
    ftp.cwd("/geo/tiger/TIGER%s/TRACT/" % year)
    print("Successfully made FTP connection")

    # Download and extract all files for states in states
    for file_name in ftp.nlst():
        state_code = file_name.split("_")[2]
        try:
            state_abbr = constants.FIPS_DICT[state_code][constants.ABBREV_IX]
        except KeyError:
            continue

        if state_code not in states_fips:
            print("%s cached" % state_abbr)

        dir_name = os.path.join(shapes_dir, "_".join([state_abbr, year]))
        try:
            os.mkdir(dir_name)
        except FileExistsError:
            continue

        resp = urlopen(TIGER_URL + file_name)
        zipfile = ZipFile(BytesIO(resp.read()))

        zipfile.extractall(dir_name)
        print("Successfully downloaded and extracted state", state_abbr)


def download_state_places_shapes(states=None, year=constants.ACS_BASE_YEAR):
    """Download census shapefiles for the census places dataset.

    states: str list - state abbreviations to download
    year : str - year for data to be collected"""
    shapes_dir = constants.PLACES_PATH
    year = str(year)
    TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER%s/PLACE/" % year

    if not states:
        states = [abbr for _, abbr, _ in constants.STATE_IDS]

    states_fips = [constants.ABBREV_DICT[abbrev][constants.FIPS_IX]
                   for abbrev in states]

    year = str(year)

    try:
        os.mkdir(shapes_dir)
    except FileExistsError:
        pass

    # Check if already have
    downloaded_files = os.listdir(shapes_dir)
    downloaded_files = [f.split("_") for f in downloaded_files]
    cached = [state for state, year in downloaded_files if year == year]
    states = [state for state in states if state not in cached]
    if not states:
        return

    # Login to census FTP server
    ftp = FTP("ftp.census.gov")
    ftp.login()
    ftp.cwd("/geo/tiger/TIGER%s/PLACE/" % year)
    print("Successfully made FTP connection")

    # Download and extract all files for states in states
    for file_name in ftp.nlst():
        state_code = file_name.split("_")[2]
        try:
            state_abbr = constants.FIPS_DICT[state_code][constants.ABBREV_IX]
        except KeyError:
            continue

        if state_code not in states_fips:
            print("%s cached" % state_abbr)

        dir_name = os.path.join(shapes_dir, "_".join([state_abbr, year]))
        try:
            os.mkdir(dir_name)
        except FileExistsError:
            continue

        resp = urlopen(TIGER_URL + file_name)
        zipfile = ZipFile(BytesIO(resp.read()))

        zipfile.extractall(dir_name)
        print("Successfully downloaded and extracted state", state_abbr)


def download_district_shapes(year):
    """Download census shapefiles for districts from 2010 - 2019"""

    district_shapes_dir = os.path.join(constants.GERRYPY_BASE_PATH,
                                       "data", "district_shapes")
    year = str(year)
    TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER%s/CD/" % year

    try:
        os.mkdir(district_shapes_dir)
    except FileExistsError:
        pass
    
    # Check if already have
    downloaded_files = os.listdir(district_shapes_dir)
    for fname in downloaded_files:
        if year in fname:
            print("You've already downloaded data for year " + year)
            return
    
    # Login to census FTP server
    ftp = FTP("ftp.census.gov")
    ftp.login()
    ftp.cwd("/geo/tiger/TIGER%s/CD/" % year)
    print("Successfully made FTP connection")
    
    # Download and extract all files for states in states
    for file_name in ftp.nlst():

        dir_name = os.path.join(district_shapes_dir, "_".join(['cd', year]))
        try:
            os.mkdir(dir_name)
        except FileExistsError:
            continue
        
        resp = urlopen(TIGER_URL + file_name)
        zipfile = ZipFile(BytesIO(resp.read()))

        zipfile.extractall(dir_name)
        print("Successfully downloaded and extracted congressional data for ", year)


def download_all_district_shapes():
    for i in range(2012, 2020, 2):
        download_district_shapes(i)


if __name__ == "__main__":
    download_vtds(states=['NC'], year=2020)
