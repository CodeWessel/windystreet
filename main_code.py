# -*- coding: utf-8 -*-
"""
Created on Fri Juli 24 10:47:28 2020

@author: Wessel de Jongh
Prediction of increased wind velocity zones in built environment
"""

import pandas as pd
import geopandas as gpd
import psycopg2
from psycopg2 import Error
import shapely
import shapely.wkt
from shapely.geometry import Point, LineString, LinearRing
import matplotlib.pyplot as plt
import descartes
import numpy as np
import csv
import math
import time
from scipy.spatial import Voronoi, voronoi_plot_2d



def plotSome_buildings(building_data, type, road_list):
    # order of operations:
    ## Create GeoDataFrame with lists
    ## create plot handle if multiple plots
    ## plot

    """Input for this is buildings, optional input (kwargs) is road data. Type variable is to either visualise height or based on density."""

    if type == "height":
        gdf = gpd.GeoDataFrame(building_data, crs='epsg:28992').set_index('id')
        ax = gdf.plot(column='height', k=10, cmap='viridis', legend=True)

    elif type == "fad":
        gdf = gpd.GeoDataFrame(building_data, crs='epsg:28992').set_index('id')
        ax = gdf.plot(column='density', k=5, cmap='hot', legend=True, vmax=300, vmin=0)

    if road_list[0]['geometry'] is not None:
        gpd.GeoDataFrame(road_list, crs='epsg:28992').set_index('id').plot(ax=ax, color='G', linewidth=0.5)

    print('Plotting: ', '{}'.format(type))
    plt.show()


def building_density(building_data):
    """No longer necessary"""

    density_list = []
    for item in building_data:
        height = item['Height']
        area = item['geometry'].area
        if height is None or area is None:
            # item['density'] = float(0.0)
            density_list.append(float(0.0))

        elif height != 0 and area > 10:
            dens = float(area) / float(height)
            # item['density'] = float(dens)
            density_list.append(float(dens))
        else:
            # item['density'] = float(0.0)
            density_list.append(float(0.0))

    building_data_copy = building_data

    return density_list


def tall_buildings(road_list, plot=None):
    """Select buildings based on threshold and based on buffer"""
    try:
        my_tallBuildingQuery = 'SELECT id, ST_AsText(geom), pand3d."roof-0.95"-pand3d."ground-0.30" as height ' \
                               'FROM pand3d ' \
                               'WHERE pand3d."roof-0.95"-pand3d."ground-0.30" > 60;'
        cursor.execute(my_tallBuildingQuery)
        record_tallBuilding = cursor.fetchall()
        my_tallBuildings = []

        for tall in record_tallBuilding:
            my_tallRecord = {'id': tall[0], 'geometry': shapely.wkt.loads(tall[1]), 'height': tall[2]}
            my_tallBuildings.append(my_tallRecord)

        if plot:
            plotSome_buildings(my_tallBuildings, "height", road_list)

        return my_tallBuildings


    except(Exception, psycopg2.Error) as error:
        print("Error while trying Tall Buildings: ", error)


def fad(road_data, plot=None):
    try:
        my_buildingQuery = 'SELECT id, ST_AsText(geom), (pand3d."roof-0.95"-pand3d."ground-0.30") as height, ST_Area(geom)/pand3d."roof-0.95" as density  ' \
                           'FROM pand3d ' \
                           'WHERE ST_Area(geom) > 0 and pand3d."roof-0.95" > 0;'
        cursor.execute(my_buildingQuery)
        record_building = cursor.fetchall()
        building_list = []

        # Put the query result in a list nested dict
        for row in record_building:
            my_buildingData = {'id': row[0], 'height': row[2], 'geometry': shapely.wkt.loads(row[1]), 'density': row[3]}
            building_list.append(my_buildingData)

        if plot:
            plotSome_buildings(building_list, 'fad', road_data)


    except(Exception, psycopg2.Error) as error:
        print("Error during fad:", error)


def get_roads():
    """
    This function queries the roads from the PostgreSQL database
    :return: road listed dict
    :rtype: list
    """
    try:
        my_roadQuery = 'SELECT id, ST_asText(geom), stt_naam FROM wegvakken;'
        cursor.execute(my_roadQuery)
        record_road = cursor.fetchall()
        road_list = []

        # Put the query result in a list nested dict
        for i in record_road:
            my_roadData = {'id': i[0], 'geometry': shapely.wkt.loads(i[1])}
            road_list.append(my_roadData)

        return road_list

    except(Exception, psycopg2.Error) as error:
        print("Error during get_roads: ", error)

def get_weather():

    fields = pd.read_fwf("http://weather.tudelft.nl/csv/fields.txt", header=None)
    weather_head = []
    for h in fields[1]:
        weather_head.append(h)
    #
    # print('banananan')
    #
    # with open("weather.csv", "w") as filepointer:
    #     for item in weather_head:
    #         filepointer.write("%s, "%item)

    initial_data = pd.read_csv("http://weather.tudelft.nl/csv/Delfshaven.csv", header=None).tail(20)._get_values
    weather_data = []
    for row in initial_data:
        weather_data.append(list(row))

    weather_data.insert(0, weather_head)
    with open("test_file.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(weather_data)
    #
    # data = []
    # temp_data = []
    # with open("test_file.csv", 'r') as my_file:
    #     reader = csv.reader(my_file)
    #     data = list(reader)
    #     print('my banana is bigger than yours')
    #
    #
    # print('banana')


def windDirection(point, angle):
    """Rewrites wind direction in degrees to linear function"""
    Ax = float(point.x)
    Ay = float(point.y)

    pass


def line_function(A, B):
    """Get the function of the line to check direction, create a table for every line seg"""

    Ax = float(A.x)
    Ay = float(A.y)
    Bx = float(B.x)
    By = float(B.y)

    # to calculate normal, first find f(x) from two points
    # y = slope * x + b
    slope = (Ay - By) / (Ax - Bx)
    b = Ay - (slope * Ax)
    # y_normal = (-1/slope)(float(AB.centroid.xy[0]) - Ax) + Ay
    slope_normal = (-1 / slope)
    b_normal = -(slope_normal * Ax) + Ay

    pass


def azimuth(pointA, pointB):
    """
    Calculates the azimuth of two points
    :return: """
    # A_x, A_y = pointA
    # B_x, B_y = pointB
    # angle = np.arctan(B_x - A_x, B_y - A_y)

    angle = np.arctan2(pointB.y - pointA.y, pointB.x - pointA.x)

    return np.degrees(angle)

def windward(wind_direction, plot=None, save=None):
    """Find the windward and leeward sides of the buildings.
    :return: list of dicts with windward sides and areas
     :rtype: list"""
    #TODO: rewrite input parameters for this function, building query

    my_wind = wind_direction
    some_buildings = tall_buildings(get_roads())
    targets = []
    exterior_id = 0
    exploded_targets = []

    # buildings = ((data["id"], data["exterior"]) for data in some_buildings)

    for i in some_buildings:
        targets.append(
            {'id': i['id'], 'geometry': i['geometry'], 'ccw': i['geometry'].exterior.is_ccw, 'height': i['height']})

    big_coords = []
    for ring in targets:
        if ring['ccw']:
            x = ring['geometry'].exterior.coords.xy[0]
            y = ring['geometry'].exterior.coords.xy[1]
            point_list = list(zip(x, y))
            for j in range(0, len(point_list)):
                A = Point(point_list[j])

                if j == (len(point_list) - 1):
                    B = Point(point_list[0])
                else:
                    B = Point(point_list[j + 1])

                AB = LineString([A, B])
                facade_area = AB.length * ring['height']
                Ax = float(A.x)
                Ay = float(A.y)
                Bx = float(B.x)
                By = float(B.y)
                my_azimuth = azimuth(A, B)

                # s1
                if Ax < Bx and Ay < By:
                    if my_azimuth < my_wind < (my_azimuth+180):
                        windward = 1
                    else:
                        windward = 0

                # s2
                elif Ax > Bx and Ay < By:
                    if (my_azimuth+180) < my_wind < (my_azimuth+360):
                        windward = 1
                    else:
                        windward = 0

                # s3 ! here I flip the wind angle for convenience, otherwise I have to deal with 360 degrees to 0
                elif Ax > Bx and Ay > By:
                    if (my_azimuth+180) < ((my_wind+180)%360) < ((my_azimuth+360)%360):
                        windward = 1
                    else:
                        windward = 0

                # s4
                elif Ax < Bx and Ay > By:
                    if (my_azimuth+180) < ((my_wind+180)%360) < (my_azimuth+360):
                        windward = 1
                    else:
                        windward = 0

                # append to output list
                exploded_targets.append({'id': ring['id'],
                                         'segment_id': j,
                                         # 'geometry':ring['geometry'],
                                         'start': A,
                                         'end': B,
                                         'geometry': AB,
                                         'height': ring['height'],
                                         'facade_area': facade_area,
                                         'azimuth': my_azimuth,
                                         'windward': windward})
        else:
            print('geometry not oriented correctly')
            continue

                # if len(exploded_targets) > 37:
                #     gdf = gpd.GeoDataFrame(exploded_targets[37:], crs='epsg:28992').set_index('id')
                #     ax = gdf.plot(column='windward', k=10, cmap='viridis', legend=True)
                #     plt.show()
                #     print('banana')
    if plot:
        gdf = gpd.GeoDataFrame(exploded_targets, crs='epsg:28992').set_index('id')
        ax = gdf.plot(column='windward', k=10, cmap='viridis', legend=True)
    if save:
        plt.savefig('temp_plot_2.png', dpi=1080)

    if plot:
        plt.show()

    print('banana')

    return exploded_targets


def urban_canyon_AR(buildings, roads):
    """
    Calculates the Aspect Ratio of given geometries and returns them in a listed dict.
    :return: space between buildings, classified
    :rtype: list
    """



    pass

def create_fill(table_name, data):
    """Creates a new PostgreSQL table and fill it with data (listed dict)"""

    pass

def connecting_roads():
    """
    This functions runs for every road segment to see if the next one has roughly the same direction.
    :return: dataframe
    """

    my_roads = get_roads()
    visualise = []

    for line in my_roads:
        visualise.append(line)
        gdf = gpd.GeoDataFrame(visualise, crs='epsg:28992').set_index('id')
        gdf.plot(color='G', linewidth=0.5)
        plt.show()

        start = Point(line['geometry'].xy[0][0], line['geometry'].xy[1][0])
        end = Point(line['geometry'].xy[0][1], line['geometry'].xy[1][1])
        my_azimuth = azimuth(start, end)

        # buffer end points for other roads
        # check all roads for azimuths, compare with my_azimuth
        # if azimuth is the same with about 15 degrees, we might consider it to be joined?

        gdf.buffer
        print('banana')


    pass

def main():
    # starttime = time.time()
    """
    Main logic of the program. Calls all function and get user inputs.
    :return: 
    """

    #TODO: pick a building and surrounding buildings, including roads
    # Or pick a street and find surrounding buildings and streets

    # tall_buildings(get_roads(), plot=True)
    # fad(get_roads(), plot=True)
    # get_weather()
    # windward(90)
    # urban_canyon_AR()

    connecting_roads()




    # df = pd.read_csv("http://weather.tudelft.nl/csv/Delfshaven.csv", header=None).tail(20)
    # whatsthis = df.to_string()
    # with open("weather.csv", mode=w):

    # endtime = time.time()
    # duration = endtime - starttime
    # print("Runtime: ", round(duration, 2), "s")

    print('banana ending')

if __name__ == '__main__':
    try:
        connection = psycopg2.connect(user="postgres",
                                      password="0000000",
                                      host="127.0.0.1",
                                      port="5432",
                                      database="thesisData")

        cursor = connection.cursor()
        # Print PostgreSQL Connection properties
        print(connection.get_dsn_parameters(), "\n")

        main()

    except(Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)

    finally:
        # closing database connection.
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
