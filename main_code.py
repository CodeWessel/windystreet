import pandas as pd
import geopandas as gpd
import psycopg2
from psycopg2 import Error
import shapely
import shapely.wkt
import matplotlib.pyplot as plt
import descartes
import numpy as np
import csv


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
            my_tallRecord = {'id':tall[0], 'geometry':shapely.wkt.loads(tall[1]), 'height':tall[2]}
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
        print("Error while retrieving road data:", error)



# fields = pd.read_fwf("http://weather.tudelft.nl/csv/fields.txt", header=None)
# weather_head = []
# for h in fields[1]:
#     weather_head.append(h)
# #
# # print('banananan')
# #
# # with open("weather.csv", "w") as filepointer:
# #     for item in weather_head:
# #         filepointer.write("%s, "%item)
#
# initial_data = pd.read_csv("http://weather.tudelft.nl/csv/Delfshaven.csv", header=None).tail(20)._get_values
# weather_data = []
# for row in initial_data:
#     weather_data.append(list(row))
#
# weather_data.insert(0, weather_head)
# with open("test_file.csv", 'w', newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(weather_data)

data = []
temp_data = []
with open("test_file.csv", 'r') as my_file:
    reader = csv.reader(my_file)
    data = list(reader)
    print('my banana is bigger than yours')


print('banana')


try:
    connection = psycopg2.connect(user = "postgres",
                                  password = "0000000",
                                  host = "127.0.0.1",
                                  port = "5432",
                                  database = "thesisData")

    cursor = connection.cursor()

    # Print PostgreSQL Connection properties
    print(connection.get_dsn_parameters(),"\n")

    # tall_buildings(get_roads(), plot=True)
    # fad(get_roads(), plot=True)

    windDirection = 45

    # some_buildings = tall_buildings(get_roads())



    # my_target = tall_buildings(get_roads())

    # df = pd.read_csv("http://weather.tudelft.nl/csv/Delfshaven.csv", header=None).tail(20)
    # whatsthis = df.to_string()
    # with open("weather.csv", mode=w):




    # for record in my_target:



    print('banana')


except(Exception, psycopg2.Error) as error :
    print("Error while connecting to PostgreSQL", error)

finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

