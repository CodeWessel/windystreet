import pandas as pd
import geopandas as gpd
import psycopg2
from psycopg2 import Error
import shapely
import shapely.wkt
import matplotlib.pyplot as plt
import descartes

"""This code was an exercise to connect to the postgreSQL database and visualize multiple datasets
in one single plot. """

def plotSome_buildings(building_data, type, road_list):
    # order of operations:
    ## Create GeoDataFrame with lists
    ## create plot handle if multiple plots
    ## plot

    """Input for this is buildings, optional input (kwargs) is road data. Type variable is to either visualise height or based on density."""

    if type == "height":
        gdf = gpd.GeoDataFrame(building_data, crs='epsg:28992').set_index('id')
        ax = gdf.plot(column='height', k=10, cmap='viridis')

    elif type == "density":
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

def tall_buildings(road_list):

    """Select buildings based on threshold and based on buffer"""
    try:
        my_tallBuildingQuery = 'SELECT id, ST_AsText(geom), pand3d."roof-0.95" as height ' \
                               'FROM pand3d ' \
                               'WHERE pand3d."roof-0.95" > 60;'
        cursor.execute(my_tallBuildingQuery)
        record_tallBuilding = cursor.fetchall()
        my_tallBuildings = []

        for tall in record_tallBuilding:
            my_tallRecord = {'id':tall[0], 'geometry':shapely.wkt.loads(tall[1]), 'height':tall[2]}
            my_tallBuildings.append(my_tallRecord)
        print('plot??')
        if road_list is not None:
            plotSome_buildings(my_tallBuildings, "height", road_list)
        else:
            plotSome_buildings(my_tallBuildings, "height")
        print('baanan')

    except(Exception, psycopg2.Error) as error:
        print("Error while trying Tall Buildings: ", error)

def fad(road_data):
    try:
        my_buildingQuery = 'SELECT id, ST_AsText(geom), pand3d."roof-0.95" as height, ST_Area(geom)/pand3d."roof-0.95" as density  ' \
                           'FROM pand3d ' \
                           'WHERE ST_Area(geom) > 0 and pand3d."roof-0.95" > 0;'
        cursor.execute(my_buildingQuery)
        record_building = cursor.fetchall()
        building_list = []

        # Put the query result in a list nested dict
        for row in record_building:
            my_buildingData = {'id': row[0], 'height': row[2], 'geometry': shapely.wkt.loads(row[1]), 'density': row[3]}
            building_list.append(my_buildingData)

        if road_data is not None:
            plotSome_buildings(building_list, 'density', road_data)
        else:
            plotSome_buildings(building_list, 'density')

    except(Exception, psycopg2.Error) as error:
        print("Error during fad:", error)

def get_road():
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


try:
    connection = psycopg2.connect(user = "postgres",
                                  password = "0000000",
                                  host = "127.0.0.1",
                                  port = "5432",
                                  database = "thesisData")

    cursor = connection.cursor()

    # Print PostgreSQL Connection properties
    # print( connection.get_dsn_parameters(),"\n")

    my_roadQuery = 'SELECT id, ST_asText(geom), stt_naam FROM wegvakken;'
    cursor.execute(my_roadQuery)
    record_road = cursor.fetchall()
    road_list = []

    # Put the query result in a list nested dict
    for i in record_road:
        my_roadData = {'id': i[0], 'geometry': shapely.wkt.loads(i[1])}
        road_list.append(my_roadData)




    my_buildingQuery = 'SELECT id, ST_AsText(geom), pand3d."roof-0.95" as height, ST_Area(geom)/pand3d."roof-0.95" as density  ' \
                       'FROM pand3d ' \
                       'WHERE ST_Area(geom) > 0 and pand3d."roof-0.95" > 0;'
    cursor.execute(my_buildingQuery)
    record_building = cursor.fetchall()
    building_list = []

    # Put the query result in a list nested dict
    for row in record_building:
        my_buildingData = {'id': row[0], 'height': row[2], 'geometry': shapely.wkt.loads(row[1]), 'density': row[3]}
        building_list.append(my_buildingData)

    plotSome_buildings(building_list, 'density', road_list)


    print('banana')

except(Exception, psycopg2.Error) as error :
    print("Error while connecting to PostgreSQL", error)

finally:
    #closing database connection.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

