# -*- coding: utf-8 -*-
"""
Created on Fri Juli 24 10:47:28 2020

@author: Wessel de Jongh
Prediction of increased wind velocity zones in built environment
"""

import shapely
import shapely.wkt
from shapely.geometry import Point, LineString, LinearRing, MultiPoint, Polygon
from shapely.ops import nearest_points
from shapely import affinity
import pandas as pd
import geopandas as gpd
import psycopg2
from psycopg2 import Error
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
# import descartes
import numpy as np
import csv
import math
from math import radians
import time
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
from shapely.ops import nearest_points
import sqlalchemy as sal
from sqlalchemy import create_engine
import geoalchemy2
import copy
from pyhull.voronoi import VoronoiTess
import itertools
from random import random
# import pygeos

def distance(p_1, p_2):
    """
    Distance between two given points.
    """
    return math.sqrt((p_2[0] - p_1[0])**2 + (p_2[1] - p_1[1])**2)

def get_equidistant_points(p_1, p_2, parts):
    """
    Source:
    https://stackoverflow.com/questions/47443037/equidistant-points-between-two-points-in-python
    """
    return list(zip(np.linspace(p_1[0], p_2[0], parts + 1),
                    np.linspace(p_1[1], p_2[1], parts + 1)))

def plotSome_buildings(building_data, type, road_list):
    # order of operations:
    ## Create GeoDataFrame with lists
    ## create plot handle if multiple plots
    ## plot

    """Input for this is buildings, optional input (kwargs) is road data. Type variable is to either visualise height or based on density."""

    if type == "height":
        gdf = gpd.GeoDataFrame(building_data, crs='epsg:28992').set_index('id')
        ax = gdf.plot(column='height', k=10, cmap='viridis', legend=True)

    elif type == "flad":
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


def tall_buildings(plot=None):
    """Select buildings based on threshold and based on buffer"""
    try:
        my_tallBuildingQuery = 'SELECT id, ST_AsText(geom), pand3d."roof-0.95"-pand3d."ground-0.30" as height ' \
                               'FROM pand3d ' \
                               # 'WHERE pand3d."roof-0.95"-pand3d."ground-0.30" > 60;'
        cursor.execute(my_tallBuildingQuery)
        record_tallBuilding = cursor.fetchall()
        my_tallBuildings = []

        for tall in record_tallBuilding:
            my_tallRecord = {'id': tall[0], 'geometry': shapely.wkt.loads(tall[1]), 'height': tall[2]}
            my_tallBuildings.append(my_tallRecord)

        if plot:
            road_list = get_roads()
            plotSome_buildings(my_tallBuildings, "height", road_list)



        return my_tallBuildings


    except(Exception, psycopg2.Error) as error:
        print("Error while trying Tall Buildings: ", error)

def all_buildingQuery():
    try:
        my_buildingQuery = 'SELECT id, ST_AsText(geom), (pand3d."roof-0.99"-pand3d."ground-0.50") as height, ST_Area(geom)/pand3d."roof-0.99" as density  ' \
                           'FROM pand3d ' \
                           'WHERE ST_Area(geom) > 0 AND pand3d."roof-0.95" > 0 AND pand3d."ground-0.50" IS NOT NULL;'
        cursor.execute(my_buildingQuery)
        record_building = cursor.fetchall()
        building_list = []

        for row in record_building:
            my_buildingData = {'id': row[0], 'geometry': shapely.wkt.loads(row[1]), 'height': row[2], 'density': row[3]}
            building_list.append(my_buildingData)

        return building_list

    except(Exception, psycopg2.Error) as error:
        print("Error while trying Tall Buildings: ", error)

def flad(road_data, plot=None):
    try:
        my_buildingQuery = 'SELECT id, ST_AsText(geom), (pand3d."roof-0.95"-pand3d."ground-0.30") as height  ' \
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
            plotSome_buildings(building_list, 'flad', road_data)


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

def get_bufferRoads():
    try:
        my_roadQuery = "SELECT *, ST_asText(ST_StartPoint(geom)) as start, " \
                       "ST_AsText(ST_EndPoint(geom)) as end, " \
                       "ST_Azimuth(ST_StartPoint(geom), ST_EndPoint(geom)) as azimuth, " \
                       "ST_AsText(ST_buffer(ST_StartPoint(geom), 2, 'quad_segs=8')) as start_buffer, " \
                       "ST_AsText(ST_buffer(ST_EndPoint(geom), 2, 'quad_segs=8')) as end_buffer, " \
                       "ST_AsText(geom) as geometry " \
                       "FROM wegvakken;"
        cursor.execute(my_roadQuery)
        record_road = cursor.fetchall()
        road_list = []

        # Put the query result in a list nested dict
        for i in record_road:
            my_roadData = {'id': i[0],
                           'geometry': shapely.wkt.loads(i[35]),
                           'start':shapely.wkt.loads(i[30]),
                           'end':shapely.wkt.loads(i[31]),
                           'azimuth': math.degrees(i[32]),
                           'start_buffer':shapely.wkt.loads(i[33]),
                           'end_buffer':shapely.wkt.loads(i[34])}
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

def windward(buildings, wind_direction, plot=None, save=None):
    """Find the windward and leeward sides of the buildings.
    :return: list of dicts with windward sides and areas
     :rtype: list"""

    my_wind = wind_direction
    some_buildings = buildings
    targets = []
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
            for j in range(0, len(point_list)-1):
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
                if Ax <= Bx and Ay < By:
                    if my_azimuth < my_wind < (my_azimuth+180):
                        windward = 1
                    else:
                        windward = 0

                # s2
                elif Ax > Bx and Ay <= By:
                    if (my_azimuth) < my_wind < (my_azimuth+180):
                        windward = 1
                    else:
                        windward = 0

                # s3 ! here I flip the wind angle for convenience, otherwise I have to deal with 360 degrees to 0
                elif Ax >= Bx and Ay > By:
                    if ((my_azimuth+180)%360) < ((my_wind+180)%360) < ((my_azimuth+360)%360):
                        windward = 1
                    else:
                        windward = 0

                # s4
                elif Ax < Bx and Ay >= By:
                    if ((my_azimuth+180)%360) < ((my_wind+180)%360) < ((my_azimuth+360)%360):
                        windward = 1
                    else:
                        windward = 0
                elif Ax == Bx and Ay == By:
                    print('faulty geometry: ', ring['id'], j)
                    continue
                else:
                    print('j %d, Ax %d, Ay %d, Bx %d, By %d' % (j, Ax, Ay, Bx, By))
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

            # ## for plotting single buildings with azimuths
            # gdf = gpd.GeoDataFrame(exploded_targets, crs='epsg:28992').set_index('id')
            # ax = gdf.loc[ring['id']].plot(column='windward', k=10, cmap='viridis', legend=True)
            # for idx, row in gdf.loc[ring['id']].iterrows():
            #     ax.annotate(text=row['segment_id'], xy=(row['start'].x,row['start'].y))
            #     ax.annotate(text=row['azimuth'], xy=(row.geometry.centroid.x, row.geometry.centroid.y))
            # plt.show()
            # print('banana')

        else:
            print('geometry not oriented correctly, skipped')
            continue

    if plot:
        gdf = gpd.GeoDataFrame(exploded_targets, crs='epsg:28992').set_index('id')
        ax = gdf.plot(column='windward', k=10, cmap='viridis', legend=True)
        ax.set_xlim(89000, 89300)
        ax.set_ylim(436250, 436500)
    if save:
        plt.savefig('temp_plot_3.png', dpi=1080)

    if plot:
        plt.show()

    print('banana')

    return exploded_targets

def create_table(table_name):
    commands = """
    CREATE Table %s (
    %s_id SERIAL PRIMARY KEY,
    %s_name VARCHAR(255) NOT NULL
    )
    """ % (table_name, table_name, table_name)
    print(commands)

    try:

        cursor.execute(commands)
        cursor.close()
        connection.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        # closing database connection
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

def connecting_roads():
    """
    This functions runs for every road segment to see if the next one has roughly the same direction.
    :return: dataframe
    """
    my_bufferedRoads = get_bufferRoads()
    gdf = gpd.GeoDataFrame(my_bufferedRoads, crs='espg:28992').set_index('id')
    # gdf.to_sql('what', connection, if_exists='replace', index=False, dtype={'geom': Geometry('POINT', srid='28992')})




    my_roads = get_roads()
    visualise = []
    enriched_roads = []
    id = 0
    p_id = 00
    data_points = []
    data = []
    my_id = []
    for line in my_roads[:20]:
        visualise.append(line)
        gdf = gpd.GeoDataFrame(visualise, crs='epsg:28992').set_index('id')
        # gdf.plot(color='G', linewidth=0.5)
        # plt.show()

        start = Point(line['geometry'].xy[0][0], line['geometry'].xy[1][0])
        end = Point(line['geometry'].xy[0][1], line['geometry'].xy[1][1])
        my_azimuth = azimuth(start, end)

        item_dict = {}
        item_dict['id'] = id
        item_dict['start'] = start
        item_dict['start_id'] = p_id
        item_dict['end'] = end
        item_dict['end_id'] = p_id+1
        item_dict['azimuth'] = my_azimuth
        item_dict['line'] = line['geometry']
        item_dict['joined'] = 0
        my_id.append(p_id)
        my_id.append(p_id+1)

        enriched_roads.append(item_dict)

        data.append([line['geometry'].xy[0][0], line['geometry'].xy[1][0], p_id])
        data.append([line['geometry'].xy[0][1], line['geometry'].xy[1][1], p_id+1])
        # data_points.append( p_id)
        # data_points.append([end, )
        start_point = {}
        start_point['id'] = p_id
        start_point['geometry'] = start
        start_point['azimuth'] = my_azimuth
        end_point = {}
        end_point['id'] = p_id+1
        end_point['geometry'] = end
        end_point['azimuth'] = my_azimuth
        data_points.append(start_point)
        data_points.append(end_point)

        ## buffer end points for other roads
        ## check all roads for azimuths, compare with my_azimuth
        ## if azimuth is the same with about 15 degrees, we might consider it to be joined?

        ## Why buffer? why not just search for similar points? e.g Point(x±2, y±2) might work and faster

        # gdf.geometry.buffer
        # print('banana')

        #TODO: enriched_roads, iterate through whole roads dataset for similar points. Check for similar azimuth
        # if similar azimuth 'join' and remove from dataset by changing joined variable 0 to 1
        id += 1
        p_id = id * 10
    # data_np = np.array(data)
    # destinations = MultiPoint(data_points)

    gdf2 = gpd.GeoDataFrame(enriched_roads, crs='epsg:28992').set_index('id')
    gdf3 = gpd.GeoDataFrame(data_points, crs='epsg:28992').set_index('id')

    dataPoint_buffers = gdf3.buffer(1)

    what = dataPoint_buffers.within(gdf3)
    for i, g in zip(my_id, dataPoint_buffers):
        print(g)
        print(i)
        this = g.contain(data_points)

    for bro in data_points:
        maybe = dataPoint_buffers.contains(i['geometry'])
        print('yes')
    print('next banana')
    # for seg in enriched_roads:
    #     #first check start point
    #     start_buffer = seg['start'].buffer(1)
    #
    #     for pt in data_points:
    #         if seg['start_id'] == pt['id']:
    #             continue
    #
    #
    #     print('bana')
    print('finished banana')

    pass

def consecutive_facades(buildings, wind_direction):
    """

    :param: simplified buildings
    :param: enriched voronoi from urbanCanyon
    :return: super enriched voronoi with facade lengths
    :type: GeoDataframe
    """

    # try:
    my_query = 'SELECT simplified.id, ST_AsText(ST_ForcePolygonCCW(ST_CurveToLine(simplified.geom))), public.cbs_buurt_2019_gegeneraliseerd.id as buurt_id ' \
    'FROM public.simplified, public.cbs_buurt_2019_gegeneraliseerd ' \
    'WHERE ST_Intersects(ST_CurveToLine(simplified.geom), cbs_buurt_2019_gegeneraliseerd.geom); '


    cursor.execute(my_query)
    record_simplified = cursor.fetchall()

    simplified = []

    for row in record_simplified:
        feature = {'simplified_id':row[0],
                   'geometry':list(shapely.wkt.loads(row[1])),
                   # 'geometry': shapely.wkt.loads(row[1]),
                   'buurt_id':row[2]
                   }
        simplified.append(feature)

    done = []
    my_id = 0
    for entry in simplified: #explode
        for n, item in enumerate(entry['geometry']):
            coords = list(item.exterior.coords)

            #check for interiors
            interior_coords = []
            for interior in item.interiors:
                interior_coords += interior.coords[:]
            interior_coords.reverse()
            if len(interior_coords) > 0:
                for it, ptn in enumerate(interior_coords):
                    if it == len(interior_coords) - 1:
                        continue
                    else:
                        line = shapely.geometry.LineString([ptn, interior_coords[it + 1]])
                        my_dict = {'id': my_id,
                                   'simplified_id': entry['simplified_id'],
                                   'sub_polygon_id': n,
                                   'interior': 1,
                                   'line_id': it,
                                   'geometry': line,
                                   'azimuth': azimuth(Point(ptn), Point(interior_coords[it + 1])),
                                   'group_line': None,
                                   'group_length': None}
                        my_id += 1
                        done.append(my_dict)
            # gdf = gpd.GeoDataFrame(done, crs='epsg:28992').set_index('id', drop=False)
            # ax = gdf.plot(column='id', legend=True, linewidth=1.5)
            # for idx, row in gdf.iterrows():
            #     ax.annotate(text=row['id'], size=6,
            #                 xy=(row.geometry.centroid.x, row.geometry.centroid.y))
            # plt.show()
            # print('hodl')
            for i, pt in enumerate(coords):
                if i == len(coords) - 1:
                    continue
                else:
                    line = shapely.geometry.LineString([pt, coords[i + 1]])
                    my_dict = {'id': my_id,
                               'simplified_id': entry['simplified_id'],
                               'sub_polygon_id': n,
                               'interior': 0,
                               'line_id': i,
                               'geometry': line,
                               'azimuth': azimuth(Point(pt), Point(coords[i+1])),
                               'group_line': None,
                               'group_length': None}
                    my_id += 1
                    done.append(my_dict)
    print('Starting windward...')
    for wdir in wind_direction:
        done = windward_after(done, wdir)
    # # Forward looking, unneccesary
    # n = 0
    # for j, segment in enumerate(done):
    #     for fw in range(4):
    #         if done[j+fw]['simplified_id'] == segment['simplified_id']:
    #             if (segment['azimuth']-15) <= done[j+fw]['azimuth'] <= (segment['azimuth']+15):
    #                 print('same %d' % n)
    #                 n += 1
    print('Done windward...')
    print(done[0])
    max_backward = 3
    max_azm_dif = radians(20)
    stop = 0
    for m, part in enumerate(done):
        if part['line_id'] == 0:
            # print('new polygon ', m)
            ## visualising the buildings
            # if m != 0:
            #     gdf = gpd.GeoDataFrame(done[stop:m])
            #     colors = [(random(), random(), random()) for i in range(m-stop)]
            #     new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=(m+1-stop))
            #     ax = gdf.plot(column='group_line', cmap=new_map, legend=True, linewidth=1.5)
            #     for idx, row in gdf.iterrows():
            #         ax.annotate(text=row['group_line'], size=6,
            #                     xy=(row.geometry.centroid.x, row.geometry.centroid.y))
            #
            #     plt.show()
            #     print('HODL')

            stop = m
            part['group_line'] = m
            part['group_length'] = part['geometry'].length

        elif part['line_id'] == 1: # I had to do this, because range(m-1, stop, -1) skips the 0 altogether
            if (radians(done[m-1]['azimuth']) - max_azm_dif) < radians(part['azimuth']) < (
                    radians(done[m-1]['azimuth']) + max_azm_dif):
                part['group_line'] = done[m-1]['group_line']
                part['group_length'] = part['geometry'].length + done[m-1]['group_length']
                done[m-1]['group_length'] = part['group_length']

            else:
                part['group_line'] = m
                part['group_length'] = part['geometry'].length


        elif part['line_id'] < max_backward:
            for l in range(m-1, stop ,-1):
                # print('m', m, 'l', l)
                # print(radians(part['azimuth']), ' ', radians(done[l]['azimuth']))
                if (radians(done[l]['azimuth'])-max_azm_dif) < radians(part['azimuth']) < (radians(done[l]['azimuth'])+max_azm_dif):
                    part['group_line'] = done[l]['group_line']
                    part['group_length'] = part['geometry'].length + done[part['group_line']]['group_length']
                    done[part['group_line']]['group_length'] = part['group_length']
                    break # because as soon as he found something, just continue with the next iteration

                else:
                    part['group_line'] = m
                    part['group_length'] = part['geometry'].length

        elif part['line_id'] >= max_backward:
            for h in range(m-1, m-max_backward, -1):
                # print('m', m, 'h', h)
                # print(radians(part['azimuth']), ' ', radians(done[h]['azimuth']))

                if (radians(done[h]['azimuth'])-max_azm_dif) < radians(part['azimuth']) < (radians(done[h]['azimuth'])+max_azm_dif):
                    part['group_line'] = done[h]['group_line']
                    part['group_length'] = part['geometry'].length + done[part['group_line']]['group_length']
                    done[part['group_line']]['group_length'] = part['group_length']
                    break # because as soon as he found something, just continue with the next iteration

                else:
                    part['group_line'] = m
                    part['group_length'] = part['geometry'].length

    for x, entry in enumerate(done): # re-assign length so everythin is consistent
        if entry['group_line'] == x:
            pass
        else:
            entry['group_length'] = done[entry['group_line']]['group_length']

    for a in done:
        del a['simplified_id'], a['sub_polygon_id'], a['interior'], a['line_id'], a['group_line']

    print(done[0])

    write_to_db(done)


    # except(Exception, psycopg2.Error) as error:
    #     print("Error during urbanCanyon: ", error)
    # finally:
    #     # closing database connection
    #     if (connection):
    #         cursor.close()
    #         connection.close()
    #         print("PostgreSQL connection is closed")

def write_to_db(done):
    print('Started writing consecutive facade to postgis...')
    db_connection_url = "postgres://postgres:0000000@127.0.0.1:5432/thesisData"
    engine = create_engine(db_connection_url)
    print('Creating GeoDataFrame...')
    gdf_done = gpd.GeoDataFrame(done, crs='epsg:28992')
    print('Now writing...')
    gdf_done.to_postgis(name='consecutivefacadewind_final_bottom', con=engine, if_exists='replace')
    print('Writen...')

def windward_after(data, wind_direction):
    for i in data:
        my_wind = wind_direction
        line = i['geometry']
        A, B = list(zip(list(line.xy[0]), list(line.xy[1])))
        A = Point(list(A))
        B = Point(list(B))
        impact = 0
        if i['interior'] == 0:

            my_azimuth = azimuth(A, B)
            Ax = float(A.x)
            Ay = float(A.y)
            Bx = float(B.x)
            By = float(B.y)

            # s1
            if Ax <= Bx and Ay < By:
                if my_azimuth < my_wind < (my_azimuth + 180):
                    windward = 1
                else:
                    windward = 0

            # s2
            elif Ax > Bx and Ay <= By:
                if (my_azimuth) < my_wind < (my_azimuth + 180):
                    windward = 1
                else:
                    windward = 0

            # s3 ! here I flip the wind angle for convenience, otherwise I have to deal with 360 degrees to 0
            elif Ax >= Bx and Ay > By:
                if ((my_azimuth + 180) % 360) < ((my_wind + 180) % 360) < ((my_azimuth + 360) % 360):
                    windward = 1
                else:
                    windward = 0

            # s4
            elif Ax < Bx and Ay >= By:
                if ((my_azimuth + 180) % 360) < ((my_wind + 180) % 360) < ((my_azimuth + 360) % 360):
                    windward = 1
                else:
                    windward = 0
            elif Ax == Bx and Ay == By:
                print('faulty geometry: ', ring['id'], j)
                continue
            else:
                print('j %d, Ax %d, Ay %d, Bx %d, By %d' % (j, Ax, Ay, Bx, By))
            # append to output list
            i['windward_%s' % my_wind] =  windward

        elif i['interior'] == 1:
            Am = B
            Bm = A
            my_azimuth = azimuth(Am, Bm)
            Ax = float(Am.x)
            Ay = float(Am.y)
            Bx = float(Bm.x)
            By = float(Bm.y)



            # s1
            if Ax <= Bx and Ay < By:
                if my_azimuth < my_wind < (my_azimuth + 180):
                    windward = 1

                else:
                    windward = 0

            # s2
            elif Ax > Bx and Ay <= By:
                if (my_azimuth) < my_wind < (my_azimuth + 180):
                    windward = 1

                else:
                    windward = 0

            # s3 ! here I flip the wind angle for convenience, otherwise I have to deal with 360 degrees to 0
            elif Ax >= Bx and Ay > By:
                if ((my_azimuth + 180) % 360) < ((my_wind + 180) % 360) < ((my_azimuth + 360) % 360):
                    windward = 1
                else:
                    windward = 0

            # s4
            elif Ax < Bx and Ay >= By:
                if ((my_azimuth + 180) % 360) < ((my_wind + 180) % 360) < ((my_azimuth + 360) % 360):
                    windward = 1
                else:
                    windward = 0
            elif Ax == Bx and Ay == By:
                print('faulty geometry: ', ring['id'], j)
                continue
            else:
                print('j %d, Ax %d, Ay %d, Bx %d, By %d' % (j, Ax, Ay, Bx, By))
            # append to output list
            i['windward_%s' % my_wind]= windward

        # if my_azimuth >= 0:
        #     impact = my_azimuth - ((wind_direction + 180) % 360)
        # elif my_azimuth < 0:
        #     impact = my_azimuth - (wind_direction - 360)

        if wind_direction > 180:
            my_wind = wind_direction - 360

            impact = abs(my_wind - my_azimuth)
        else:
            impact = abs(my_wind - my_azimuth)


        if impact > 90:
            impact = abs(impact-180)

        if impact > 90:
            impact = abs(impact-180)


        i['impact_%s' % wind_direction]= round(impact, 2)

    return data


def discretise_to_points(input):

    eq_points_flat = []

    for j in range(len(building_pts) - 1):
        first = building_pts[j]
        second = building_pts[j + 1]
        dist = distance(first, second)

        eq_pts = get_equidistant_points(first, second, int(dist / 2))
        eq_pts_buildings_in_buurt[index_subset].extend(eq_pts)
        eq_points_flat.extend(eq_pts)

    plt.scatter(*zip(*eq_points_flat), c='grey', s=4)
    plt.show()

def urbanCanyon(buildings):
    """
    1: query buildings inside buurt
    2: discretize buildings and buurt polygons
    3: do voronoi
    """

    try:
        my_query = 'SELECT id, ST_asText(geom), jrstatcode ' \
                   'FROM public.cbs_buurt_2019_gegeneraliseerd;'
        cursor.execute(my_query)
        record_buurt = cursor.fetchall()
        buurt = []

        for row in record_buurt:
            feature = {'buurt_id':row[0],
                       'geometry': shapely.wkt.loads(row[1]),
                       'jrstatcode': row[2],
                       'eq_pts': []}
            buurt.append(feature)

        gdf_final = gpd.GeoDataFrame()
        db_connection_url = "postgres://postgres:0000000@127.0.0.1:5432/thesisData"
        engine = create_engine(db_connection_url)

        gdf_buildings = gpd.GeoDataFrame(buildings, crs='epsg:28992').set_index('id')
        building_index = gdf_buildings.sindex
        gdf_buurt = gpd.GeoDataFrame(buurt, crs='epsg:28992').set_index('buurt_id')
        gdf_buurt['eq_pts'] = np.nan
        it = 0
        full = {}

        for index, area in gdf_buurt.iloc[22:].iterrows(): # Loop to select buildings inside Buurt polygon
            polygon = area.geometry[0]
            selection = gdf_buildings.within(polygon)
            subset = gdf_buildings[selection]  # True means buildings inside buurt, now
            bld_indices = [bi for bi in subset.index if selection[bi]]
            eq_pts_buildings_in_buurt = dict.fromkeys(bld_indices, [])

            eq_points_flat = []

            for index_subset, building in subset.iterrows(): # Discretize the building footprints
                building_pts = list(building.geometry.exterior.coords)

                for j in range(len(building_pts) - 1):
                    first = building_pts[j]
                    second = building_pts[j + 1]
                    dist = distance(first, second)

                    eq_pts = get_equidistant_points(first, second, int(dist / 2))
                    eq_pts_buildings_in_buurt[index_subset].extend(eq_pts)
                    eq_points_flat.extend(eq_pts)

            plt.scatter(*zip(*eq_points_flat), c='grey', s=4)
            plt.show()

            full[index] = eq_pts_buildings_in_buurt
            bbox = []
            area_pts = list(polygon.exterior.coords)

            for k in range(len(area_pts) - 1): # Discretize the Buurt polygon into points for voronoi
                first = area_pts[k]
                second = area_pts[k + 1]
                dist = distance(first, second)
                eq_area_pts = get_equidistant_points(first, second, int(dist / 2))
                bbox.extend(eq_area_pts)

            # extra_bbox = []
            # extra_area_pts = list(polygon.envelope.exterior.coords)
            # for k in range(len(extra_area_pts) - 1): # Discretize the Buurt polygon into points for voronoi
            #     first = extra_area_pts[k]
            #     second = extra_area_pts[k + 1]
            #     dist = distance(first, second)
            #     eq_extra_area_pts = get_equidistant_points(first, second, int(dist / 2))
            #     extra_bbox.extend(eq_extra_area_pts)

            plt.scatter(*zip(*bbox), c='grey', s=1)
            plt.show()
            # plt.scatter(*zip(*extra_bbox), c='grey', s=1)
            # plt.show()

            # initiate Voronoi
            voronoi_points = eq_points_flat + bbox
            try:
                voronoi_schematic = VoronoiTess(voronoi_points, add_bounding_box=True)
            except:
                print("Could not create Voronoi for this segment, id: ", index_subset)
                continue

            regions_to_coords = []
            regions_to_polygons = []

            for region in voronoi_schematic.regions:  # add to sub_polygons then to regions_to_polygons
                sub_polygons = []

                if len(region) < 1:
                    pass
                else:
                    for v in region:
                        to_check_for_inf = tuple(voronoi_schematic.vertices[v])
                        if to_check_for_inf[0] == -10.101 or to_check_for_inf[1] == -10.101:
                            continue
                        else:
                            sub_polygons.append(tuple(voronoi_schematic.vertices[v]))
                    regions_to_coords.append(sub_polygons)

                # if len(region) < 1:
                #     pass
                # else:
                #     for v in region:
                #         sub_polygons.append(tuple(voronoi_schematic.vertices[v]))
                #
                #     regions_to_coords.append(sub_polygons)

            for b, g in enumerate(regions_to_coords):
                try:
                    # g.append(g[0])
                    if len(g) > 0:
                        regions_to_polygons.append(shapely.geometry.Polygon(g))
                except:
                    print("error with this region: id = ", b)
                    continue

            filtered = []
            for geom in regions_to_polygons: # Filter to check if entire polygon is inside the Buurt polygon
                # if area.geometry.contains(geom.centroid):
                filtered.append(geom)

            # Create new dataframe to fill with list values later
            gdf_polygons = gpd.GeoDataFrame(regions_to_polygons, columns=['geometry'], crs='epsg:28992')
            gdf_polygons['related_building_id'] = pd.Series([[] for i in range(len(gdf_polygons))])
            gdf_polygons['related_building_height'] = pd.Series([[] for i in range(len(gdf_polygons))])
            gdf_polygons['related_edge_length'] = pd.Series([[] for i in range(len(gdf_polygons))])



            exploded = explode_polygons(filtered)
            gdf_edge = gpd.GeoDataFrame(exploded, crs='epsg:28992')

            # works but not satisfactory

            gdf_test = gpd.GeoDataFrame(([{'id': index, 'geometry' : Polygon(list(polygon.exterior.coords))}]))
            gdf_test = gdf_test.set_crs(epsg=28992, inplace=True)
            gdf_polygons = gpd.overlay(gdf_polygons, gdf_test, how='intersection', make_valid=True)
            gdf_polygons = gdf_polygons[~gdf_polygons.is_empty]
            gdf_edge = gpd.overlay(gdf_edge, gdf_test, how='intersection', make_valid=True)


            # Create spatial index on edge
            building_index = gdf_buildings.sindex
            edge_index = gdf_edge.sindex

            print('Started with determining cell lengths')

            for index_subset, building in subset.iterrows(): # Loop for building / voronoi cell intersection
                bounds = list(building.geometry.bounds)
                environment = list(building_index.intersection(bounds)) #
                candidates = list(edge_index.intersection(bounds))
                possible_matches = gdf_edge.iloc[candidates]
                immediate_environment = gdf_buildings.iloc[environment] #
                final = possible_matches.loc[possible_matches.intersects(building.geometry.exterior)]

                ## code to plot intermediary results
                # diff = final.difference(building.geometry)
                # alternative = final.difference(immediate_environment.geometry) #
                # ax = alternative.plot() #


                res_dif = gpd.overlay(final, immediate_environment, how='difference') # Find only street part of the edge
                related_polygons = gdf_polygons.iloc[list(res_dif['polygon_id'])]


                for indx in related_polygons.index: # combine the building data with the voronoi data
                    gdf_polygons.iloc[indx]['related_building_id'].append(index_subset)
                    gdf_polygons.iloc[indx]['related_building_height'].append(building['height'])


                for line_id, part in res_dif.iterrows(): # add edge lenght (street width) to voronoi data
                    if part.geometry.length > 1:
                        gdf_polygons.iloc[part['polygon_id']]['related_edge_length'].append(part.geometry.length)

            #TODO: find problem in wrong edge length assignment.

            print('Calculating Urban Canyon H/W ratio')

            # Calculate the Urban Canyon ratio
            gdf_polygons['average_length'] = gdf_polygons['related_edge_length'].apply(lambda x: np.mean(x) if (len(x) > 0) else np.nan)
            gdf_polygons['average_height'] = gdf_polygons['related_building_height'].apply(lambda x: np.mean(x) if (len(x) > 0) else np.nan)
            gdf_polygons['UC'] = gdf_polygons.apply(lambda uc: uc['average_height'] / uc['average_length'], axis=1)

            # Unneccesary code:
            # gdf_polygons = gdf_polygons.dropna()
            # gdf_polygons = gdf_polygons.drop(columns=['related_edge_length', 'related_building_height', 'related_building_id'] )

            gdf_polygons.plot()
            plt.show()

            gdf_polygons['buurt'] = index
            # gdf_final = gdf_final.append(gdf_polygons)
            print('banana')
            if it == 0:
                gdf_polygons.to_postgis(name='urbancanyon_table_hodl', con=engine, if_exists='replace')
                it =+ 1
                print('Table replaced')
            else:
                gdf_polygons.to_postgis(name='urbancanyon_table_hodl', con=engine, if_exists='append')
                print('Table appended')

            # gdf_polygons.to_file('final_UC.geojson', driver='GeoJSON', if_exists='append')
            # gdf_final.plot(column='UC', legend=True, cmap='Reds')
            # plt.show()
            print('Done with buurt_%d' % (index))



        # gdf_final.to_file('gdf_final.geojson', driver='GeoJSON')





        return gdf_final

    except(Exception, psycopg2.Error) as error:
        print("Error during urbanCanyon: ", error)
    finally:
        # closing database connection
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

def query():

    try:
        my_BuildingQuery = 'SELECT id, ST_AsText(geom), pand3d."roof-0.95"-pand3d."ground-0.30" as height ' \
                               'FROM pand3d WHERE pand3d."roof-0.95"-pand3d."ground-0.30" > 0;'

        cursor.execute(my_BuildingQuery)
        record_Building = cursor.fetchall()
        my_Buildings = []

        for tall in record_Building:
            ratio = 0.65
            width = (tall[2] / ratio)/2

            my_Record = {'id': tall[0], 'geometry': shapely.wkt.loads(tall[1]), 'height': tall[2], 'buffer_width':width}
            my_Buildings.append(my_Record)

        # plotSome_buildings(my_Buildings, "height", get_roads())

    except(Exception, psycopg2.Error) as error:
        print("Error while trying query: ", error)

    return my_Buildings

def explode_polygons(list_of_polygons):
    done = []

    for n, polygon in enumerate(list_of_polygons):
        coords = list(polygon.exterior.coords)

        for i, pt in enumerate(coords):
            if i == len(coords) - 1:
                continue
            else:
                line = shapely.geometry.LineString([pt, coords[i + 1]])
                my_dict = {'polygon_id': n}
                my_dict['line_id'] = i
                my_dict['geometry'] = line
                done.append(my_dict)

    return done

def merge(new):
    print('merging building %d' % new[0]['id'])
    # if len(new) > 1:
    translated = [list(item['new_geometry'].coords) for item in new]
    regular = [list(item['geometry'].coords) for item in new]
    t_chain = list(itertools.chain.from_iterable(translated))
    r_chain = list(itertools.chain.from_iterable(regular))
    r_chain.extend(list(reversed(t_chain)))
    poly = shapely.geometry.Polygon(r_chain).simplify(0)

    entry = {}
    entry['id'] = new[0]['id']
    entry['geometry'] = poly
    entry['height'] = new[0]['height']

    return entry

def flows(buildings, wind_direction, percentile):
    """
    creates shadows from leeward sides of the buildings.
    :return: building ids with shadow polygons
    :type: list with dicts
    """

    my_Buildings = buildings

    facades = windward(my_Buildings, wind_direction, plot=True)
    leeward_facades = [item for item in facades if item['windward'] == 0]
    # try: https://stackoverflow.com/questions/2612802/list-changes-unexpectedly-after-assignment-how-do-i-clone-or-copy-it-to-prevent

    for fac in leeward_facades:
        distance = (fac['height'] * percentile)
        dx = math.cos(math.radians(wind_direction)) * distance
        dy = math.sin(math.radians(wind_direction)) * distance
        fac['new_geometry'] = shapely.affinity.translate(fac['geometry'], dx, dy)
        fac['new_start'] = shapely.affinity.translate(fac['start'], dx, dy)
        fac['new_end'] = shapely.affinity.translate(fac['end'], dx, dy)

    #step 3: join connecting lines
    to_join = []
    shadow = []
    for shade in leeward_facades:
        if len(to_join) == 0:
            to_join.append(shade)
            latest_id = shade['id']

        elif shade['id'] == latest_id:
            if shade['segment_id'] == to_join[-1]['segment_id']+1:
                to_join.append(shade)

            else:
                # initiate to_join segments and empty to_join
                shadow.append(merge(to_join))
                to_join = []
                to_join.append(shade)
                print('new')


        elif shade['id'] != latest_id:
            print('new building')
            shadow.append(merge(to_join))
            to_join = []
            to_join.append(shade)
            latest_id = shade['id']

    shadow.append(merge(to_join))
            #initate to_join segments

    # gdf_windward = gpd.GeoDataFrame(facades, crs='epsg:28992')
    # ax = gdf_windward.plot(column='windward', k=2, cmap='viridis', legend=True, linewidth=0.5)
    # gpd.GeoDataFrame(shadow, crs='epsg:28992').set_index('id').plot(ax=ax, linewidth=0.5, color='k')
    # ax.set_xlim(91500, 92100)
    # ax.set_ylim(436000, 436500)
    # ax.set_xlim(89000, 89300)
    # ax.set_ylim(436250, 436500)
    # plt.show()

    print('banana')

    return shadow

def building_stats():
    my_buildings = all_buildingQuery()
    gdf_building = gpd.GeoDataFrame(my_buildings)
    fs=14
    h = gdf_building['height']
    # ax1 = h.plot.kde(bw_method=0.3)
    # ax1 = np.sqrt(h).plot.hist(bins=24)
    # ax1.set_xlabel('Height [m]', fontsize=fs)
    # plt.show()
    #
    # ax2 = np.log10(h).plot.hist(bins=24)
    # ax2.set_title('Logarithmic (log 10)')
    #

    # ax3 = np.sqrt(h).plot.kde(bw_method=0.3)
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    # ax3 = fig.add_subplot(spec[0,0])
    # ax3 = h.plot.kde(bw_method=0.3)
    #
    #
    # ax3.set_title('Square root')

    ax4 = fig.add_subplot(spec[0,:])
    ax4 = h.plot.hist(bins=256, color='red', edgecolor='k', alpha=0.65)
    ax4.set_title('Building height histogram')
    ax4.set_xlabel('Height [m]', fontsize=fs-2)
    ax4.set_ylabel('Frequency', fontsize=fs-2)
    ax4 = plt.axvline(h.mean(), color='c', linestyle='dashed', linewidth=1)
    ax4 = plt.annotate('mean: %s' % round(h.mean(),1), xy=(h.mean(), plt.ylim()[1]*0.9), fontsize=8, horizontalalignment='r', rotation=20, color='c')
    ax4 = plt.axvline(h.median(), color='firebrick', linestyle='dashed', linewidth=1)
    ax4 = plt.annotate('median: %s' % round(h.median(),1), xy=(h.median(), plt.ylim()[1]*0.75), fontsize=8, horizontalalignment='r', rotation=20, color='firebrick')

    ax5 = fig.add_subplot(spec[1,:])
    ax5.set_title('Boxplot of building height', fontsize=fs)
    ax5.set_xlabel('Height [m]', fontsize=fs-2)
    flierprops = dict(marker='o', markersize=4)
    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    ax5 = plt.boxplot(h, showfliers=True, vert=False, flierprops=flierprops, medianprops=medianprops)
    ax5 = plt.plot([23,23],[0,1], '--', marker='o', markevery=[1], ms=3)
    # ax5 = plt.xticks(list(plt.xticks()[0]) + [23])
    ax5 = plt.annotate('$height=23$', xy=(23,0.70), fontsize=8, horizontalalignment='r', rotation=20, color='b')
    # fig.savefig('pdf_hist_box.pdf', bbox_inches='tight')
    plt.show()
    print('hodl')

def roughness():
    # step 1: query
    # step 2: put in pandas and do; df['Max'] = df[['Communications','Business']].idxmax(axis=1)
    # step 3: dict with layer names and roughness length values
    # step 4: return to DB
    sql = "SELECT * FROM public.overlap_analysis"
    db_connection_url = "postgres://postgres:0000000@127.0.0.1:5432/thesisData"
    engine = create_engine(db_connection_url)
    gdf = gpd.read_postgis(sql, con=engine)
    column_names = ['bgt_area_gras_pc', 'bgt_area_plants_pc', 'bgt_area_shrub_pc', 'bgt_area_tree_pc', 'bgt_area_obg_erf_pc',
                    'bgt_area_obg_gesloten_pc', 'bgt_area_obg_half_pc', 'bgt_area_obg_onverhard_pc', 'bgt_area_obg_open_pc', 'bgt_area_wgd_gesloten_pc',
                    'bgt_area_wgd_half_pc', 'bgt_area_wgd_open_pc', 'bgt_waterdeel_pc']
    z0_values = {'bgt_area_gras_pc': 0.03, 'bgt_area_plants_pc': 0.1, 'bgt_area_shrub_pc': 0.25, 'bgt_area_tree_pc': 0.5,
                 'bgt_area_obg_erf_pc': 1, 'bgt_area_obg_gesloten_pc': 0.005, 'bgt_area_obg_half_pc': 0.005, 'bgt_area_obg_onverhard_pc': 0.03,
                 'bgt_area_obg_open_pc': 0.005, 'bgt_area_wgd_gesloten_pc': 0.005, 'bgt_area_wgd_half_pc': 0.005, 'bgt_area_wgd_open_pc': 0.005,
                 'bgt_waterdeel_pc': 0.0002}
    gdf['max'] = gdf[column_names].idxmax(axis=1)
    gdf['z0'] = gdf['max'].replace(to_replace=z0_values)
    selection = gdf[(gdf[column_names] == 0).all(1)]
    gdf['z0'].iloc[selection.index] = np.nan


    print('to_postgis')
    gdf.to_postgis(name='incl_roughness', con=engine, if_exists='replace')
    print('hodl ape')



def calc_score(gdf):
    wdir_list = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    gdf['average_height_neg'] = gdf['average_height'] * -1
    gdf['LH_neg'] = gdf['LH'] * -1
    thresholds_5 = {'average_height_neg':[-23, -15.7, -13.34, -10.3, -8.6],
                   'UC':[0.05, 0.4, 0.65, 1, 2],
                   'impact':[18, 36, 54, 72, 90],
                   # 'impact_180':[18, 36, 54, 72, 90],
                   # 'impact_250': [18, 36, 54, 72, 90],
                   # 'impact_290': [18, 36, 54, 72, 90],
                   }
    
    thresholds_3 = {'WL':[0.07, 0.56, 3.20],
                    # 'LH':[1.42, 3.69, 7.46]
                    'LH_neg':[-7.46, -3.69, -1.42]
                    }

    thresholds_2 = {'windward':[0,1],
                    # 'windward_180':[0, 1],
                    # 'windward_250':[1, 0],
                    # 'windward_290': [0, 1]
                     }
    thresholds_z = {'z0':[0.0002, 0.005, 0.03, 0.1, 0.25, 0.5, 1]
                    }

    # gdf['base_score'] = 0

    for wdir in wdir_list:
        columns = ['average_height_neg', 'UC', 'WL', 'LH_neg', 'z0']
        to_score = 'score_%s' % wdir
        gdf['score_%s' % wdir] = 0

        for key in thresholds_5:
            if key == 'impact':
                nkey_5 = 'impact_%s' % wdir
                columns.append(nkey_5)
            else:
                nkey_5 = key
            gdf[to_score].loc[gdf[nkey_5]<=thresholds_5[key][0]] += 5
            gdf[to_score].loc[gdf[nkey_5].between(thresholds_5[key][0], thresholds_5[key][1])] += 4
            gdf[to_score].loc[gdf[nkey_5].between(thresholds_5[key][1], thresholds_5[key][2])] += 3
            gdf[to_score].loc[gdf[nkey_5].between(thresholds_5[key][2], thresholds_5[key][3])] += 2
            gdf[to_score].loc[gdf[nkey_5].between(thresholds_5[key][3], thresholds_5[key][4])] += 1
            gdf[to_score].loc[gdf[nkey_5]>thresholds_5[key][4]] += 1
    
        for key in thresholds_3:
            gdf[to_score].loc[gdf[key]<=thresholds_3[key][0]] += 3
            gdf[to_score].loc[gdf[key].between(thresholds_3[key][0], thresholds_3[key][1])] += 2
            gdf[to_score].loc[gdf[key].between(thresholds_3[key][1], thresholds_3[key][2])] += 1
            gdf[to_score].loc[gdf[key]>thresholds_3[key][2]] += 1
    
        for key in thresholds_2:
            if key == 'windward':
                nkey_2 = 'windward_%s' % wdir
                columns.append(nkey_2)
            else:
                nkey_2 = key
            gdf[to_score].loc[gdf[nkey_2]==thresholds_2[key][0]] += 2
            gdf[to_score].loc[gdf[nkey_2]==thresholds_2[key][1]] += 1
    
        for key in thresholds_z:
            gdf[to_score].loc[gdf[key]==thresholds_z[key][0]] += 5
            gdf[to_score].loc[gdf[key]==thresholds_z[key][1]] += 4
            gdf[to_score].loc[gdf[key]==thresholds_z[key][2]] += 3
            gdf[to_score].loc[gdf[key]==thresholds_z[key][3]] += 2
            gdf[to_score].loc[gdf[key]>=thresholds_z[key][4]] += 1

        gdf.loc[gdf[columns].isna().any(1), 'score_%s' % wdir] = np.nan

    # pd.options.display.max_columns = None
    # print(gdf.head(20))
    # print('hodl')
    # # ['score_0', 'score_30', 'score_60', 'score_90', 'score_120', 'score_150',
    # # 'score_180', 'score_210', 'score_240', 'score_270', 'score_300', 'score_330']
    # gdf.loc[gdf.isnull(), ['score_0', 'score_30', 'score_60', 'score_90', 'score_120', 'score_150',
    # 'score_180', 'score_210', 'score_240', 'score_270', 'score_300', 'score_330']] = np.nan
    # print(gdf.isnull())
    # gdf.loc[gdf['average_height_neg'].isnull(), 'score_0'] = np.nan
    return gdf
    # for key in my_dict:
    #     print(gdf[key].describe())

def score():
    sql = "SELECT * FROM public.final_all_wdir"
    db_connection_url = "postgres://postgres:0000000@127.0.0.1:5432/thesisData"
    engine = create_engine(db_connection_url)
    gdf = gpd.read_postgis(sql, con=engine)
    # gdf = gdf[['id_2', 'geom', 'average_length', 'average_height', 'UC',
    #            'group_length', 'max', 'z0', 'azimuth', 'windward_250', 'impact_250',
    #            'windward_180', 'impact_180', 'windward_290', 'impact_290']]
    gdf = gdf[['id_3', 'geom', 'average_length', 'average_height', 'UC',
               'group_length', 'max', 'z0', 'azimuth',
               'windward_0', 'impact_0',
               'windward_30', 'impact_30',
               'windward_60', 'impact_60',
               'windward_90', 'impact_90',
               'windward_120', 'impact_120',
               'windward_150', 'impact_150',
               'windward_180', 'impact_180',
               'windward_210', 'impact_210',
               'windward_240', 'impact_240',
               'windward_270', 'impact_270',
               'windward_300', 'impact_300',
               'windward_330', 'impact_330',
               ]]
    gdf['WL'] = gdf['average_length'] / gdf['group_length']
    gdf['LH'] = gdf['group_length'] / gdf['average_height']
    # gdf[['impact_250', 'impact_180', 'impact_290']] = gdf[['impact_250', 'impact_180', 'impact_290']].abs()

    # my_scores = {'average_height': 5,
    #              'UC': 5,
    #              'WL': 3,
    #              'LH':3,
    #              'z0': 6,
    #              'windward_180':2,
    #              'windward_250':2,
    #              'windward_290':2,
    #              'impact_180':5,
    #              'impact_250':5,
    #              'impact_290':5,
    #              }

    calc_score(gdf).to_postgis(name='scored_nan_2', con=engine, if_exists='replace')

def score_table():
    print('hodl')
    table = pd.read_excel("/Users/wesseldejongh/Dropbox/MScThesis/Files/temp_table_values.xlsx", sheet_name="Sheet2")
    table['parameter/score'] = ["\textbf{"+x+"}" for x in table['parameter/score']]
    print(table.to_latex(na_rep="",escape=False, index=False, bold_rows=True, multirow=True, column_format="@{}l >{$}r<{$} >{$}r<{$} >{$}r<{$} >{$}r<{$} >{$}r<{$}@{}"))
def main():
    # starttime = time.time()
    """
    Main logic of the program. Calls all function and get user inputs.
    :return: 
    """

    #TODO: pick a building and surrounding buildings, including roads
    # Or pick a street and find surrounding buildings and streets

    # usefull parameters:
    wind_direction_compas = [200, 270, 160]
    # wind_direction = (450 - wind_direction_compas) % 360
    wind_direction = [(450 - x) % 360 for x in wind_direction_compas]
    percentile = 0.65
    wind_direction_top = [0, 30, 60, 90, 120, 150]

    wind_direction_bottom = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    # tall_buildings(get_roads(), plot=True)
    # fad(get_roads(), plot=True)
    # get_weather()
    # windward(tall_buildings(plot=None), wind_direction, plot=True)
    # urban_canyon_AR()
    # mine = get_bufferRoads()
    # my_shadows = flows(tall_buildings(), wind_direction, percentile)

    ##unfinshed:
    # create_table('what')
    # connecting_roads()

    ## Working on:
    # building_stats()

    start = time.time()
    # UC = urbanCanyon(all_buildingQuery())
    # consecutive_facades(all_buildingQuery(), wind_direction_bottom)
    # roughness()
    score()
    # score_table()
    elapsed_time_UC = (time.time() - start)
    print(elapsed_time_UC)
    print('baana')

    # my_query = 'SELECT simplified.id, ST_AsText(ST_ForcePolygonCCW(ST_CurveToLine(simplified.geom))), public.cbs_buurt_2019_gegeneraliseerd.id as buurt_id ' \
    #            'FROM public.simplified, public.cbs_buurt_2019_gegeneraliseerd ' \
    #            'WHERE ST_Intersects(ST_CurveToLine(simplified.geom), cbs_buurt_2019_gegeneraliseerd.geom); '
    #
    # cursor.execute(my_query)
    # record_simplified = cursor.fetchall()
    #
    # simplified = []
    #
    # for row in record_simplified:
    #     for n, item in enumerate(list(row[1])):
    #         coords = list(item.exterior.coords)
    #         feature = {'id_simp': row[0],
    #                    'id_n': n,
    #                    'geometry': list(shapely.wkt.loads(item)),
    #                    # 'geometry': shapely.wkt.loads(row[1]),
    #                    'buurt_id': row[2]
    #                    }
    #
    #         simplified.append(feature)
    #
    # ## to write windward to db and merge with with voronoi
    # # db_connection_url = "postgres://postgres:0000000@127.0.0.1:5432/thesisData"
    # # engine = create_engine(db_connection_url)
    # # wind_n = windward(simplified, wind_direction)
    # # gdf_done = gpd.GeoDataFrame(wind_n, crs='epsg:28992').set_index('id', drop=False)
    # # gdf_done.to_postgis(name='windward_%d_simp' % wind_direction, con=engine, if_exists='replace')


    # endtime = time.time()
    # duration = endtime - starttime
    # print("Runtime: ", round(duration, 2), "s")

    print('hodl ending, paper hands')

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
