
import numpy as np
from scipy.stats import norm, lognorm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


# Distance functions
def haversine(lon, lat, coords):
    '''
    Returns the haversine distance between a pair of coordinates and the coordinates of the grid
    
     Parameters
    ------------
        lon: int, Longitude of a point in the grid
        lat: int, Latitude of a point in the grid
        coords: tuple, Coordinates of the piece of information
            
    Return
    -----------
        distance : int
            Difference between pair of coordinates and the coordinates of the grid 
    
    
    
    ''' 
    lon1, lat1 = coords
    R = 6371  # Earth radius in km
    diff_lat = np.radians(lat1 - lat)
    diff_lon = np.radians(lon1 - lon)
    a = np.sin(diff_lat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat)) * np.sin(diff_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

    


def great_circle_distance(lon, lat, coords):
    
    '''
    Returns the great circle distance between a pair of coordinates and the coordinates of the grid
    
     Parameters
    ------------
        lon: int, Longitude of a point in the grid
        lat: int, Latitude of a point in the grid
        coords: tuple, Coordinates of the piece of information
            
    Return
    -----------
        distance : int
            Difference between pair of coordinates and the coordinates of the grid 
    '''
    
    
    
    
    lon1, lat1 = coords[0]
    lon2, lat2 = coords[1]
    R = 6371  # Earth radius in km
    diff_lat = np.radians(lat2 - lat1)
    diff_lon = np.radians(lon2 - lon1)
    a = np.sin(diff_lat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(diff_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    # Calculate the distance between each point on the grid and the great circle path
    distance_to_path = np.abs(distance * np.sin(np.radians(lon) - np.radians(coords[0][0])))
    return distance_to_path



# Probability functions 
def gaussian_prob(distance_to_thames):
    '''
    Returns a gaussian probability
    
     Parameters
    ------------
        distance_to_thames: int, distance (from the grid point) to thames (point)
                
    Return
    -----------
        pdf : int, probability distribution of the distance to thames
            
    '''

    pdf = norm.pdf(distance_to_thames, loc=0, scale=2730)
    return pdf


def lognormal_prob(distance_to_boe):
    '''
    Returns a lognormal probability
    
     Parameters
    ------------
        distance_to_boe: int, distance (from the grid point) to Bank of England (point)
                
    Return
    -----------
        pdf : int, probability distribution of the distance to Bank of England
            
    '''
    pdf = lognorm.pdf(distance_to_boe, s=0.625, scale=np.exp(8.460))
    return pdf


def normal_prob(distance_to_satellite_path):
    '''
    Returns a lognormal probability
    
     Parameters
    ------------
        distance_to_satellite: int, distance (from the grid point) to points of the satellite path
                
    Return
    -----------
        pdf : int, probability distribution of the distance to the satellite points of the path
            
    '''
    pdf = norm.pdf(distance_to_satellite_path, loc=0, scale=3160)
    return pdf
