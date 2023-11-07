# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:15:11 2021

Updated 19/07/2021

@author: Pratik Samant
"""
import pandas as pd
from dicompylercore import dicomparser,dvh
from dicompylercore.config import skimage_available
if skimage_available:
    from skimage.transform import rescale
from dvha.tools.roi_formatter import dicompyler_roi_coord_to_db_string, get_planes_from_string
import numpy as np
import copy
import numpy.ma as ma
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

import matplotlib.path
import collections
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
from six import iteritems

import logging
logger = logging.getLogger('dicompylercore.dvhcalc')


def IsodosePoints(rtss,rtdose,isodoselevelGy):
    """
    We don't strictly speaking need the rtss, however this rtss gives us
    information about which z planes we evaluate the rtdose at'

    Parameters
    ----------
    rtss : dicomparser.DicomParser
        The parsed dicom struct file (not filename).
    rtdose : dicomparser.DicomParser
        Parsed RT dose (not filename).
    isodoselevelGy : Float
        Threshold level (in Gy) of desired isodose.

    Returns
    -------
    isodosepoints : Array of float64
        Isodose points as a numpy array.

    """
    
    #Extract z coordinates of all planes by using body plane
    body_roi=get_structure_id(rtss.GetStructures(), 'BODY') #find the roi number of the structure called 'BODY'
    body_coords=rtss.GetStructureCoordinates(body_roi)
    plane_Zs=np.array(list(body_coords.keys()),dtype=np.float32) #extract and cast to a numpy array


    threshold=round(isodoselevelGy/float(rtdose.ds.DoseGridScaling)) #scale threshold to be in the dicom file's desired units. 

    isodosepoints=[None]*1000000 #allocate length of 1M to begin for the isodoses
    i=0 #initialize i
    
    for z in plane_Zs: #loop over all planes #z in dicompylercore actuall seems to correspond to y in eclipse. which is not ideal.
            isodoseZ=rtdose.GetIsodosePoints(z,threshold,0.1) # this will give list of (x,y) points
            isodoseZ=[point+(z,) for point in isodoseZ] # this adds z values to the (x,y) tuples
            
            if i+1+len(isodoseZ)>len(isodosepoints): #if the isodosepoints collection has run out of room, add more room.
                isodosepoints.append([None]*1000000)
            
            #if isodoseZ is not empty
            if isodoseZ:
                isodosepoints[i:i+len(isodoseZ)]=isodoseZ
                i=i+len(isodoseZ)

    isodosepoints=[point for point in isodosepoints if point !=None] #remove all points with none
    isodosepoints=np.array(isodosepoints)
    
    return isodosepoints

def IsodoseCentroid(rtss,rtdose,isodoselevelGy):
    """
    

    Parameters
    ----------
    rtss : dicomparser.DicomParser
        The parsed dicom struct file (not filename).
    rtdose : dicomparser.DicomParser
        Parsed RT dose (not filename).
    isodoselevelGy : Float
        Threshold level (in Gy) of desired isodose.
    Returns
    -------
    centroid : Array of float64
        Centroid of isodose.

    """
    isodosepoints=IsodosePoints(rtss,rtdose,isodoselevelGy)
    
    centroid=np.around([np.mean(isodosepoints[:,0]),
                        np.mean(isodosepoints[:,1]),
                        np.mean(isodosepoints[:,2])
                        ],2
                       )
    return centroid



def Centroid(rtss,contour):
    """
    Returns centroid of a contour

    Parameters
    ----------
    rtss : dicomparser.DicomParser
        parsed RTSS.
    contour : String
        contour name.

    Returns
    -------
    centroid : numpy array
        centroid of contour.

    """
    vertices=Get_Vertices(rtss, oar=contour)
    centroid=np.around([np.mean(vertices[:,0]),
                        np.mean(vertices[:,1]),
                        np.mean(vertices[:,2])
                        ],2
                       )
    return centroid


def RT_FeatureExtract(rtss_path,
                      rtdose_path,
                      PTVs_list=['PTV'], 
                      OARs_list=['VISCERALOAR_3CM'], 
                      PTV_Metric_Names=['V40Gy'],
                      OAR_Metric_Names=['D0.1cc','D0.5cc','D1cc','D5cc','D10cc','D20cc','V36Gy'],
                      shift_vector=[0,0,0],
                      n=0,):
    """
    This function extractes RT features of interest given a dose and struct file.
    If needed, the function can apply a shift vector to all structs in the struct file prior
    to DVH calculation and feature extraction.
    
    The features extracted are named in the two lists OAR_Metric_Names and 
    PTV_Metric_Names.The function will extract the features in OAR_Metric_Names
    for all OARs inthe OARs_list input variable. It will also extract all the 
    features in PTV_Metric_Names for the PTV. If PTVHIGH and PTVLOW are also
    wanted, then the appropriate line defining PTVs_list can be commented/uncommented.
    
    
   The shift vector is usually only relevant (aka nonzero) in the case 
   when nonadaptive plan features are wanted given the adaptive rt-struct file
   and nonadaptive rt-dose and rt-struct files. In this case, shift vector of 
   translation such that applying this translation to the adaptive structure
   set will maximize GTV overlap with the base structure set. Such a shift
   vector can be found using the find_GTV_overlap_vector function later in this
   toolbox.

    Parameters
    ----------
    rtss_path : string
        Path to rtss file.
    rtdose_path : string
        Path to rtdose file.
    PTVs_list : list of strings, optional
        Full list of all PTV names (in case there is more than one PTV contour of interest.
                                    The default is ['PTV'].
    OARs_list : list of strings, optional
        Full list of all OAR names. The default is ['VISCERALOAR_3CM'].
    PTV_Metric_Names : list of strings, optional
        Full list of all metrics of interest for the PTV. The default is ['V40Gy'].
    OAR_Metric_Names : list of strings, optional
        Full list of all OAR metric names. The default is ['D0.1cc','D0.5cc','D1cc','D5cc','D10cc','D20cc','V36Gy'].
    shift_vector : list, optional
        Shift vector in list formatted [x ,y ,z]. The default is [0,0,0].
    n : int, optional
        Power of n to interpolate if interpolation is performed.. Default is 0
    Returns. The default is 0.

    Returns
    -------
    Features_df : pandas Dataframe
        Pandas dataframe returning all features in a single row..

    """

    #rtdose=dicomparser.DicomParser(rtdose_path)
    rtss = dicomparser.DicomParser(rtss_path)

    
    # Get a dict of structure information
    structures = rtss.GetStructures()
    OAR_ids=[]
    PTV_ids=[]
    
    #extract ids for organs of interest and PTVs
    for key in structures.keys():
        if structures[key]['name'] in OARs_list:
            OAR_ids.append(structures[key]['id'])
        elif structures[key]['name'] in PTVs_list:
            PTV_ids.append(structures[key]['id'])
    
    
    
    columns=[]
    metrics=[]
    #Feature Extract
    for ptv_name in PTVs_list: #for all PTVs
        
        #extract dvh
        if n>0:
            ptv_dvh=get_dvh(rtss_path, rtdose_path, ptv_name,shift_vector=shift_vector,n=n)
        else:
            ptv_dvh=get_dvh(rtss_path, rtdose_path, ptv_name,shift_vector=shift_vector)
    
        #create list of metric names, called columns
        columns.extend([ptv_name+'_'+column for column in PTV_Metric_Names])
        
        metrics.extend([ptv_dvh.statistic(i).value for i in PTV_Metric_Names]) #this extracts the doses all in one list
    
        
    
    for oar_name in OARs_list: #for all OARs
        
    
        
        #extract dvh
        if n>0:
            structure_dvh=get_dvh(rtss_path, rtdose_path, oar_name,shift_vector=shift_vector,n=n)
        else:
            structure_dvh=get_dvh(rtss_path, rtdose_path, oar_name,shift_vector=shift_vector)
    
        #create list of metric names, called Columns
        columns.extend([oar_name+'_'+column for column in OAR_Metric_Names])
        
        #Put features in the metrics, here we extract all metrics as above
        metrics.extend([structure_dvh.statistic(i).value for i in OAR_Metric_Names])
    
    #put everything into a dataframe
    Features_df=pd.DataFrame([metrics],columns=columns)
    
    #[optional] add the patient id as a row and set it to be the index of the dataframe
    #Features_df['Patient_ID']=[rtdose.ds.PatientID]
    
    #Features_df=Features_df.set_index('Patient_ID')
    
    return Features_df

    
def Translate(structure_coords,shift_vector):
    """
    This applies a shift vector to extracted structure coords

    Parameters
    ----------
    structure_coords : dict
        dicompyler structure coordinates from GetStructureCoordinates().
    shift_vector : list
        A list of size 3, specifying the  shift vector as [x,y,z]. If this
        is used to compute overlap, then the z value *must* be evenly divisible by 
        slice thickness. i.e. if slice thickness is 3.5 then z must be an integer
        multiple of 3.5. Otherwise the volume overlap function will incorrectly
        evaluate to 0.


    Returns
    -------
    translated_structure_coords : dict
        translated dicompyler structure coordinates, formatted in same was as structure_coords.

    """
    translated_structure_coords={}
    
    for key,plane in list(structure_coords.items()): #iterate over dictionary, k are keys representing each plane
        new_key='{:0.2f}'.format(float(key)+shift_vector[2]) #the new key should match the z location, retain 2 decimal places
        #print(key)
        plane_copy=copy.deepcopy(plane)
        
        #we want to now access plane[0]['data'] and change all the values by shift vector
        data=plane_copy[0]['data'] #extract data 
        
        #loop over all points
        for point_index in range(0,len(data)): #loop over all point triplets
            #add shift vector to all points in data by converting to numpy, then convert back to string
            data[point_index]=[str(i) for i in list(np.array(data[point_index])+np.array(shift_vector))] 
        
        translated_structure_coords[new_key]=plane_copy
    
    return translated_structure_coords


def get_structure_id(structures,structure_name):
    """
    Return structure index from structures dictionary and name

    Parameters
    ----------
    structures : dict
        structures dictionary extracted with dicompuler .GetStructures method
        after DICOM parsing.
    structure_name : str
        name of structure.
        
    Returns
    -------
    int
        id number of structure.

    """
    
    for k, v in structures.items():
        if v["name"] == structure_name:
           return v["id"]
       
        
      
def GetPlane(parsed_rt_struct,structure_name):
    """
    Return a "sets of points" formatted dictionary from structure_name string
    and parsed rt_struct

    Parameters
    ----------
    parsed_rt_struct : dicomparser.DicomParser
        The parsed dicom struct file, this can be generated by calling
        dicomparser.DicomParser(filename).
    structure_name : str
        name of structure.

    Returns
    -------
    structure_plane : list
        a "sets of points" formatted dictionary.

    """
    structure_id=get_structure_id(parsed_rt_struct.GetStructures(), structure_name)
    structure_plane=get_planes_from_string(dicompyler_roi_coord_to_db_string(parsed_rt_struct.GetStructureCoordinates(structure_id)))
    return structure_plane
        

def find_GTV_overlap_vector(initial_guess,base_rtss_path,frac_rtss_path,oarbase='GTV',oarada='GTV'):
    """
    This script takes a base and fraction rt-struct and computes the translation vector
    that, when applied to the fraction, will maximize GTV overlap with the base. The negative
    of this vector can be added to the base RT-dose to create a plan for the fraction. Alternately,
    The vector itself can be added to the fraction verts to overlap with the base dose cube. The latter
    approach is what we usually take.

    Parameters
    ----------
    initial_guess : numpy ndarray of size 3
        Intial guess for vector of translation, which will be applied to the ada_gtv_vertices.
    base_rtss_path : str
        Path of base rtstruct file
    frac_rtss_path : str
        path of fraction rtstruct file.
    oarbase : str
        name of structure in base rtss, default is 'GTV'
    oarfrac : str
        name of structure in frac rtss, default is 'GTV'        


    Returns
    -------
    Array of float64
         An array of size 3 specifying the vector that maximized gtv overlap.

    """

    #load vertices of base and adaptive fraction gtv
    base_vertices=Get_Vertices(dicomparser.DicomParser(base_rtss_path),oar=oarbase)
    ada_vertices=Get_Vertices(dicomparser.DicomParser(frac_rtss_path),oar=oarada)
    
    Min_Obj=minimize(hull_volume_translate,initial_guess,args=(base_vertices,
                                                               ada_vertices))
    return Min_Obj.x

def find_overlap_vector(initial_guess,base_vertices, frac_vertices):
    """
    This script takes a pair of vertex sets, and then finds a shift vector
    that when applied to frac_vertices, maximizes overlap with base_vertices.
    Parameters
    ----------
    initial_guess : numpy ndarray of size 3
        Intial guess for vector of translation, which will be applied to the ada_gtv_vertices.
    base_vertices : np.array
        array of coordinate triplets containing base contour vertices.
    frac_vertices : np.array
        array of coordinate triplets containing frac contour vertices. 


    Returns
    -------
    Array of float64
         An array of size 3 specifying the vector that maximized gtv overlap.

    """

    Min_Obj=minimize(hull_volume_translate,initial_guess,args=(base_vertices,
                                                               frac_vertices))
    return Min_Obj.x


def Get_Vertices(rtss,oar='GTV'):
    """
    Get contour vertices. By default this is for GTV only

    Parameters
    ----------
    rtss : dicomparser.Dicomparser (parsed dicomparser object)
        Parsed RT-STRUCT.
    oar : str, optional
        OAR name. The default is 'GTV'.

    Returns
    -------
    verts3d : np.array
        array of coordinate triplets containing contour vertices.

    """
    planes=GetPlane(rtss, oar)
    
    verts3d=[]
    for plane_key in planes:
        plane_values=planes[plane_key][0]
        for entry in plane_values: 
            verts3d.append(entry)    
    return np.array(verts3d)

def hull_volume_translate(shift_vector,base_verts,ada_verts):
    """
    Applies a translation vector to contour, then computes convex hull of that contour;s
    points with the base contour (TP0 GTV)'s points. Can be used with an optimizer to find
    shift vector that minimizes convex hull volume (this is the same as maximizing overlap)

    Parameters
    ----------
    shift_vector : numpy ndarray
        Vector of translation, which will be applied to the ada_gtv_vertices.
    base_gtv_vertis : numpy ndarray
        List of coordinate triplets containing contour vertices (TP0).
    ada_gtv_verts : numpy ndarray
         List of coordinate triplets containing contour vertices (fraction).

    Returns
    -------
    volume : float
        Volume of convex hull after translation vector is applied to adaptive vertices.

    """
    #translate the adaptive fraction vertices and concatenate with base vertices
    ada_verts_translated=ada_verts+shift_vector
    allpoints=np.concatenate((base_verts,ada_verts_translated))
    #compute volume and return
    cv=ConvexHull(allpoints)
    volume=cv.volume
    return volume

def hull_volume_translate_verts(base_verts,ada_verts,shift_vector=[0,0,0]):
    """
    Applies a translation vector to contour, then computes convex hull of that contour;s
    points with the base contour (TP0 GTV)'s points. Can be used with an optimizer to find
    shift vector that minimizes convex hull volume (this is the same as maximizing overlap).
    Can be used to compute volume of CV if the shift_vector is set to [0,0,0]

    Parameters
    ----------
    base_gtv_vertices : numpy ndarray
        List of coordinate triplets containing contour vertices (TP0).
    secondOrganVerts : numpy ndarray
         List of coordinate triplets containing contour vertices (fraction).
    shift_vector : numpy ndarray
        Vector of translation, which will be applied to the ada_gtv_vertices.
    Returns
    -------
    volume : float
        Volume of convex hull after translation vector is applied to adaptive vertices.

    """
    #translate the adaptive fraction vertices and concatenate with base vertices
    ada_verts_translated=ada_verts+shift_vector
    allpoints=np.concatenate((base_verts,ada_verts_translated))
    #compute volume and return
    volume=ConvexHull(allpoints).volume
    return volume


def hull_volume_compute(firstOrganVerts,secondOrganVerts):
    """
    Computes the volume of the convex hull of two sets of vertices

    Parameters
    ----------
    firstOrganVerts : numpy ndarray
        List of coordinate triplets containing contour vertices (TP0).
    secondOrganVerts : numpy ndarray
         List of coordinate triplets containing contour vertices (fraction).

    Returns
    -------
    volume : float
        Volume of convex hull.

    """
    #translate the adaptive fraction vertices and concatenate with base vertices
    allpoints=np.concatenate((firstOrganVerts,secondOrganVerts))
    #compute volume and return
    volume=ConvexHull(allpoints).volume
    return volume



#%%Imported functions directly from dicompyler-core, these are used for tweaked code in calculating dvh's
def get_dvh(structure,
            dose,
            roi_name,
            shift_vector=[0,0,0],
            limit=None,
            calculate_full_volume=True,
            use_structure_extents=False,
            n=0,
            interpolation_segments_between_planes=0,
            thickness=None,
            callback=None):
    """Calculate a cumulative DVH in Gy from a DICOM RT Structure Set & Dose.

    Parameters
    ----------
    structure : str
        path to rtss dicom file.
    dose : str
        path to rtdose dicom file.
    roi_name: str
        name of structure.
    shift_vector: list
        vector of translation to apply to rtss prior to DVH calculation. The default is [0,0,0]
    limit : int, optional
        Dose limit in cGy as a maximum bin for the histogram.
    calculate_full_volume : bool, optional
        Calculate the full structure volume including contours outside of the
        dose grid.
    use_structure_extents : bool, optional
        Limit the DVH calculation to the in-plane structure boundaries.
    n : int, optional
        Power of n in interpolation, if n=0 then interpolation_resolution is set to None.
    interpolation_segments_between_planes : integer, optional
        Number of segments to interpolate between structure slices.
    thickness : float, optional
        Structure thickness used to calculate volume of a voxel.
    callback : function, optional
        A function that will be called at every iteration of the calculation.
    """
    #from dicompylercore import dicomparser
    rtss = dicomparser.DicomParser(structure)
    rtdose = dicomparser.DicomParser(dose)
    structures = rtss.GetStructures()
    roi=get_structure_id(structures, roi_name)
    
    
    #determine interpolation resolution
    if n>0:
        dose_data=rtdose.GetDoseData()
        pixel_spacing=rtdose.GetDoseData()['pixelspacing']
        
        if pixel_spacing[0]==pixel_spacing[1]:
            interpolation_resolution=pixel_spacing[0]/2**n
        else:
            interpolation_resolution=(pixel_spacing[0]/2**n,pixel_spacing[1]/2**n)
    else:
        interpolation_resolution=None
            
    
    
    s = structures[roi]
    #apply translation if shift_vector is nonzero
    if not shift_vector==[0,0,0]: #if nonzero shift vector, shift the RTSS structures by the shift vector, which brings it to overlap with the base rtdose
        s['planes'] = Translate(rtss.GetStructureCoordinates(roi), shift_vector)
    else:
        s['planes']=rtss.GetStructureCoordinates(roi)
        
    s['thickness'] = thickness if thickness else rtss.CalculatePlaneThickness(s['planes'])

    calcdvh = calculate_dvh(s, rtdose, limit, calculate_full_volume,
                            use_structure_extents, interpolation_resolution,
                            interpolation_segments_between_planes,
                            callback)
    return dvh.DVH(counts=calcdvh.histogram,
                   bins=(np.arange(0, 2) if (calcdvh.histogram.size == 1) else
                         np.arange(0, calcdvh.histogram.size + 1) / 100),
                   dvh_type='differential',
                   dose_units='Gy',
                   notes=calcdvh.notes,
                   name=s['name']).cumulative


def calculate_dvh(structure,
                  dose,
                  limit=None,
                  calculate_full_volume=True,
                  use_structure_extents=False,
                  interpolation_resolution=None,
                  interpolation_segments_between_planes=0,
                  callback=None):
    """Calculate the differential DVH for the given structure and dose grid.

    Parameters
    ----------
    structure : dict
        A structure (ROI) from an RT Structure Set parsed using DicomParser
    dose : DicomParser
        A DicomParser instance of an RT Dose
    limit : int, optional
        Dose limit in cGy as a maximum bin for the histogram.
    calculate_full_volume : bool, optional
        Calculate the full structure volume including contours outside of the
        dose grid.
    use_structure_extents : bool, optional
        Limit the DVH calculation to the in-plane structure boundaries.
    interpolation_resolution : float, optional
        Resolution in mm to interpolate the structure and dose data to.
    interpolation_segments_between_planes : integer, optional
        Number of segments to interpolate between structure slices.
    callback : function, optional
        A function that will be called at every iteration of the calculation.
    """
    planes = collections.OrderedDict(sorted(iteritems(structure['planes'])))
    calcdvh = collections.namedtuple('DVH', ['notes', 'histogram'])
    logger.debug("Calculating DVH of %s %s", structure['id'],
                 structure['name'])

    # Create an empty array of bins to store the histogram in cGy
    # only if the structure has contour data or the dose grid exists
    if ((len(planes)) and ("PixelData" in dose.ds)):

        # Get the dose and image data information
        dd = dose.GetDoseData()
        id = dose.GetImageData()

        # Determine structure and respectively dose grid extents
        if interpolation_resolution or use_structure_extents:
            extents = []
            if use_structure_extents:
                extents = structure_extents(structure['planes'])
            dgindexextents = dosegrid_extents_indices(extents, dd)
            dgextents = dosegrid_extents_positions(dgindexextents, dd)
            # Determine LUT from extents
            if use_structure_extents:
                dd['lut'] = \
                    (dd['lut'][0][dgindexextents[0]:dgindexextents[2]],
                     dd['lut'][1][dgindexextents[1]:dgindexextents[3]])
            # If interpolation is enabled, generate new LUT from extents
            if interpolation_resolution:
                dd['lut'] = get_resampled_lut(
                    dgindexextents,
                    dgextents,
                    new_pixel_spacing=interpolation_resolution,
                    min_pixel_spacing=id['pixelspacing'][0])
            dd['rows'] = dd['lut'][1].shape[0]
            dd['columns'] = dd['lut'][0].shape[0]

        # Generate a 2d mesh grid to create a polygon mask in dose coordinates
        # Code taken from Stack Overflow Answer from Joe Kington:
        # https://stackoverflow.com/q/3654289/74123
        # Create vertex coordinates for each grid cell
        x, y = np.meshgrid(np.array(dd['lut'][0]), np.array(dd['lut'][1]))
        x, y = x.flatten(), y.flatten()
        dosegridpoints = np.vstack((x, y)).T

        #default option
        maxdose = int(dd['dosemax'] * dd['dosegridscaling'] * 100) + 1

        # Remove values above the limit (cGy) if specified
        if isinstance(limit, int):
            if (limit < maxdose):
                maxdose = limit
        hist = np.zeros(maxdose)
    else:
        return calcdvh('Empty DVH', np.array([0]))

    n = 0
    notes = None
    planedata = {}
    # Interpolate between planes in the direction of the structure
    if interpolation_segments_between_planes:
        planes = interpolate_between_planes(
            planes, interpolation_segments_between_planes)
        # Thickness derived from total number of segments relative to original
        structure['thickness'] = structure[
            'thickness'] / (interpolation_segments_between_planes + 1)

    # Iterate over each plane in the structure
    for z, plane in iteritems(planes):
        # Get the dose plane for the current structure plane
        if interpolation_resolution or use_structure_extents:
            doseplane = get_interpolated_dose(
                dose, z, interpolation_resolution, dgindexextents)
        else:
            doseplane = dose.GetDoseGrid(z)
        if doseplane.size:
            planedata[z] = calculate_plane_histogram(plane, doseplane,
                                                     dosegridpoints, maxdose,
                                                     dd, id, structure, hist)
            # print(f'Slice: {z}, volume: {planedata[z][1]}')
        else:
            # If the dose plane is not found, still perform the calculation
            # but only use it to calculate the volume for the slice
            if not calculate_full_volume:
                logger.warning('Dose plane not found for %s. Contours' +
                               ' not used for volume calculation.', z)
                notes = 'Dose grid does not encompass every contour.' + \
                    ' Volume calculated within dose grid.'
            else:
                origin_z = id['position'][2]
                logger.warning('Dose plane not found for %s.' +
                               ' Using %s to calculate contour volume.',
                               z, origin_z)
                _, vol = calculate_plane_histogram(
                    plane, dose.GetDoseGrid(origin_z), dosegridpoints, maxdose,
                    dd, id, structure, hist)
                planedata[z] = (np.array([0]), vol)
                notes = 'Dose grid does not encompass every contour.' + \
                    ' Volume calculated for all contours.'
        n += 1
        if callback:
            callback(n, len(planes))
    # Volume units are given in cm^3
    volume = sum([p[1] for p in planedata.values()]) / 1000
    # print(f'total volume: {volume}')
    # Rescale the histogram to reflect the total volume
    hist = sum([p[0] for p in planedata.values()])
    if hist.max() > 0:
        hist = hist * volume / sum(hist)
    else:
        return calcdvh('Empty DVH', np.array([0]))
    # Remove the bins above the max dose for the structure
    hist = np.trim_zeros(hist, trim='b')

    return calcdvh(notes, hist)


def calculate_plane_histogram(plane, doseplane, dosegridpoints, maxdose, dd,
                              id, structure, hist):
    """Calculate the DVH for the given plane in the structure."""
    contours = [[x[0:2] for x in c['data']] for c in plane]

    # Create a zero valued bool grid
    grid = np.zeros((dd['rows'], dd['columns']), dtype=np.uint8)

    # Calculate the dose plane mask for each contour in the plane
    # and boolean xor to remove holes
    for i, contour in enumerate(contours):
        m = get_contour_mask(dd, id, dosegridpoints, contour)
        grid = np.logical_xor(m.astype(np.uint8), grid).astype(np.bool)

    hist, vol = calculate_contour_dvh(grid, doseplane, maxdose, dd, id,
                                      structure)
    return (hist, vol)


def get_contour_mask(dd, id, dosegridpoints, contour):
    """Get the mask for the contour with respect to the dose plane."""
    doselut = dd['lut']

    c = matplotlib.path.Path(list(contour))

    # def inpolygon(polygon, xp, yp):
    #     return np.array(
    #         [Point(x, y).intersects(polygon) for x, y in zip(xp, yp)],
    #         dtype=np.bool)

    # p = Polygon(contour)
    # x, y = np.meshgrid(np.array(dd['lut'][0]), np.array(dd['lut'][1]))
    # mask = inpolygon(p, x.ravel(), y.ravel())
    # return mask.reshape((len(doselut[1]), len(doselut[0])))

    grid = c.contains_points(dosegridpoints)
    grid = grid.reshape((len(doselut[1]), len(doselut[0])))

    return grid


def calculate_contour_dvh(mask, doseplane, maxdose, dd, id, structure):
    """Calculate the differential DVH for the given contour and dose plane."""
    # Multiply the structure mask by the dose plane to get the dose mask
    mask = ma.array(doseplane * dd['dosegridscaling'] * 100, mask=~mask)
    # Calculate the differential dvh
    hist, edges = np.histogram(mask.compressed(),
                               bins=maxdose,
                               range=(0, maxdose))

    # Calculate the volume for the contour for the given dose plane
    vol = sum(hist) * ((np.mean(np.diff(dd['lut'][0]))) *
                       (np.mean(np.diff(dd['lut'][1]))) *
                       (structure['thickness']))
    return hist, vol


def structure_extents(coords):
    """Determine structure extents in patient coordinates.

    Parameters
    ----------
    coords : dict
        Structure coordinates from dicomparser.GetStructureCoordinates.

    Returns
    -------
    list
        Structure extents in patient coordintes: [xmin, ymin, xmax, ymax].
    """
    bounds = []
    for z in sorted(coords.items()):
        contours = [[x[0:2] for x in c['data']] for c in z[1]]
        for contour in contours:
            x, y = np.array([x[0:1] for x in contour]), np.array(
                [x[1:2] for x in contour])
            bounds.append([np.min(x), np.min(y), np.max(x), np.max(y)])
    extents = np.array(bounds)
    return np.array(
        [np.amin(extents, axis=0)[0:2],
         np.amax(extents, axis=0)[2:4]]).flatten().tolist()


def dosegrid_extents_indices(extents, dd, padding=1):
    """Determine dose grid extents from structure extents as array indices.

    Parameters
    ----------
    extents : list
        Structure extents in patient coordintes: [xmin, ymin, xmax, ymax].
        If an empty list, no structure extents will be used in the calculation.
    dd : dict
        Dose data from dicomparser.GetDoseData.
    padding : int, optional
        Pixel padding around the structure extents.

    Returns
    -------
    list
        Dose grid extents in pixel coordintes as array indices:
        [xmin, ymin, xmax, ymax].
    """
    if not len(extents):
        return [0, 0, dd['lut'][0].shape[0] - 1, dd['lut'][1].shape[0] - 1]
    dgxmin = np.argmin(np.fabs(dd['lut'][0] - extents[0])) - padding
    if dd['lut'][0][dgxmin] > extents[0]:
        dgxmin -= 1
    dgxmax = np.argmin(np.fabs(dd['lut'][0] - extents[2])) + padding
    dgymin = np.argmin(np.fabs(dd['lut'][1] - extents[1])) - padding
    dgymax = np.argmin(np.fabs(dd['lut'][1] - extents[3])) + padding
    dgxmin = 0 if dgxmin < 0 else dgxmin
    dgymin = 0 if dgymin < 0 else dgymin
    if dgxmax == dd['lut'][0].shape[0]:
        dgxmax = dd['lut'][0].shape[0] - 1
    if dgymax == dd['lut'][1].shape[0]:
        dgymax = dd['lut'][1].shape[0] - 1
    return [dgxmin, dgymin, dgxmax, dgymax]


def dosegrid_extents_positions(extents, dd):
    """Determine dose grid extents in patient coordinate indices.

    Parameters
    ----------
    extents : list
        Dose grid extents in pixel coordintes: [xmin, ymin, xmax, ymax].
    dd : dict
        Dose data from dicomparser.GetDoseData.

    Returns
    -------
    list
        Dose grid extents in patient coordintes: [xmin, ymin, xmax, ymax].
    """
    return [
        dd['lut'][0][extents[0]], dd['lut'][1][extents[1]],
        dd['lut'][0][extents[2]], dd['lut'][1][extents[3]]
    ]


def get_resampled_lut(index_extents,
                      extents,
                      new_pixel_spacing,
                      min_pixel_spacing):
    """Determine the patient to pixel LUT based on new pixel spacing.

    Parameters
    ----------
    index_extents : list
        Dose grid extents as array indices.
    extents : list
        Dose grid extents in patient coordinates.
    new_pixel_spacing : float
        New pixel spacing in mm
    min_pixel_spacing : float
        Minimum pixel spacing used to determine the new pixel spacing

    Returns
    -------
    tuple
        A tuple of lists (x, y) of patient to pixel coordinate mappings.

    Raises
    ------
    AttributeError
        Raised if the new pixel_spacing is not a factor of the minimum pixel
        spacing.

    Notes
    -----
    The new pixel spacing must be a factor of the original (minimum) pixel
    spacing. For example if the original pixel spacing was ``3`` mm, the new
    pixel spacing should be: ``3 / (2^n)`` mm, where ``n`` is an integer.

    Examples
    --------
    Original pixel spacing: ``3`` mm, new pixel spacing: ``0.375`` mm
    Derived via: ``(3 / 2^16) == 0.375``

    """
    if (min_pixel_spacing % new_pixel_spacing != 0.0):
        raise AttributeError(
            "New pixel spacing must be a factor of %g/(2^n),"
            % min_pixel_spacing +
            " where n is an integer. Value provided was %g."
            % new_pixel_spacing)
    sampling_rate = np.array([
        abs(index_extents[0] - index_extents[2]),
        abs(index_extents[1] - index_extents[3])
    ])
    xsamples, ysamples = sampling_rate * min_pixel_spacing / new_pixel_spacing
    x = np.linspace(extents[0], extents[2], int(xsamples), dtype=np.float)
    y = np.linspace(extents[1], extents[3], int(ysamples), dtype=np.float)
    return x, y


def get_interpolated_dose(dose, z, resolution, extents):
    """Get interpolated dose for the given z, resolution & array extents.

    Parameters
    ----------
    dose : DicomParser
        A DicomParser instance of an RT Dose.
    z : float
        Index in mm of z plane of dose grid.dose
    resolution : float
        Interpolation resolution less than or equal to dose grid pixel spacing.
    extents : list
        Dose grid index extents.

    Returns
    -------
    ndarray
        Interpolated dose grid with a shape larger than the input dose grid.
    """
    # Return the dose bounded by extents if interpolation is not required
    d = dose.GetDoseGrid(z)
    extent_dose = d[extents[1]:extents[3],
                    extents[0]:extents[2]] if len(extents) else d
    if not resolution:
        return extent_dose
    if not skimage_available:
        raise ImportError(
            "scikit-image must be installed to perform DVH interpolation.")
    scale = (np.array(dose.ds.PixelSpacing) / resolution).tolist()
    interp_dose = rescale(
        extent_dose,
        scale=scale,
        mode='symmetric',
        order=1,
        preserve_range=True)
    return interp_dose


def interpolate_between_planes(planes, n=2):
    """Interpolate n additional structure planes (segments) in between planes.

    Parameters
    ----------
    planes : dict
        RT Structure plane data from dicomparser.GetStructureCoordinates.
    n : int, optional
        Number of planes to interpolate in between the existing planes.

    Returns
    -------
    dict
        Plane data with additional keys representing interpolated planes.
    """
    keymap = {np.array([k], dtype=np.float32)[0]: k for k in planes.keys()}
    sorted_keys = np.sort(np.array(list(planes.keys()), dtype=np.float32))
    num_new_samples = (len(planes.keys()) * (n + 1)) - n
    newgrid = np.linspace(sorted_keys[0], sorted_keys[-1], num_new_samples)
    new_planes = {}
    # If the plane already exists in the dictionary, use it
    # otherwise use the closest plane
    # TODO: Add actual interpolation of structure data between planes
    for plane in newgrid:
        new_plane = sorted_keys[np.argmin(np.fabs(sorted_keys - plane))]
        new_planes[plane] = planes[keymap[new_plane]]
    return new_planes
