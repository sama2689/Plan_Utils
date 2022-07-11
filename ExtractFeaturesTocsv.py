"""
Created on Mon Jul 12 11:26:45 2021

Updated 19/07/2021

@author: Pratik Samant
"""

import pandas as pd
import numpy as np
import Plan_Utils as plu
import os
from dicompylercore import dicomparser
import csv


# =============================================================================
# This script loops over all patients in a directory and extracts DVH features for base, 
# adaptive, and nonadaptive fractions. It finds patients by first looking for all sub-directories
# that begin with 'ANON' (adjustable in the code)
# The script then looks inside these subdirectories for a folder called 'TP0_SIMUL' to find the 
# folder with base rt struct and rt dose info. Similarly, it looks for folders 
# called 'TPN_FRACN' where N is the fraction number to find fraction RT dose and 
# RT Struct files. all of these can be changed by changing the relevant lines. What the script looks for, the list of PTVs it requests (if there is more than one named PTV),
# the list of OARs for which metrics are wanted, and the PTV and OAR metrics can all be changed. 

# Whether or not interpolation is performed can be checked with the n value when RT_FeatureExtract is called, if the value n>0 then some interpolation is performed


#If you have any difficulties deploying the script or accompanying toolbox please contact Pratik Samant at pratik.samant@ouh.nhs.uk
# =============================================================================

#this is the interpolation value, n must be an integer and if n>0 then the computation time increased by a lot
n=2
#%%extract directory containing all patients
rootpath=os.getcwd()

##save shift vectors option (True if you want the code to save shift vectors, false otherwise)
save_sv=True

#Adjustable: get list of patient directories by getting directories that start with 'ANON'
patient_dirs_list=[subdir for subdir in os.listdir() if subdir.startswith('AU_PA')]

#list ptv names, oar names, ptv metrics, and oar metrics to extract
PTVs_list_orig=['PTV']
OARs_list=['VISCERALOAR_3CM']
PTV_Metric_Names=['D25','D50','D75','D99','V40Gy']
OAR_Metric_Names=['D1cc','D5cc','D10cc','D20cc','V40Gy']

PTVs_list=['PTV']
#%%loop over all patient directories (thereby looping over all patients)
for patient_dir in [patient_dirs_list[1]]:
    
    if patient_dir=='AU_PA_0009':
        PTVs_list=['PTV_4000']
    if patient_dir=='AU_PA_0010':
        PTVs_list=['PTV']
    
    
    #change directory to patient directory
    os.chdir(patient_dir)
   
    #now that we are in the patient directory, we can extract the base rtss and rtstruct files
    
    #Adjustable: directory name of the base plan
    base_dir='TP0'
    
    #here the code lookds for fraction directories according to their name
    frac_dirs_list=[frac_dir for frac_dir  in os.listdir() if 'TP0' not in frac_dir and os.path.isdir(frac_dir)]
    
    
    
    
    #create paths for base rtss and rtdose files, these are important
    base_rtss_path=base_dir+'/RT-STRUCT.dcm'
    base_rtdose_path=base_dir+'/RT-DOSE.dcm'
    
    
    
    #compute base features and update user
    print('Computing Base Features for '+patient_dir)
    base_df=plu.RT_FeatureExtract(base_rtss_path, 
                                  base_rtdose_path,
                                  PTVs_list=PTVs_list, 
                                  OARs_list=OARs_list, 
                                  PTV_Metric_Names=PTV_Metric_Names,
                                  OAR_Metric_Names=OAR_Metric_Names,
                                  n=n).set_index(pd.Index(['Base']),)
    #base_df['GTV_Volume']=plu.get_dvh(base_rtss_path,base_rtdose_path,'GTV').V0Gy.value
    #compute base features and update user
    print('done.')    
    #base_df.to_csv(base_dir+'/baseDVHFeatures.csv')
    
    #create a dataframe that will later hve all features of all fractions
    FeaturesDF=base_df
    
    
    if save_sv:
        shift_vectors_dict={}
        
        
    #loop over all patient fractions
    for i, frac_dir in enumerate(frac_dirs_list):
        
        
        #regular adaptive fraction dataframe extraction
        frac_rtss_path=frac_dir+'/RT-STRUCT.dcm'
        frac_rtdose_path=frac_dir+'/RT-DOSE.dcm'
        
        print('Computing F'+str(i+1)+' Features for '+patient_dir)
        frac_df=plu.RT_FeatureExtract(frac_rtss_path, 
                                      frac_rtdose_path, 
                                      PTVs_list=PTVs_list, 
                                      OARs_list=OARs_list, 
                                      PTV_Metric_Names=PTV_Metric_Names,
                                      OAR_Metric_Names=OAR_Metric_Names,
                                      n=n).set_index(pd.Index(['F'+str(i+1)]))
        #frac_df['GTV_Volume']=plu.get_dvh(frac_rtss_path,frac_rtdose_path,'GTV').V0Gy.value
        print('done.')
        #add to features DF dataframe
        FeaturesDF.append(frac_df)
        
        #shift vector calculation for nonadaptive plan, round to 2 decimal places
        shift_vector=list(np.around(plu.find_GTV_overlap_vector([0,0,0], 
                                                 base_rtss_path, 
                                                 frac_rtss_path),2))
        
        if save_sv:
            #add shift vector and frac name to list
            shift_vectors_dict['NAF'+str(i+1)]=shift_vector
  
        print('Computing NAF'+str(i+1)+' Features for '+patient_dir)
        #get nonadaptive features using shift vector
        nonadap_df=plu.RT_FeatureExtract(frac_rtss_path, 
                                          base_rtdose_path, 
                                          PTVs_list=PTVs_list, 
                                          OARs_list=OARs_list, 
                                          PTV_Metric_Names=PTV_Metric_Names,
                                          OAR_Metric_Names=OAR_Metric_Names,
                                          shift_vector=shift_vector,
                                          n=n).set_index(pd.Index(['NAF'+str(i+1)]))
                                             
        FeaturesDF=FeaturesDF.append([frac_df,nonadap_df])
        print('done.')

    #sort the features DF dataframe and save as csv
    FeaturesDF=FeaturesDF.sort_index()
    FeaturesDF.to_csv(patient_dir+'_Features.csv')
    print('Features file for patient '+patient_dir+' saved as '+patient_dir+
          '_Features.csv')
    del FeaturesDF
    
    if save_sv:
    #save all shift vectors
        pd.DataFrame(shift_vectors_dict).to_csv(patient_dir+'_shift_vectors.csv')
        
        print('Shift vectors file for patient'+patient_dir+' saved as '+patient_dir+
              '_shift_vectors.csv')
      
    
    
    if save_sv:
        del shift_vectors_dict


    #return to all patients root
    os.chdir(rootpath)
    
