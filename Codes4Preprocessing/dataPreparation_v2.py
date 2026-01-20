 """
Created on Fri Nov 26 15:54:58 2021

@author: chenj
"""

#import pylidc as pl
#from pylidc.utils import volume_viewer
import matplotlib.pyplot as plt
import numpy as np
import xlwt
import os
import pydicom
#from pydicom.dataset import Dataset, FileDataset
import SimpleITK as sitk
import pydicom_seg
from pydicom import dcmread
import cv2
from rt_utils import RTStructBuilder
import pandas as pd
import nibabel as nib

root_source_path='D:/DatasetFromTCIA/HandNeckDataset/RADCURE/'

out_store_path='D:/DatasetFromTCIA/HandNeckDataset/End2EndTranining/'
available_patient=pd.read_excel('D:/DatasetFromTCIA/HandNeckDataset/Labels/RADCURE-Label_for_HPV_FM.xlsx')

patient_ID=available_patient.iloc[:,0]#os.listdir(root_source_path)

for i in range(1):#1461,len(patient_ID)):

    image_series_path=[]
    mask_path=[]
    
    patient_path=root_source_path+patient_ID[i]

    second_patient_path=patient_path+'/'+os.listdir(patient_path)[0]
    
            
    Imageitem=os.listdir(second_patient_path)
    if len(Imageitem)>=2:
        for t in range(len(Imageitem)):
            if(len(os.listdir(second_patient_path+'/'+Imageitem[t]))==1):
                mask_path=second_patient_path+'/'+Imageitem[t]
                mask_name_list=os.listdir(mask_path)
                mask_name=mask_path+'/'+mask_name_list[0]
            if(len(os.listdir(second_patient_path+'/'+Imageitem[t]))>1):
                image_series_path=second_patient_path+'/'+Imageitem[t]
                
        reader = sitk.ImageSeriesReader()
        img_names = reader.GetGDCMSeriesFileNames(image_series_path)
        reader.SetFileNames(img_names)
        image = reader.Execute()
        image_array = sitk.GetArrayFromImage(image) # z, y, x
        image_array = image_array.astype(np.int16)

        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=image_series_path, 
            rt_struct_path=mask_name
            )
        ROIlist=rtstruct.get_roi_names()    
        mask = rtstruct.get_roi_mask_by_name(ROIlist[0])
        
        mask=mask.transpose((2,0,1))
        sum_mask_dim1=np.zeros([np.shape(mask)[0],1])
        sum_mask_dim2=np.zeros([np.shape(mask)[1],1])
        sum_mask_dim3=np.zeros([np.shape(mask)[2],1])
        for dim1 in range(np.shape(mask)[0]):
            sum_mask_dim1[dim1,0]=sum(sum(mask[dim1,:,:]))
        for dim2 in range(np.shape(mask)[1]):
            sum_mask_dim2[dim2,0]=sum(sum(mask[:,dim2,:]))
            sum_mask_dim3[dim2,0]=sum(sum(mask[:,:,dim2]))
        
        
        available_index_z=np.where(sum_mask_dim1>0)[0]
        available_index_x=np.where(sum_mask_dim2>0)[0]
        available_index_y=np.where(sum_mask_dim3>0)[0]
        if available_index_x[0]<32:
            available_x_min=0
        elif available_index_x[0]>416:
            available_x_min=384
        else:
            available_x_min=available_index_x[0]-32
            
        if available_index_y[0]<32:
            available_y_min=0
        elif available_index_y[0]>416:
            available_y_min=384
        else:
            available_y_min=available_index_y[0]-32
            
        print(available_x_min)
        print(available_y_min)
        patient_path=out_store_path+'patient_'+patient_ID[i][-4:]
        os.makedirs(patient_path, exist_ok=True)
        mask=mask.astype(np.int16)
        out_data=image_array[available_index_z[0]-10:available_index_z[0]+54,available_x_min:available_x_min+128,available_y_min:available_y_min+128]
        out_data=out_data.transpose(1,2,0)
        mask_out=mask[available_index_z[0]-10:available_index_z[0]+54,available_x_min:available_x_min+128,available_y_min:available_y_min+128]
        mask_out=mask_out.transpose(1,2,0)
        img=nib.Nifti1Image(out_data, np.eye(4))
        nib.save(img,patient_path+'/image.nii.gz')
        mask_data=nib.Nifti1Image(mask_out, np.eye(4))
        nib.save(mask_data,patient_path+'/mask.nii.gz')
        file = open(patient_path+'/label.txt', "w", encoding="utf-8")
        file.write(str(available_patient.iloc[i,5]))
        file.close()