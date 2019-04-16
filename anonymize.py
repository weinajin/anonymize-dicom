'''
Given a folder containing the raw dicom images, with gt marked as folder name,
and a csv_file associate the folder name with anonymized one,
convert the folder id to anonymize, and add its association in csv_file.
copy the data into a new folder with the anonymized folder name.
remove all the patient name info in the dicom image.
'''

import pydicom
from pydicom.filereader import read_dicomdir
import os
from os.path import dirname, join
import stat

import numpy as np
from pathlib import Path

import pandas as pd
from distutils.dir_util import copy_tree
import cv2
import shutil
import json
# tmp
import pdb


def anonymize_folder(folder_path, new_folder_path_str, csv_link_file = None, cnvt_img = True):
    '''
    Method:
        Given a folder containing the raw dicom images, with gt marked as folder name,
        and a csv_file associate the folder name with anonymized one,
        convert the folder id to anonymize, and add its association in csv_file.
        copy the data into a new folder with the anonymized folder name.
        remove all the patient name info in the dicom image.
    Input:
        - folder_path: a pathlib Path obj, or str. It is the dicom file folder to be anonymized
        - new_folder_path_str: a str, the root to save all processed anonymized pt img folder
        - csv_link_file: a csv file link the orginal folder name with the anonymized folder id, and other information about the mri scan.
        - cnvt_img: if true, create a parella folder with the same level as new_file_path, with the folder name new_folder_path_str+'_jpg'
    Output:
        - anonymized dicom folder in new_folder_path
        - dicom_dict: recording the folder dicom info, and anonymized_folder_id
    '''
    # sanity check for input type
    new_folder_path = Path(new_folder_path_str)
    if not new_folder_path.is_dir():
        new_folder_path.mkdir()
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    assert folder_path.is_dir(), "[Error] {} folder does not exist!".format(folder)

    # record dicom and remove patient name
    dicom_dict, anonymized_folder = record_dicoms(folder_path, new_folder_path_str, cnvt_img)
    if cnvt_img:
        new_folder_path_jpg = new_folder_path_str+'_jpg'
        anonymized_folder_jpg = Path(new_folder_path_jpg)/anonymized_folder.name
        assert anonymized_folder_jpg.is_dir(), '[Error!] There is no cnvt_img folder {}, but cnvt_img is {}.'.format(anonymized_folder_jpg, cnvt_img)

    # get the unique patient name for the folder to be anonymized
    pt_name = None
    # remove anonmyous in the 'PatientName' field, it only happend when run multiple times of a folder
    if 'anonymous' in dicom_dict['PatientName']:
        name_list = dicom_dict['PatientName']
        name_list.remove('anonymous')
        dicom_dict['PatientName'] = name_list
    # get unique patient name from the gt folder name or the dicom metadata
    if len(dicom_dict['PatientName']) > 1:
        print('!!! [Error] More than one name exist in {}  {}, please input patient name:'.format(dicom_dict['PatientName'], dicom_dict['folder_gt'])  )
        pt_name = input()
        print('Received patient name input: {}'.format(pt_name))
    else:
        pt_name = dicom_dict['PatientName'][0]

    # get the pt_idx and mri_idx
    if (csv_link_file is None) or (not os.path.isfile(csv_link_file)):
        pt_idx = 1
        mri_idx = 1
        df = pd.DataFrame()
    else:
        df = pd.read_csv(csv_link_file)
        if pt_name in set(df['PatientName']):    # check if same pt exist, generate pt_code
            # sanity check: check if the same record already exists
            assert dicom_dict['folder_gt'] not in set(df['folder_gt']), '!!! [ERROR] The same folder {} already been processed! '.format(dicom_dict['folder_gt'])
            # get pt_idx, idx is to generate the code. pt_code: P0001, P0002; mri_codes: P0001_01
            pt_code = df['anonymized_pt'].loc[df['PatientName'] == pt_name]
            assert len(set(pt_code))  == 1, '[ERROR] multiple anonymized_pt {} exist with the same PatientName {}'.format(pt_code, pt_name )
            pt_idx = int(list(pt_code)[0].strip('P'))
            # assign new mri_idx
            mri_codes = df['anonymized_scanfolder'].loc[df['PatientName'] == pt_name]
            mri_idx = max([int(i.strip('P').split('_')[1]) for i in mri_codes]) + 1
        else:
            pt_idx = max([int(i.strip('P')) for i in df['anonymized_pt']]) + 1
            mri_idx = 1
    pt_code = 'P'+"%04d" % pt_idx
    pt_mri_code = pt_code +'_'+ "%02d" % mri_idx

    # update pt_code and pt_mri_code to dicom_dict, it will write into csv_link_file
    dicom_dict['PatientName']= pt_name
    dicom_dict['anonymized_pt'] = pt_code
    dicom_dict['anonymized_scanfolder'] = pt_mri_code
    print('=== Anonymized patient code is: {} '.format(pt_code))
    print('=== Anonymized scan code is: {} '.format(pt_mri_code))

    # rename hash_pt_folder name with pt_code
    anonymized_folder.rename(Path(new_folder_path_str)/pt_mri_code)
    if cnvt_img:
        anonymized_folder_jpg.rename(Path(new_folder_path_jpg)/pt_mri_code)

    # get ride of the list in dicom_dict and save multiple value as string seperated with ;
    # add dicom_dict info to csv_link_file
    df = df.append(dicom_dict, ignore_index = True)

    # save the csv
    if csv_link_file == None:
        csv_link_file = 'link_anonymize.csv'
    df.to_csv('../output/'+csv_link_file, index = False, encoding="utf_8_sig")

    return dicom_dict, df

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def person_names_callback(dataset, data_element):
    '''
    ref: https://pydicom.github.io/pydicom/stable/auto_examples/metadata_processing/plot_anonymize.html
    '''
    if data_element.VR == "PN":
        data_element.value = "anonymous"

def record_dicoms(folder_path, new_folder_path_str, cnvt_img = True):
    '''
    Method:
        1. record patient information in the folder_path.
        2. copy the folder_path to anonymize folder, with pt name metadata deleted
    Input:
        - folder_path, a str or a Path obj
        - new_folder_path_str: a str directory for saving new deidentified pt img folder, no "/" at the end. must be exist
        - cnvt_img: if true, create a parella folder with the same level as new_file_path, with the folder name new_folder_path_str+'_jpg'
    Output:
        - a anonymized dicomdir folder under new_folder_path_str
    '''
    # copy dicomdir to anonymized_folders
    df = pd.DataFrame()
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    anonymized_folder_name = str(hash(folder_path.name)) # a placeholder name to be renamed by deidentified patient code
    anonymized_folder = Path(new_folder_path_str)/anonymized_folder_name
    if not anonymized_folder.is_dir():
        anonymized_folder.mkdir() # create a new folder in new_file_path
    copytree(str(folder_path), str(anonymized_folder))

    # convert dicom to images
    if cnvt_img:
        # make a new root folder if not exist
        new_folder_path_str_jpg = new_folder_path_str+'_jpg'
        new_folder_path_jpg = Path(new_folder_path_str_jpg)
        if not new_folder_path_jpg.is_dir():
            new_folder_path_jpg.mkdir()
        # create pt folder inside the jpg folder
        anonymized_folder_jpg = Path(new_folder_path_jpg)/anonymized_folder_name
        if not anonymized_folder_jpg.is_dir():
            anonymized_folder_jpg.mkdir()

    # change the target folder to can be written
    for root, dirs, files in os.walk(str(anonymized_folder)):
        for fname in files:
            full_path = os.path.join(root, fname)
            os.chmod(full_path, stat.S_IWRITE)

    # read dicom dir
    # ref: https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_read_dicom_directory.html
    dicom_dir = read_dicomdir(str(anonymized_folder)+'/DICOMDIR')
    dicom_dict = {'folder_gt': folder_path.name,  'hash_folder_id' :anonymized_folder_name, 'PatientName': set(), 'PatientSex': set(), 'PatientID': set(), 'PatientBirthDate': set()}

    # go through the patient record and print information
    for patient_record in dicom_dir.patient_records:
        if hasattr(patient_record, 'PatientName'):
            dicom_dict['PatientName'].add(str(patient_record.PatientName))
            print("[PatientName] {}".format(dicom_dict['PatientName'] ))
        studies = patient_record.children
        series_dict = dict()
        # got through each serie
        for study in studies:
            print(" " * 4 + "Study {}: {}: {}".format(study.StudyID,
                                                      study.StudyDate,
                                                      study.StudyDescription))
            if 'StudyDescription' not in dicom_dict:
                dicom_dict['StudyDescription'] = study.StudyDescription
            else:
                print("Study {} already exist, please give a input for {}:".format(dicom_dict['StudyDescription'], study.StudyDescription ))
            dicom_dict['StudyDate'] = study.StudyDate
            all_series = study.children
            # go through each serie
            for series in all_series:
                # Write basic series info and image count
                image_count = len(series.children)
                plural = ('', 's')[image_count > 1]

                # Put N/A in if no Series Description
                if 'SeriesDescription' not in series:
                    series.SeriesDescription = "N/A"
                series_dict[series.SeriesDescription] = {'image_count': image_count, 'plane': set()}
                print(" " * 8 + "Series {}: {}: {} ({} image{})".format(
                    series.SeriesNumber, series.Modality, series.SeriesDescription,
                    image_count, plural))

                # Open and read from each image, save the image as jpg with
                image_folder_name = str(series.SeriesDescription).replace(' ', '_').replace('*', 'Star').replace(':','_').replace('/', '-').replace('?', '-')
                print(" " * 12 + "Reading images...")
                image_records = series.children
                image_filenames = [join(str(anonymized_folder), *image_rec.ReferencedFileID)
                                   for image_rec in image_records]

                # print(image_filenames)
                # datasets = [pydicom.dcmread(image_filename)
                #             for image_filename in image_filenames]
                datasets = []
                for image_filename in image_filenames:
                    try:
                        datasets.append(pydicom.dcmread(image_filename))
                    except:
                        image_filename += '(1)'
                        datasets.append(pydicom.dcmread(image_filename))
                if cnvt_img:
                    # save the dicom as image files in the image_folder_name
                    series_folder = Path(anonymized_folder_jpg)/image_folder_name
                    if not series_folder.is_dir():
                        series_folder.mkdir()
                for i in range(len(datasets)):
                    ds = datasets[i]
                    dicom_dict['PatientName'].add(str(ds.PatientName))
                    dicom_dict['PatientID'].add(str(ds.PatientID))
                    dicom_dict['PatientBirthDate'].add(str(ds.PatientBirthDate))
                    dicom_dict['PatientSex'].add(str(ds.PatientSex))
                    try:
                        series_dict[series.SeriesDescription]['plane'].add(file_plane(ds.ImageOrientationPatient))
                    except AttributeError:
                        series_dict[series.SeriesDescription]['plane'].add(str(ds.ImageType))

                    # anonymize and save with replace the dicom
                    # print(image_filenames[i])
                    ds.walk(person_names_callback)
                    ds.save_as(image_filenames[i])

                    if cnvt_img:
                        image_2d = ds.pixel_array.astype(float)
                        # Rescaling grey scale between 0-255
                        image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0  # todo: divide by 0 warning!
                        # Convert to uint
                        image_2d_scaled = np.uint8(image_2d_scaled)
                        cv2.imwrite( str(series_folder) + '\\' + str(image_records[i].ReferencedFileID)+'_'+str(series_dict[series.SeriesDescription]['plane'])+'.jpg', image_2d_scaled)

                # break #tmp
        dicom_dict['series'] = series_dict

    # todo: since pydicom do not support write dicomdir, there is no way to write anonymized dicomdir file
    # dicom_dir.walk(person_names_callback)
    # dicom_dir.save_as(str(anonymized_folder)+'/DICOMDIR')
    # print('dicom saved at ', anonymized_folder)

    # change set into list in the directory
    for k, v in dicom_dict.items():
        if isinstance(v, set) :
            if k == 'PatientName':
                dicom_dict[k] = list(v)
            else:
                dicom_dict[k] = ';'.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, set):
                    dicom_dict[k][kk] = ';'.join(vv)
                elif isinstance(vv, dict):
                    for kkk, vvv in vv.items():
                        if isinstance(vvv, set):
                            dicom_dict[k][kk][kkk] = ';'.join(vvv)

    # save the dicom_dict as json file
    # dicom_json = 'tmp'
    # with open(dicom_json + '.json', 'w') as fp:
    #     json.dump(dicom_dict, fp)
    return dicom_dict, anonymized_folder



def file_plane(IOP):
    """Get image plane from dicom.ImageOrientationPatient
    ref: https://stackoverflow.com/questions/34782409/understanding-dicom-image-attributes-to-get-axial-coronal-sagittal-cuts
    """
    IOP_round = [round(x) for x in IOP]
    plane = np.cross(IOP_round[0:3], IOP_round[3:6])
    plane = [abs(x) for x in plane]
    if plane[0] == 1:
        return "Sagittal"
    elif plane[1] == 1:
        return "Coronal"
    elif plane[2] == 1:
        return "Axial"


# def record_dicom(folder_path, new_folder_dir, remove_name = True, cnvt_img = True):
#     """
#     copy the dicom to a new folder, with random folder name using hash.
#     Input:
#         - folder_path: the folder containing the dicom scans for one patient
#         - new_folder_dir: a str directory for saving new deidentified pt img folder, no "/" at the end. must be exist
#         - remove_name: whether to remove name in the new dicom file
#         - cnvt_img: whether convert the dicom to jpg images, if so, create a new folder called new_folder_dir_jpg and save under it.
#     Output:
#         - a dicom df file, the new_folder_name
#     """
#     pt_name = None
#     df = pd.DataFrame()
#     anonymized_folder_name = str(hash(folder_path.name)) # a placeholder name to be renamed by deidentified patient code
#     anonymized_folder = Path(new_folder_dir)/anonymized_folder_name
#     if not anonymized_folder.is_dir():
#         anonymized_folder.mkdir() # create a new folder in new_file_path
#     copy_tree(str(folder_path), str(anonymized_folder))
#     if cnvt_img:
#         # make a new root folder if not exist
#         new_folder_dir_jpg = new_folder_dir+'_jpg'
#         new_folder_dir_jpg = Path(new_folder_dir_jpg)
#         if not new_folder_dir_jpg.is_dir():
#             new_folder_dir_jpg.mkdir()
#         # create pt folder inside the jpg folder
#         anonymized_folder_jpg = Path(new_folder_dir_jpg)/anonymized_folder_name
#         if not anonymized_folder_jpg.is_dir():
#             anonymized_folder_jpg.mkdir()
#     for entry in anonymized_folder.glob('**/*'):
#         dicom_dict = {}
#         if entry.is_file() and entry.name[0] == 'I':
#             dicom = pydicom.dcmread(str(entry), force=True)
#             dicom_dict['folder_id'] = folder_path.name
#             try:
#                 plane = file_plane(dicom.ImageOrientationPatient)
#                 dicom_dict['plane'] = plane
#             except AttributeError:
#                 dicom_dict['plane'] = dicom.ImageType
#             # get the sequence name
#             dicom_dict['series'] = dicom.SeriesDescription
#             # get patient metadata
#             dicom_dict['pt_id'] = dicom.PatientID
#             # since pt_name is important of deidentification and split, make sure the field is not empty
#             if len(str(dicom.PatientName)) == 0 and pt_name == None:
#                 print('[Input request] No patient name in dicom, please input for {}'.format(folder_path.name))
#                 pt_name = input()
#             elif len(str(dicom.PatientName)) == 0 and pt_name:
#                 dicom_dict['pt_name'] = pt_name
#             else:
#                 dicom_dict['pt_name'] = str(dicom.PatientName)
#             dicom_dict['pt_birthday'] = dicom.PatientBirthDate
#             dicom_dict['pt_sex'] = dicom.PatientSex
#             dicom_dict['mri_date'] = dicom.AcquisitionDate
#             dicom_dict['manufacturer'] = dicom.Manufacturer
#             dicom_dict['manufacturer_model'] = dicom.ManufacturerModelName
#             try:
#                 dicom_dict['pt_pos'] = dicom.PatientPosition
#             except:
#                 dicom_dict['pt_pos'] = None
#             dicom_dict['file_name'] = entry.parent.name+'_'+entry.name
#             dicom_dict['hash_folder_id'] = anonymized_folder_name # to be changed later when assigned a pt_code
#             df = df.append(dicom_dict, ignore_index=True)
#             if remove_name:
#                 # ref: https://pydicom.github.io/pydicom/stable/auto_examples/metadata_processing/plot_anonymize.html
#                 dicom.PatientName = 'anonymize'
#             # save the dicom file in the new folder path with its root folder name
# #             print('[DONE FILE SAVE] {} saved at {}.'.format(dicom_dict['file_name'], anonymized_folder))
# #             dicom.save_as(str(anonymized_folder.resolve()) + '\\' + str(dicom_dict['file_name']+'.dcm'))
#             dicom.save_as(str(entry)+'.dcm')
#             if cnvt_img:
#                 # save the dicom file as jpg in new_folder_dir_jpg, with its root folder name
#                 # ref: https://github.com/pydicom/pydicom/issues/352
#                 # Convert to float to avoid overflow or underflow losses.
#                 image_2d = dicom.pixel_array.astype(float)
#
#                 # Rescaling grey scale between 0-255
#                 image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0  # todo: divide by 0 warning!
#
#                 # Convert to uint
#                 image_2d_scaled = np.uint8(image_2d_scaled)
#
#                 cv2.imwrite( str(anonymized_folder_jpg.resolve()) + '\\' + str(dicom_dict['file_name']+'.jpg'), image_2d_scaled)
# #                 print('[DONE JPG SAVE] {} saved at {}.'.format(dicom_dict['file_name'], anonymized_folder_jpg))
#             break
#     print('[DONE] {} saved at {}.'.format(folder_path, str(anonymized_folder.resolve())))
#     return df, anonymized_folder, anonymized_folder_jpg
