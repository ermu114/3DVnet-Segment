import utility.data_converter as converter

def test_nii_2_dicomrt():
    src_cases_root_path = 'D:\\MLDO\\datasets\\nii\\HaN_OAR_50_1'
    src_cases_root_path = 'D:\\MLDO\\datasets\\nii\\HaN_OAR_50_2'
    dst_cases_root_path = 'D:\\MLDO\\datasets\\dicomrt\\HaN_OAR'
    converter.nii_2_dicomrt(src_cases_root_path, dst_cases_root_path)
    pass

def dicomrt_2_nii_canlan():
    src_cases_root_path = 'D:\\NIIDCMRT\\rt'
    dst_cases_root_path = 'D:\\NIIDCMRT\\nii'
    converter.dicomrt_2_nii(src_cases_root_path, dst_cases_root_path)
    pass

def nii_2_dicomrt_canlan():
    src_cases_root_path = 'D:\\NIIDCMRT\\nii'
    dst_cases_root_path = 'D:\\NIIDCMRT\\rt'
    converter.nii_2_dicomrt(src_cases_root_path, dst_cases_root_path)
    pass

#test_nii_2_dicomrt()
#dicomrt_2_nii_canlan()
nii_2_dicomrt_canlan()