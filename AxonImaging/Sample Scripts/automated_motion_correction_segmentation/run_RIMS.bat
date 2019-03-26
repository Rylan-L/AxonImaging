@echo on
C:\Anaconda2\python.exe 01_BATCH_STIA_motionCorrect_JSON.py
C:\Anaconda2\python.exe 02_Batch_Merge_Corrected_Downsample_Create_MMAP_JSON.py
call activate caiman
python 03_caiman_segmentation_Python_3.py
call deactivate caiman
C:\Anaconda2\python.exe 04_get_cells_file_json.py
C:\Anaconda2\python.exe 05_refine_cells_json.py
C:\Anaconda2\python.exe 06_get_weighted_rois_and_surrounds_JSON.py
C:\Anaconda2\python.exe 07_filter_ROIs_JSON.py
C:\Anaconda2\python.exe 08_Add_full_trace_with_plotting_JSON.py

pause