import glob
import os
import shutil
import time
import random

loop_count = 50

image_dir = '../edge_assets/model_sync/test_images/ppe_inference/'
target_list = ['../edge_assets//image_sink_volume/Mfg_Vision_CIS_Modular_Camera_1/', 
    '../edge_assets//image_sink_volume/Mfg_Vision_CIS_Modular_Camera_2/', 
    '../edge_assets//image_sink_volume/Mfg_Vision_CIS_Modular_Camera_3/', 
    '../edge_assets//image_sink_volume/Mfg_Vision_CIS_Modular_Camera_4/',
    '../edge_assets//image_sink_volume/Mfg_Vision_CIS_Monolithic_Camera_1/'
    ]

for l in range(loop_count):
    image_files = glob.glob("../edge_assets/model_sync/test_images/ppe_inference/*.jpg")
    image_count = len(image_files)
    x = 0
    for i in range(image_count):
        i = random.randint(0, image_count -1)
        image_name = image_files[i]
        shutil.copy(image_name, target_list[x])
        x += 1
        time.sleep(1)
        if x == 5:
            x = 0
    
