import argparse
import os
import sys
import pickle
import json
from time import sleep
from xml.etree.ElementInclude import default_loader
from flask import Flask, request, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import (
    StringField,
    TextAreaField,
    SubmitField,
    PasswordField,
    DateField,
    SelectField,
    BooleanField,
    IntegerField,
    RadioField)
from wtforms.validators import DataRequired, Length, URL
from aiot_device import get_edge_devices, get_edge_modules, get_module_twins, patch_module_twins, restart_module
from aiot_cosmos import cosmos_create_items, cosmos_query_dm, cosmos_query_mt, cosmos_delete_db
from aiot_storage import get_model_names

pkl_path = "/config/"
col_hub_updates = "edgemoduletwins"
col_twin_updates = "edgetwinupdates"
app = Flask(__name__)
app.config['SECRET_KEY'] = '5199F5E8F6D16AF4FF452D3D8B74D'

Bootstrap(app)
print(os.environ['STORE_CONN_STR'])
print(os.environ['BLOB_AZURE_MODEL_CONTAINER'])
store_conn_str = os.environ['STORE_CONN_STR']
blob_azure_model_container = os.environ['BLOB_AZURE_MODEL_CONTAINER']
models = get_model_names(store_conn_str, blob_azure_model_container)

camera_type_list = [
    'Allied Vision GVSP',
    'Basler GVSP',
    'RTSP Camera',
    'Read from file',
]

model_type_list = [
    'Azure Custom Vision - Object Detection', 
    'Azure Custom Vision - Multi-Class Classification', 
    'Azure Custom Vision - Multi-Label Classification', 
    'Azure Computer Vision - OCR Read', 
    'AutoML for Images - YOLOv5', 
    'AutoML for Images - Faster-RCNN', 
    'AutoML for Images - Retinanet', 
    'AutoML for Images - Multi-Label Classification', 
    'AutoML for Images - Multi-Class Classification', 
    'AutoML for Images - Mask-RCNN', 
]

class SvcConfig(FlaskForm):
    hub_name = StringField('IoT Hub Name', validators=[DataRequired()])
    hub_own_str = StringField('IoT Hub Owner Shared Access Connection String', validators=[DataRequired()])
    # hub_svc_str = StringField('IoT Hub Service Shared Access Connection String', validators=[DataRequired()])
    cosmos_db = StringField('CosmosDB Name (lowercase, no spaces)', validators=[DataRequired()])
    cosmos_uri = StringField('CosmosDB URI', validators=[DataRequired()])
    cosmos_key = StringField('CosmosDB Key', validators=[DataRequired()])
    store_conn_str = StringField('Blob Connection String', validators=[DataRequired()])
    container_name = StringField('Blob Container Name', validators=[DataRequired()])
    submit = SubmitField('Save Config')

class TwinConfig(FlaskForm):
    device_id = StringField('Device Name*', validators=[DataRequired()])
    module_id = StringField('Module Name*', validators=[DataRequired()])
    camera_type = SelectField('Camera Type', choices=camera_type_list, default='Read from file')
    camera_trigger = BooleanField('Camera Trigger Installed', default=False)
    camera_id = StringField('Camera Name/ID*', validators=[DataRequired()])
    camera_uri = StringField('Camera URI (if using IP address)')
    camera_location = StringField('Camera Location*', validators=[DataRequired()])
    camera_position = StringField('Camera Position*', validators=[DataRequired()])
    camera_fps = StringField('Camera FPS (if no trigger)')
    inference_fps = StringField('Inference FPS desired(if no trigger)')
    model_type = SelectField('Model Type', choices=model_type_list, default='AutoML for Images - YOLOv5')
    model_name = SelectField('Model Name*', choices=models, default='<none>')
    model_version = StringField('Model Version*', validators=[DataRequired()], default='1')
    model_acv_ocr_uri = StringField('Azure Computer Vision OCR URI (if applicable)', default="http://localhost:5000/vision/v3.2/read/analyze")
    model_acv_ocr_secondary = BooleanField('Model: Azure Computer Vision OCR Secondary -> Cropped Object Detection Bbox')
    target_dim = StringField('Model Input Dimension* (i.e. 640)*', validators=[DataRequired()])
    prob_thres = StringField('Probability Threshold* (i.e. .65)*', validators=[DataRequired()])
    iou_thres = StringField('IOU Threshold* (i.e. .45, only applies to obj det or mask)*', validators=[DataRequired()])
    retrain_interval = StringField('Retraining Interval (i.e. 500)*', validators=[DataRequired()])
    store_raw_frames = BooleanField('Store Raw Frames in Azure')
    store_all_inferences = BooleanField('Store All Inferences')
    submit = SubmitField('Save Twin')

def read_config(config):
    config_read = open(f"{pkl_path}{config}", "rb")
    config_dict = pickle.load(config_read)
    return config_dict

def get_device_module(hub_own_str, cosmos_db, cosmos_uri, cosmos_key):
        try:
            m_list = get_edge_modules(hub_own_str)
            # print(d_list)
            print(m_list)
            for kvp in m_list:
                deviceId = kvp['deviceId']
                moduleId = kvp['moduleId']
                item = get_module_twins(hub_own_str, deviceId, moduleId)        
                create_item = cosmos_create_items(cosmos_db, cosmos_uri, cosmos_key, col_hub_updates, item)        
                print(f'Cosmos item {create_item} for "{deviceId}-{moduleId}"')

        except Exception as e:
            print(f"{e}\n Error in configuration information - please check and re-enter.")

@app.route('/', methods=['GET', 'POST'])
def index():
    cfg_list = os.listdir(pkl_path)
    config_list = []
    if len(cfg_list) < 1:
        return redirect( url_for('svc_config') )

    else:
        for config in cfg_list:
            config_split = os.path.splitext(config)[0]
            config_entry = {
                "file_name": config,
                "base_name": config_split
            }
            config_list.append(config_entry)

    return render_template('index.html', config_list=config_list)
        
@app.route('/main_list/', methods=['GET', 'POST'])
def main_list(): 
    config_name = request.args.get('config')    
    config = read_config(config_name)
    print(f"Stored config: {config}")
    hub_own_str = config["hub_own_str"]
    cosmos_db = config["cosmos_db"]
    cosmos_uri = config["cosmos_uri"]
    cosmos_key = config["cosmos_key"]

    hub_poll = get_device_module(hub_own_str, cosmos_db, cosmos_uri, cosmos_key)

    d,m,dm = cosmos_query_dm(cosmos_db, cosmos_uri, cosmos_key, col_hub_updates)        
    print(f'Cosmos retrieved: \n {d} \n {dm}')
    d_count = len(d)
    m_count = len(m)
    dm_count = len(dm)
        
    print(f"Device count: {d_count} Module count: {dm_count}")

    return render_template('main_list.html', dm=dm, dm_count=dm_count, d = d, d_count=d_count, config_file=config_name)  

@app.route('/svc_config/', methods=['GET', 'POST'])
def svc_config():

    action = request.args.get('action')  
    print(f"Action = {action}")
    config_name = request.args.get('config')

    if action == "edit":
        config = read_config(config_name)
        form = SvcConfig(
            hub_name = config["hub_name"],
            hub_own_str = config["hub_own_str"],
            cosmos_db = config["cosmos_db"],
            cosmos_uri = config["cosmos_uri"],
            cosmos_key = config["cosmos_key"],
            store_conn_str = config["store_conn_str"],
            container_name = config["container_name"],
        )
        message = "Edit Service Configuration"

    elif action == "delete":
        config = read_config(config_name)
        hub_name = config["hub_name"]
        hub_own_str = config["hub_own_str"]
        cosmos_db = config["cosmos_db"]
        cosmos_uri = config["cosmos_uri"]
        cosmos_key = config["cosmos_key"]
        store_conn_str = config["store_conn_str"]
        container_name = config["container_name"]

        config_path = os.path.join(pkl_path, config_name)
        os.remove(config_path)
        cosmos_delete_db(cosmos_db, cosmos_uri, cosmos_key)

        return redirect( url_for('index') )    
    else:
        form = SvcConfig()

    message = "New Service Configuration"

    if form.validate_on_submit():
        hub_name = form.hub_name.data
        hub_own_str = form.hub_own_str.data
        cosmos_db = form.cosmos_db.data
        cosmos_uri = form.cosmos_uri.data
        cosmos_key = form.cosmos_key.data
        store_conn_str = form.store_conn_str.data
        container_name = form.container_name.data

        svc_cfg_dict = {
            "hub_name": hub_name,
            "hub_own_str": hub_own_str,
            "cosmos_db": cosmos_db,
            "cosmos_uri": cosmos_uri,
            "cosmos_key": cosmos_key,
            "store_conn_str": store_conn_str,
            "container_name": container_name,
        }
        config_write = open(f"{pkl_path}{hub_name}.pkl", "wb")
        pickle.dump(svc_cfg_dict, config_write)
        config_write.close()
        message = "Configuration saved - database and collections created. Click on Home."

        # hub_poll = get_device_module(hub_own_str, cosmos_db, cosmos_uri, cosmos_key)

    return render_template('svc_config.html', form=form, message=message)

@app.route('/twin_config/', methods=['GET', 'POST'])
def twin_config(): 
    deviceId = request.args.get('deviceId')
    moduleId = request.args.get('moduleId')
    config_name = request.args.get('config')
    config = read_config(config_name)
    hub_own_str = config["hub_own_str"]
    cosmos_db = config["cosmos_db"]
    cosmos_uri = config["cosmos_uri"]
    cosmos_key = config["cosmos_key"]

    try:
        mt_data = cosmos_query_mt(cosmos_db, cosmos_uri, cosmos_key, col_twin_updates, deviceId, moduleId)
        mt_hub_data = cosmos_query_mt(cosmos_db, cosmos_uri, cosmos_key, col_hub_updates, deviceId, moduleId)
        if len(mt_data) > 0:
            form = TwinConfig(
            device_id = deviceId,
            module_id = moduleId,
            camera_type = mt_data[0]['CAMERA_TYPE'],
            camera_trigger = mt_data[0]['CAMERA_TRIGGER'],
            camera_id = mt_data[0]['CAMERA_ID'],
            camera_uri = mt_data[0]['CAMERA_URI'],
            camera_location = mt_data[0]['CAMERA_LOCATION'],
            camera_position = mt_data[0]['CAMERA_POSITION'],
            camera_fps = mt_data[0]['CAMERA_FPS'],
            inference_fps = mt_data[0]['INFERENCE_FPS'],
            model_type = mt_data[0]['MODEL_TYPE'],
            model_name = mt_data[0]['MODEL_NAME'],
            model_version = mt_data[0]['MODEL_VERSION'],
            model_acv_ocr_uri = mt_data[0]['MODEL_ACV_OCR_URI'],
            model_acv_ocr_secondary = mt_data[0]['MODEL_ACV_OCR_SECONDARY'],
            target_dim = mt_data[0]['TARGET_DIM'],
            prob_thres = mt_data[0]['PROB_THRES'],
            iou_thres = mt_data[0]['IOU_THRES'],
            retrain_interval = mt_data[0]['RETRAIN_INTERVAL'],
            store_raw_frames = mt_data[0]['STORE_RAW_FRAMES'],
            store_all_inferences = mt_data[0]['STORE_ALL_INFERENCES'],
            )
        elif len(mt_hub_data) > 0:
            form = TwinConfig(
            device_id = deviceId,
            module_id = moduleId,
            camera_type = mt_hub_data[0]['CAMERA_TYPE'],
            camera_trigger = mt_hub_data[0]['CAMERA_TRIGGER'],
            camera_id = mt_hub_data[0]['CAMERA_ID'],
            camera_uri = mt_hub_data[0]['CAMERA_URI'],
            camera_location = mt_hub_data[0]['CAMERA_LOCATION'],
            camera_position = mt_hub_data[0]['CAMERA_POSITION'],
            camera_fps = mt_hub_data[0]['CAMERA_FPS'],
            inference_fps = mt_hub_data[0]['INFERENCE_FPS'],
            model_type = mt_hub_data[0]['MODEL_TYPE'],
            model_name = mt_hub_data[0]['MODEL_NAME'],
            model_version = mt_hub_data[0]['MODEL_VERSION'],
            model_acv_ocr_uri = mt_hub_data[0]['MODEL_ACV_OCR_URI'],
            model_acv_ocr_secondary = mt_hub_data[0]['MODEL_ACV_OCR_SECONDARY'],
            target_dim = mt_hub_data[0]['TARGET_DIM'],
            prob_thres = mt_hub_data[0]['PROB_THRES'],
            iou_thres = mt_hub_data[0]['IOU_THRES'],
            retrain_interval = mt_hub_data[0]['RETRAIN_INTERVAL'],
            store_raw_frames = mt_hub_data[0]['STORE_RAW_FRAMES'],
            store_all_inferences = mt_hub_data[0]['STORE_ALL_INFERENCES'],
            )
    except:
        form = TwinConfig(
            device_id = deviceId,
            module_id = moduleId
        )

    message = ""

    if form.validate_on_submit():
        device_id = form.device_id.data
        module_id = form.module_id.data
        camera_type = form.camera_type.data
        camera_trigger = form.camera_trigger.data
        camera_id = form.camera_id.data
        camera_uri = form.camera_uri.data
        camera_location = form.camera_location.data
        camera_position = form.camera_position.data
        camera_fps = form.camera_fps.data
        inference_fps = form.inference_fps.data
        model_type = form.model_type.data
        model_name = form.model_name.data
        model_version = form.model_version.data
        model_acv_ocr_uri = form.model_acv_ocr_uri.data
        model_acv_ocr_secondary = form.model_acv_ocr_secondary.data
        target_dim = form.target_dim.data
        prob_thres = form.prob_thres.data
        iou_thres = form.iou_thres.data
        retrain_interval = form.retrain_interval.data
        store_raw_frames = form.store_raw_frames.data
        store_all_inferences = form.store_all_inferences.data


        twin_cfg_dict = {
            "id": f"{device_id}-{module_id}",
            "deviceId": device_id,
            "moduleId": module_id,
            "CAMERA_TYPE": camera_type,
            "CAMERA_TRIGGER": camera_trigger,
            "CAMERA_ID": camera_id,
            "CAMERA_URI": camera_uri,
            "CAMERA_LOCATION": camera_location,
            "CAMERA_POSITION": camera_position,
            "CAMERA_FPS": camera_fps,
            "INFERENCE_FPS": inference_fps,
            "MODEL_TYPE": model_type,
            "MODEL_NAME": model_name,
            "MODEL_VERSION": model_version,
            "MODEL_ACV_OCR_URI": model_acv_ocr_uri,
            "MODEL_ACV_OCR_SECONDARY": model_acv_ocr_secondary,
            "TARGET_DIM": target_dim,
            "PROB_THRES": prob_thres,
            "IOU_THRES": iou_thres,
            "RETRAIN_INTERVAL": retrain_interval,
            "STORE_RAW_FRAMES": store_raw_frames,
            "STORE_ALL_INFERENCES": store_all_inferences,
        }
        twin_patch_dict = {
            "CAMERA_TYPE": camera_type,
            "CAMERA_TRIGGER": camera_trigger,
            "CAMERA_ID": camera_id,
            "CAMERA_URI": camera_uri,
            "CAMERA_LOCATION": camera_location,
            "CAMERA_POSITION": camera_position,
            "CAMERA_FPS": camera_fps,
            "INFERENCE_FPS": inference_fps,
            "MODEL_TYPE": model_type,
            "MODEL_NAME": model_name,
            "MODEL_VERSION": model_version,
            "MODEL_ACV_OCR_URI": model_acv_ocr_uri,
            "MODEL_ACV_OCR_SECONDARY": model_acv_ocr_secondary,           
            "TARGET_DIM": target_dim,
            "PROB_THRES": prob_thres,
            "IOU_THRES": iou_thres,
            "RETRAIN_INTERVAL": retrain_interval,
            "STORE_RAW_FRAMES": store_raw_frames,
            "STORE_ALL_INFERENCES": store_all_inferences,
        }
        
        create_item = cosmos_create_items(cosmos_db, cosmos_uri, cosmos_key, col_twin_updates, twin_cfg_dict)
        print(f'Cosmos item {create_item} for "{deviceId}-{moduleId}"')
        create_patch = patch_module_twins(hub_own_str, deviceId, moduleId, twin_patch_dict)
                    
        if create_patch:
            device_twin_method = restart_module(hub_own_str, deviceId, moduleId)
            if device_twin_method:
                message = "Twin desired properties stored, module twin updated and module restarted."
        else:
            message = "Twin desired properties store/update failed."
        
    return render_template('twin_config.html', form=form, message=message)   

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# keep this as is
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blob-string',
                        help='Add blob connection string')
    parser.add_argument('-container-name',
                        help='Add container name')

    # run locally
    # app.run(host='127.0.0.1', port=22000) or
    # app.run(debug=True)

    # run in container
    app.run(host='0.0.0.0', port=22000)
    