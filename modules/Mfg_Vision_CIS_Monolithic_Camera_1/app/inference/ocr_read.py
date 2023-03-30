import json
import os
import threading
import time
import uuid
import requests
from datetime import datetime
from requests.models import Response
from typing import Any, Callable, Optional, Union
import numpy as np

ocr_endpoint = "http://localhost:5000/vision/v3.2/read/analyze"

def _send_frame_for_OCR_processing(frame: bytes) -> Union[Response, str]:
        """Sends a frame for processing to the OCR image processing endpoint.

        Args:
            frame (bytes): Frame encoded as bytes.

        Returns:
            Union[Response, str]: Response from the endpoint.
        """
        headers = {"Content-Type": "application/octet-stream"}
        params = {"language": "eng", "detectOrientation": "true", 'readSlim': 'true'}
        try:
            response = requests.post(ocr_endpoint, headers=headers, params=params, data=frame)
        except Exception as e:
            print("_send_frame_for_OCR_processing Exception -" + str(e))
            return "[]"

        return response

def _get_response(url: str) -> Optional[Any]:
        """Sends a GET request to the URL. Returns None if the endpoint
        reports the job is still running. Else, returns the response as JSON.

        Args:
            url (str): Endpoint to get response from.

        Returns:
            Optional[Any]: JSON response from the endpoint. None if the job is still running.
        """
        response = requests.get(url)
        if response.json()["status"] == "running":
            return None
        else:
            # print(response.json())
            return response.json()

def _process_frame_for_ocr(frame: np.ndarray) -> Optional[Any]:
    try:
        start_ocr = time.time()
        ocr_response = _send_frame_for_OCR_processing(frame)
        ocr_response_url = ocr_response.headers["Operation-Location"]
        # print("_process_frame_for_ocr - ocr_response_url: " + ocr_response_url)
        ocr_results= None

        while ocr_results is None:
            ocr_results = _get_response(ocr_response_url)
            if ocr_results is None:
                time.sleep(.25)
        end_ocr = time.time()
    except Exception as e:
        print(e)
        return

    ocr_inference_time = (end_ocr - start_ocr)*1000
    print("OCR Inference Time: " + str(ocr_inference_time))
    print(f'[{datetime.now()}] Results: {ocr_results["analyzeResult"]["readResults"]}')

    return ocr_results["analyzeResult"]["readResults"][0]["lines"][0]["text"]