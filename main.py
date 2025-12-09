from fastapi import FastAPI
import requests
import numpy as np
import cv2

app = FastAPI()

def url_to_image(url):
    resp = requests.get(url)
    image = np.asarray(bytearray(resp.content), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

@app.get("/match")
def match_images(big_url: str, small_url: str):
    big_image = url_to_image(big_url)
    small_image = url_to_image(small_url)

    # Check if images loaded
    if big_image is None:
        return {"error": "Failed to load big image"}
    if small_image is None:
        return {"error": "Failed to load small image"}

    if small_image.shape[0] > big_image.shape[0] or small_image.shape[1] > big_image.shape[1]:
        return {"error": "Small image is bigger than big image"}

    # Template Matching
    result = cv2.matchTemplate(big_image, small_image, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    threshold = 0.7

    is_present = max_val >= threshold

    return {
        "present": is_present,
        "confidence": float(max_val),
        "position": max_loc if is_present else None
    }
