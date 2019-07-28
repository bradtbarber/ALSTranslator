from Camera_Capture import camera_capture
from ASL_Detector_Model import asl_detector_model
from Word_Search import word_search

image_path = camera_capture.begin_image_capture()
word = asl_detector_model.translate_asl_images(image_path)
check_word_result = word_search.check_word(word)

if check_word_result == -1:
    print('ERROR')