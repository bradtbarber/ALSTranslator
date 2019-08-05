from Camera_Capture import camera_capture
from ASL_Detector_Model import asl_detector_model
from Word_Search import word_search


# function to return the second element of the 
# two elements passed as the paramater 
def sortFirst(val): 
    return val[0]

def get_top_predictions(predictions, limit):
    reduced_predictions = []
    for prediction in predictions:
        max_pred = max(prediction)
        min_pred = min(prediction)
        i = 0

        reduced_prediction = []
        while i < limit:
            curr_max_pred = max(prediction)
            curr_max_pred_i = prediction.index(curr_max_pred)
            if curr_max_pred_i != 9:
                normalized_max_pred = (curr_max_pred - min_pred) / (max_pred - min_pred)
                reduced_prediction.append((normalized_max_pred, chr(curr_max_pred_i + 65)))
                i = i + 1
            prediction[curr_max_pred_i] = 0

        reduced_predictions.append(reduced_prediction)
    return reduced_predictions

def calculate_subtree(k_neg1_layer, k_layer):
    subtree = []
    for k_neg1_prob, k_neg1_char in k_neg1_layer:
            for k_prob, k_char in k_layer:                
                subtree.append((k_neg1_prob * k_prob, k_neg1_char + k_char))
    return subtree

def caclulate_outcome_probabilities(predictions):
    # Initialize first subtree products
    final_idex = len(predictions) - 1
    current_subtree = calculate_subtree(predictions[final_idex - 1], predictions[final_idex])

    # Build up outcome probabilities by working back through predictions tree,
    # calculating each subtree in turn
    for letter_pos in range(final_idex - 1, 0, -1):
        current_subtree = calculate_subtree(predictions[letter_pos - 1], current_subtree)

    return current_subtree


# Begin Interactive Image Capture script to collect Input Images
image_path = camera_capture.begin_image_capture()

# Run ASL Detector CNN model on Input Images
best_guess, predictions = asl_detector_model.translate_asl_images(image_path)

# Check if Prediction with Highest Certainty is a Valid Word
check_word_result = word_search.check_word(best_guess)
if check_word_result == -1: # Terminate if error occurred
    print('ERROR - Exception thrown. Unable to perform translation.')
elif  check_word_result == 0: # Return best guess if valid
    print('Translation: ' + best_guess)
else: 
    # Run decision making algorithm on top 5 predicitions for each image
    top_predictions = get_top_predictions(predictions, 5)
    outcome_probabilities = caclulate_outcome_probabilities(top_predictions)
    outcome_probabilities.sort(key = sortFirst, reverse = True)

    for prob, word in outcome_probabilities:
        print('prob: ' + str(prob) + ', word: ' + word)
        check_word_result = word_search.check_word(word)
        if check_word_result == -1: # Terminate if error occurred
            print('ERROR - Exception thrown. Unable to perform translation.')
            break
        elif  check_word_result == 0: # Return best guess if valid
            print('Translation: ' + best_guess)
            break