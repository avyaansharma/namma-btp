import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def try_format_plate(text):
    """
    Try to format and validate the plate text against UK or Indian formats,
    correcting common OCR character misreadings.
    """
    patterns = {
        'UK': ['L', 'L', 'D', 'D', 'L', 'L', 'L'],
        'IN_STD_8': ['L', 'L', 'D', 'D', 'D', 'D', 'D', 'D'],
        'IN_STD_9': ['L', 'L', 'D', 'D', 'L', 'D', 'D', 'D', 'D'],
        'IN_STD_10': ['L', 'L', 'D', 'D', 'L', 'L', 'D', 'D', 'D', 'D'],
        'IN_BH_9': ['D', 'D', 'L', 'L', 'D', 'D', 'D', 'D', 'L'],
        'IN_BH_10': ['D', 'D', 'L', 'L', 'D', 'D', 'D', 'D', 'L', 'L']
    }
    
    letters = set(string.ascii_uppercase)
    digits = set(string.digits)
    
    char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'B': '8', 'Z': '2'}
    int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '8': 'B', '2': 'Z'}
    
    n = len(text)
    matching_patterns = [p for p in patterns if len(patterns[p]) == n]
    if not matching_patterns:
        return None
        
    for p_name in matching_patterns:
        pat = patterns[p_name]
        formatted = []
        possible = True
        for i in range(n):
            char = text[i]
            expected = pat[i]
            if expected == 'L':
                if char in letters:
                    formatted.append(char)
                elif char in int_to_char:
                    formatted.append(int_to_char[char])
                else:
                    possible = False
                    break
            elif expected == 'D':
                if char in digits:
                    formatted.append(char)
                elif char in char_to_int:
                    formatted.append(char_to_int[char])
                else:
                    possible = False
                    break
        
        if possible:
            formatted_str = "".join(formatted)
            # For BH series, verify letters at positions 2 and 3 are indeed BH
            if p_name.startswith('IN_BH'):
                if formatted_str[2:4] != 'BH':
                    continue
            return formatted_str
            
    return None


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    best_raw_text = None
    best_raw_score = -1

    for detection in detections:
        bbox, text, score = detection

        # Standardize OCR output: uppercase, no spaces/hyphens
        text = text.upper().replace(' ', '').replace('-', '').replace('_', '').replace('.', '')

        if score > best_raw_score:
            best_raw_score = score
            best_raw_text = text

        formatted_text = try_format_plate(text)
        if formatted_text is not None:
            return formatted_text, score

    if best_raw_text is not None and len(best_raw_text) > 3: # Basic sanity check
        return best_raw_text, best_raw_score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate
    plate_cx = (x1 + x2) / 2
    plate_cy = (y1 + y2) / 2

    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        # Relaxed check: check if the center of the plate is within the car's bounding box (with margin)
        if xcar1 - 50 < plate_cx < xcar2 + 50 and ycar1 - 50 < plate_cy < ycar2 + 50:
            return vehicle_track_ids[j]

    return -1, -1, -1, -1, -1
