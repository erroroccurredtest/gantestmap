import os
import numpy as np
import re
from sklearn.model_selection import train_test_split

def extract_timing_points(file_path):
    with open(file_path, encoding='utf-8') as file:
        content = file.read()

    timing_points = []
    in_timing_points_section = False

    for line in content.split('\n'):
        if line.startswith('['):
            in_timing_points_section = line.startswith('[TimingPoints]')
            continue

        if not in_timing_points_section:
            continue

        if not line.strip():
            break

        values = line.strip().split(',')
        timing_points.append([float(values[0]), float(values[1])])

    return np.array(timing_points)

def extract_hit_objects(file_path):
    with open(file_path, encoding='utf-8') as file:
        content = file.read()

    hit_objects = []
    in_hit_objects_section = False

    slider_pattern = re.compile(r'slider_points:(\S+)')

    for line in content.split('\n'):
        if line.startswith('['):
            in_hit_objects_section = line.startswith('[HitObjects]')
            continue

        if not in_hit_objects_section:
            continue

        if not line.strip():
            break

        values = line.strip().split(',')
        x, y, time = map(float, (values[0], values[1], values[2]))

        object_type = 'circle'
        slider_points = []

        if '2' in values[3]:
            object_type = 'slider'
            if len(values) > 5:
                slider_points = [tuple(map(float, point.split(':'))) for point in slider_pattern.findall(values[5])]

        hit_objects.append([x, y, time, object_type, slider_points])

    return np.array(hit_objects, dtype=object)

def load_osu_data(file_paths):
    timing_points_data = []
    hit_objects_data = []
    metadata_data = []  # You need to define and extract metadata

    for file_path in file_paths:
        timing_points = extract_timing_points(file_path)
        hit_objects = extract_hit_objects(file_path)
        metadata = None  # You need to define and extract metadata

        timing_points_data.append(timing_points)
        hit_objects_data.append(hit_objects)
        metadata_data.append(metadata)

    return np.array(timing_points_data, dtype=object), np.array(hit_objects_data, dtype=object), np.array(metadata_data, dtype=object)

def normalize_data(data, max_values):
    normalized_data = []
    for item in data:
        norm_item = []
        for ho in item:
            norm_ho = ho.copy()
            norm_ho[0] /= max_values[0]
            norm_ho[1] /= max_values[1]
            norm_ho[2] /= max_values[2]
            norm_item.append(norm_ho)
        normalized_data.append(norm_item)
    return np.array(normalized_data, dtype=object)

def normalize_timing_points_data(data, max_values):
    normalized_data = []
    for item in data:
        norm_item = item.copy()
        for idx, max_value in enumerate(max_values):
            norm_item[:, idx] /= max_value
        normalized_data.append(norm_item)
    return np.array(normalized_data, dtype=object)

def split_data(normalized_timing_points, normalized_hit_objects, train_size=0.7, test_size=0.2, random_state=42):
    X = np.column_stack((normalized_timing_points, normalized_hit_objects))
    y = np.array([i for i in range(len(X))])  # dummy labels for compatibility with train_test_split

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=test_size/(train_size+test_size), random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    osu_directory = r'D:\\traindatabase'
    osu_file_paths = [os.path.join(osu_directory, file) for file in os.listdir(osu_directory) if file.endswith('.osu')]

    timing_points, hit_objects, metadata = load_osu_data(osu_file_paths)

    osu_data = {'timing_points': timing_points, 'hit_objects': hit_objects, 'metadata': metadata}
    np.save('osu_data.npy', osu_data)

    # Calculate the maximum values for timing points and hit objects
    max_timing_point_time = np.max([np.max(tp[:, 0]) for tp in timing_points])
    max_timing_point_beat_length = np.max([np.max(tp[:, 1]) for tp in timing_points])

    max_hit_object_x = -np.inf
    max_hit_object_y = -np.inf
    max_hit_object_time = -np.inf
    for ho_array in hit_objects:
        for ho in ho_array:
            max_hit_object_x = max(max_hit_object_x, float(ho[0]))
            max_hit_object_y = max(max_hit_object_y, float(ho[1]))
            max_hit_object_time = max(max_hit_object_time, float(ho[2]))

   # Normalize data
    normalized_timing_points = normalize_timing_points_data(timing_points, [max_timing_point_time, max_timing_point_beat_length])
    normalized_hit_objects = normalize_data(hit_objects, [max_hit_object_x, max_hit_object_y, max_hit_object_time])

    # Preprocess metadata
    preprocessed_metadata = preprocess_metadata(metadata)

    # Save preprocessed metadata
    np.save('preprocessed_osu_metadata.npy', preprocessed_metadata)

    # Save normalized data
    normalized_data = {'timing_points': normalized_timing_points, 'hit_objects': normalized_hit_objects}
    np.save('normalized_osu_data.npy', normalized_data)

    # Split the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(normalized_timing_points, normalized_hit_objects)

    # Extract training, validation, and testing data
    timing_points_train, hit_objects_train = X_train[:, 0], X_train[:, 1]
    timing_points_val, hit_objects_val = X_val[:, 0], X_val[:, 1]
    timing_points_test, hit_objects_test = X_test[:, 0], X_test[:, 1]

def load_normalized_data():
    loaded_normalized_data = np.load('normalized_osu_data.npy', allow_pickle=True).item()
    timing_points_normalized = loaded_normalized_data['timing_points']
    hit_objects_normalized = loaded_normalized_data['hit_objects']
    # Load preprocessed metadata
    preprocessed_metadata = np.load('preprocessed_osu_metadata.npy', allow_pickle=True)

    # You can now print some data or perform other checks to ensure the data is as expected.
    print(timing_points_normalized[0])  # This line prints the first normalized timing points data
    print(hit_objects_normalized[0])    # This line prints the first normalized hit objects data

if __name__ == '__main__':
    main()
    load_normalized_data()
