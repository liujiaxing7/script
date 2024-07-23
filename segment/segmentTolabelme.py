import os
import json


def convert_to_labelme_format(input_data):
    shapes = []
    for region in input_data['region']:
        if 'points' not in region['coordinates']:
            continue  # Skip this region if 'points' key is missing

        shape = {
            "label": region['coordinates']['class'],
            "points": [],
            "group_id": region.get('groupId', None),
            "shape_type": region['type'],
            "flags": {}
        }

        # Ensure all points are floats
        points = region['coordinates']['points']
        points = [float(p) for p in points]

        # Group points into pairs
        shape['points'] = [[points[i], points[i + 1]] for i in range(0, len(points), 2)]

        shapes.append(shape)

    labelme_format = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": input_data['file']['name'],
        "imageData": None,
        "imageHeight": 1024,  # Assumes fixed height, update as needed
        "imageWidth": 1024  # Assumes fixed width, update as needed
    }

    return labelme_format


def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, 'r', encoding='utf-8') as infile:
                input_data = json.load(infile)

            labelme_data = convert_to_labelme_format(input_data)

            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(labelme_data, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_directory = "/work/datasets/RUBBY/20240718_purchase/20240718/7.18数据5809/障碍物/"
    output_directory = "/work/datasets/RUBBY/20240718_purchase/20240718/labelme/障碍物/"

    process_directory(input_directory, output_directory)
