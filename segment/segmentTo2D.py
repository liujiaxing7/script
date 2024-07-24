import json
import os
from xml.etree.ElementTree import Element, SubElement, ElementTree


def create_xml(file_name, file_path, img_width, img_height, objects):
    annotation_elem = Element('annotation')

    filename_elem = SubElement(annotation_elem, 'filename')
    filename_elem.text = file_name

    path_elem = SubElement(annotation_elem, 'path')
    path_elem.text = file_path

    source_elem = SubElement(annotation_elem, 'source')
    database_elem = SubElement(source_elem, 'database')
    database_elem.text = 'Unknown'

    size_elem = SubElement(annotation_elem, 'size')
    width_elem = SubElement(size_elem, 'width')
    width_elem.text = str(img_width)
    height_elem = SubElement(size_elem, 'height')
    height_elem.text = str(img_height)
    depth_elem = SubElement(size_elem, 'depth')
    depth_elem.text = '1'

    segmented_elem = SubElement(annotation_elem, 'segmented')
    segmented_elem.text = '0'

    for obj in objects:
        object_elem = SubElement(annotation_elem, 'object')
        name_elem = SubElement(object_elem, 'name')
        name_elem.text = obj['label']

        pose_elem = SubElement(object_elem, 'pose')
        pose_elem.text = 'Unspecified'

        truncated_elem = SubElement(object_elem, 'truncated')
        truncated_elem.text = '0'

        difficult_elem = SubElement(object_elem, 'difficult')
        difficult_elem.text = '0'

        distance_elem = SubElement(object_elem, 'distance')
        distance_elem.text = '0'

        score_elem = SubElement(object_elem, 'score')
        score_elem.text = '0.0'

        bndbox_elem = SubElement(object_elem, 'bndbox')
        xmin_elem = SubElement(bndbox_elem, 'xmin')
        xmin_elem.text = str(obj['bbox'][0])
        ymin_elem = SubElement(bndbox_elem, 'ymin')
        ymin_elem.text = str(obj['bbox'][1])
        xmax_elem = SubElement(bndbox_elem, 'xmax')
        xmax_elem.text = str(obj['bbox'][2])
        ymax_elem = SubElement(bndbox_elem, 'ymax')
        ymax_elem.text = str(obj['bbox'][3])

    return annotation_elem


def calculate_bounding_box(points):
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def process_json_to_xml(json_file, img_file, output_folder):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    objects = []
    for shape in data['shapes']:
        bbox = calculate_bounding_box(shape['points'])
        objects.append({
            'label': shape['label'],
            'bbox': bbox
        })

    file_name = os.path.basename(data['imagePath'])
    file_path = data['imagePath']
    img_width = data['imageWidth']
    img_height = data['imageHeight']

    annotation_elem = create_xml(file_name, file_path, img_width, img_height, objects)

    xml_file_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.xml')
    tree = ElementTree(annotation_elem)
    tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)


def batch_process(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            json_file = os.path.join(input_folder, file_name)
            img_file = os.path.join(input_folder, file_name.replace('.json', '.jpg'))
            if os.path.exists(img_file):
                process_json_to_xml(json_file, img_file, output_folder)
                print(f'Processed: {file_name} and generated XML.')


# Example usage
input_folder = '/work/datasets/RUBBY/20240718_purchase/20240722/labelme/house/'
output_folder = '/work/datasets/RUBBY/20240718_purchase/20240722/labelme/house_xml/'

batch_process(input_folder, output_folder)
