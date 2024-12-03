import os
import xml.etree.ElementTree as ET
from pathlib import Path

# 定义类别与对应的面积阈值（像素数）
# 请根据您的实际需求填写这22个类别及其阈值
IMG_SIZE_STD = 4624*3468
CLASS_AREA_THRESHOLDS = {
    "shoes": 30000,
    "bin": 50000,
    "pedestal": 25000,
    "wire": 35000,
    "socket": 35000,
    "cat": 10000,
    "dog": 10000,
    "desk_rect": 85000,
    "desk_circle": 85000,
    "weighing-scale": 20000,
    "key": 20000,
    "person": 75040,
    "chair": 55000,
    "couch": 66000,
    "bed": 75000,
    "tvCabinet": 70000,
    "fridge": 120000,
    "television": 55000,
    "washingMachine": 55000,
    "electricFan": 55000,
    "remoteControl": 35000,
    "shoeCabinet": 55000
}

def calculate_area(bndbox):
    try:
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        width = xmax - xmin
        height = ymax - ymin
        if width < 0 or height < 0:
            print(f"警告: 负面积检测框，xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")
            return 0
        return width * height
    except Exception as e:
        print(f"错误计算面积: {e}")
        return 0

def filter_objects(root, class_thresholds):
    objects = root.findall("object")
    for obj in objects:
        name = obj.find("name").text
        if name in class_thresholds:
            bndbox = obj.find("bndbox")
            area = calculate_area(bndbox)
            threshold = class_thresholds[name]/IMG_SIZE
            if area < threshold:
                root.remove(obj)
                print(f"删除对象: {name}, 面积: {area} < 阈值: {threshold}")

def process_xml_file(input_xml_path, output_xml_path, class_thresholds):
    try:
        tree = ET.parse(input_xml_path)
        root = tree.getroot()
        filter_objects(root, class_thresholds)
        tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
        print(f"处理完成: {input_xml_path} -> {output_xml_path}")
    except ET.ParseError as e:
        print(f"XML解析错误: {input_xml_path}, 错误: {e}")
    except Exception as e:
        print(f"处理文件时出错: {input_xml_path}, 错误: {e}")

def batch_process_xmls(input_dir, output_dir, class_thresholds):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    xml_files = list(input_path.glob("*.xml"))
    if not xml_files:
        print(f"在目录 {input_dir} 中未找到XML文件。")
        return

    for xml_file in xml_files:
        relative_path = xml_file.relative_to(input_path)
        output_file = output_path / relative_path
        process_xml_file(xml_file, output_file, class_thresholds)

if __name__ == "__main__":
    # 设置输入和输出目录
    INPUT_DIR = "/data1/datasets/20240718_purchase/20241113_waicai/xml_1/ck8/"    # 替换为您的XML文件所在的文件夹
    OUTPUT_DIR = "/data1/datasets/20240718_purchase/20241113_waicai/xml_1/ck8/"  # 替换为您希望保存处理后XML文件的文件夹

    # 开始批量处理
    batch_process_xmls(INPUT_DIR, OUTPUT_DIR, CLASS_AREA_THRESHOLDS)