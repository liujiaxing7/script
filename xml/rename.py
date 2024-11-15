import os
import re
import xml.etree.ElementTree as ET

# 定义中英文对照表

translation_map = {
    '鞋⼦': 'shoes',
    '垃圾桶': 'bin',
    '底座': 'pedestal',
    '线缆': 'wire',
    '插线板': 'socket',
    '沙发（ignore）': 'couch',
    'cat/dog': 'dog',
    '宠物': 'dog',
    '餐桌': 'desk_rect',
    '桌子': 'desk_rect',
    '体重秤': 'weighing-scale',
    '钥匙': 'key',
    '人': 'person',
    '遥控器': 'remoteControl',
    '床': 'bed',
    '冰箱': 'fridge',
    '洗⾐机': 'washingMachine',
    '电风扇': 'electricFan',
    '屏幕': 'television',
    '椅子': 'chair',
    '鞋柜': 'shoeCabinet',
    '电视柜': 'tvCabinet',
    '沙发': 'couch',
    '线缆/电线': 'wire',
    '电视/电脑屏幕': 'television',
}

def Walk(path, suffix:list):
    file_list = []
    suffix = [s.lower() for s in suffix]
    if not os.path.exists(path):
        print("not exist path {}".format(path))
        return []

    if os.path.isfile(path):
        return [path,]

    for root, dirs, files in os.walk(path):
        for file in files:
            a = os.path.splitext(file)[1].lower()[1:]
            if os.path.splitext(file)[1].lower()[1:] in suffix:
                file_list.append(os.path.join(root, file))

    file_list.sort(key=lambda x:int(re.findall('\d+', os.path.splitext(os.path.basename(x))[0])[0]))

    return file_list

# 遍历文件夹中的XML文件
def process_folder(folder_path):
    for xml_file_path in folder_path:
        process_xml(xml_file_path)


# 处理单个XML文件
def process_xml(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 遍历<object>标签并替换名称
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is not None:
                chinese_name = name_elem.text.strip()
                if chinese_name in translation_map:
                    name_elem.text = translation_map[chinese_name]

        # 保存修改后的XML文件
        tree.write(xml_file, encoding='utf-8', xml_declaration=True)
        print(f"Processed: {xml_file}")

    except Exception as e:
        print(f"Error processing {xml_file}: {e}")


# 调用处理函数，传入包含XML文件的文件夹路径
file_list = Walk("/data/VOC/ABBY/1/Annotations/TRAIN/purchase/", ['xml'])
process_folder(file_list)
