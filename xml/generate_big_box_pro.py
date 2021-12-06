import os
from xml.dom import minidom
from xml.etree.ElementTree import ElementTree, Element
from lxml import etree as ET

def read_xml(in_path):
    '''''读取并解析xml文件
       in_path: xml路径
       return: ElementTree'''
    tree = ElementTree()
    tree.parse(in_path)
    return tree


def write_xml(tree, out_path):
    '''''将xml文件写出
       tree: xml树
       out_path: 写出路径'''
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def if_match(node, kv_map):
    '''''判断某个节点是否包含所有传入参数属性
       node: 节点
       kv_map: 属性及属性值组成的map'''
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True


# ----------------search -----------------
def find_nodes(tree, path):
    '''''查找某个路径匹配的所有节点
       tree: xml树
       path: 节点路径'''
    return tree.findall(path)


def get_node_by_keyvalue(nodelist, kv_map):
    '''''根据属性及属性值定位符合的节点，返回节点
       nodelist: 节点列表
       kv_map: 匹配属性及属性值map'''
    result_nodes = []
    for node in nodelist:
        if if_match(node, kv_map):
            result_nodes.append(node)
    return result_nodes


# ---------------change ----------------------
def change_node_properties(nodelist, kv_map, is_delete=False):
    '''修改/增加 /删除 节点的属性及属性值
       nodelist: 节点列表
       kv_map:属性及属性值map'''
    for node in nodelist:
        for key in kv_map:
            if is_delete:
                if key in node.attrib:
                    del node.attrib[key]
            else:
                node.set(key, kv_map.get(key))


def change_node_text(nodelist, text, is_add=False, is_delete=False):
    '''''改变/增加/删除一个节点的文本
       nodelist:节点列表
       text : 更新后的文本'''
    for node in nodelist:
        if is_add:
            node.text += text
        elif is_delete:
            node.text = ""
        else:
            node.text = text


def create_node(tag, property_map, content):
    '''新造一个节点
       tag:节点标签
       property_map:属性及属性值map
       content: 节点闭合标签里的文本内容
       return 新节点'''
    element = Element(tag, property_map)
    element.text = content
    return element


def indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def add_child_node(nodelist, element):
    '''''给一个节点添加子节点
       nodelist: 节点列表
       element: 子节点'''
    for node in nodelist:
        node.append(element)


def del_node_by_tagkeyvalue(nodelist, tag, kv_map):
    '''''同过属性及属性值定位一个节点，并删除之
       nodelist: 父节点列表
       tag:子节点标签
       kv_map: 属性及属性值列表'''
    for parent_node in nodelist:
        children = parent_node.getchildren()
        for child in children:
            if child.tag == tag and if_match(child, kv_map):
                parent_node.remove(child)


def get_xml(input_dir):
    xml_path_list = []
    for (root_path, dirname, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.xml'):
                xml_path = root_path + "/" + filename
                xml_path_list.append(xml_path)
    return xml_path_list


import argparse
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--area', type=int, default="468")
    parser.add_argument('--input_dir', type=str, default="/mnt/sdb2/DATA/x06/anno")
    parser.add_argument('--output_dir', type=str,
                        default="/mnt/sdb2/DATA/x06/anno1")
    args = parser.parse_args()

    xml_path_list = get_xml(args.input_dir)
    for xml_file in tqdm(xml_path_list):
        escalator_handrails=[]

        is_escalator=False
        ################ 1. 读取xml文件  ##########
        root1 = read_xml(xml_file)
        root = root1.getroot()

        # Find annotations.
        Object = root.findall('object')
        for i in range(len(Object)):
            if str(Object[i].find('name').text)=='escalator_handrails':
                ymin_handrails = int(Object[i].find('bndbox').find('ymin').text)  # 修改节点文本
                xmin_handrails = int(Object[i].find('bndbox').find('xmin').text)
                ymax_handrails = int(Object[i].find('bndbox').find('ymax').text)
                xmax_handrails = int(Object[i].find('bndbox').find('xmax').text)
                escalator_handrails.append([xmin_handrails,ymin_handrails,xmax_handrails,ymax_handrails])


        for i in range(len(Object)):
            if str(Object[i].find('name').text)=='dog':
                xminbox = 0
                xmaxbox = 0
                yminbox = 0
                ymaxbox = 0
                ymin = int(Object[i].find('bndbox').find('ymin').text)  # 修改节点文本
                xmin = int(Object[i].find('bndbox').find('xmin').text)
                ymax = int(Object[i].find('bndbox').find('ymax').text)
                xmax = int(Object[i].find('bndbox').find('xmax').text)
                nodes = find_nodes(Object[i], "name")
                for box_handrails in escalator_handrails:
                    if box_handrails[0]>xmin and box_handrails[1]>ymin and box_handrails[2]<xmax and box_handrails[3]<ymax:

                        if box_handrails[0]<xminbox or xminbox==0:
                            xminbox=box_handrails[0]
                        if box_handrails[2]>xmaxbox or xmaxbox==0:
                            xmaxbox=box_handrails[2]
                        if box_handrails[1]<yminbox or yminbox==0:
                            yminbox=box_handrails[1]
                        if box_handrails[3]>ymaxbox or ymaxbox==0:
                            ymaxbox=box_handrails[3]
                Object[i].find('bndbox').find('ymin').text=str(yminbox)
                Object[i].find('bndbox').find('xmin').text=str(xminbox)
                Object[i].find('bndbox').find('xmax').text=str(xmaxbox)
                Object[i].find('bndbox').find('ymax').text=str(ymaxbox)
                # # a=Object[i].find('bndbox')
                # a=Object[i].find('bndbox')
                # a.set('ymin',str(yminbox))  # 修改节点文本
                # b=Object[i].find('bndbox')
                # b.set('xmin',str(xmaxbox))
                # c=Object[i].find('bndbox')
                # c.set('ymax',str(ymaxbox))
                # d=Object[i].find('bndbox')
                # d.set('xmax',str(xmaxbox))

                # text_nodes = get_node_by_keyvalue(find_nodes(root1, "processers/services/service/chain"),
                #                                   {"sequency": "chain3"})
                #
                # change_node_text(text_nodes, "new text")
            #     is_escalator=True
            # else:
            #     continue
        # if is_escalator:
        #     element=create_node("object", {}, "")
        #     element1=create_node("name", {}, "escalator")
        #     element2=create_node("pose", {}, "Unspecified")
        #     element3=create_node("truncated", {}, "0")
        #     element4=create_node("difficult", {}, "0")
        #     element5=create_node("bndbox", {}, "")
        #     element6=create_node("xmin", {}, str(xminbox))
        #     element7=create_node("ymin", {}, str(yminbox))
        #     element8=create_node("xmax", {}, str(xmaxbox))
        #     element9=create_node("ymax", {}, str(ymaxbox))
        #
        #     element.append(element1)
        #     element.append(element2)
        #     element.append(element3)
        #     element.append(element4)
        #     element.append(element5)
        #     element5.append(element6)
        #     element5.append(element7)
        #     element5.append(element8)
        #     element5.append(element9)

            # root.append(element)
        ################ 输出到结果文件  ##########
        output_file = os.path.join(args.output_dir, xml_file[len(args.input_dir) + 1:].strip('/'))
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        write_xml(root1, output_file)
        # file=output_file
        # root = ET.parse(file)
        # file_lines = minidom.parseString(ET.tostring(root, encoding="Utf-8")).toprettyxml(
        #     indent="\t")
        # file_line = open(file, "w", encoding="utf-8")
        # file_line.write(file_lines)
        # file_line.close()
