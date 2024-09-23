from PIL import Image  # 注意：虽然导入了PIL，但在这个示例中并未使用
import cv2
import os


def resize_and_grayscale(image_path, output_path):
    """
    将指定路径的图像resize到640x480并转换为灰度图像。

    :param image_path: 输入图像的路径
    :param output_path: 输出图像的路径
    """
    # 打开图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image at {image_path}")
        return

        # 转换为灰度图像
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize图像到640x480
    resized_img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

    # 保存图像
    cv2.imwrite(output_path, resized_img)


def process_images(input_file_path, output_folder):
    """
    从文件中读取图像路径，对每个图像进行resize和灰度处理。

    :param input_file_path: 包含图像路径的文本文件路径
    :param output_folder: 输出图像的文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # 读取图像路径
    with open(input_file_path, 'r') as file:
        image_paths = file.readlines()

        # 遍历图像路径，处理每个图像
    for path in image_paths:
        # 去除路径末尾的换行符
        path = path.strip()
        # 构造输出路径，使用原始文件名（可能包含扩展名）
        filename = os.path.basename(path)
        output_path = os.path.join(output_folder, filename)

        # 处理图像
        try:
            resize_and_grayscale(path, output_path)
            print(f"Processed {path} and saved as {output_path}")
        except Exception as e:
            print(f"Failed to process {path}: {e}")

        # 示例用法


input_file_path = '/work/datasets/RUBBY/20240921_wire_negative/all.txt'  # 修改为你的图像路径文件
output_folder = '/work/datasets/RUBBY/20240921_wire_negative/wire0921_resize_negative/'  # 修改为你的输出文件夹路径
process_images(input_file_path, output_folder)