import os
import json


def rename_files_with_json(directory):
    # 遍历指定目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            # 构建JSON文件的完整路径
            json_file_path = os.path.join(directory, filename)

            # 读取JSON文件内容
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

            # 获取新的图像文件名
            new_image_name = data.get('imagePath')
            if new_image_name:
                # 获取图像文件的当前名称（假设图像文件和JSON文件同名，扩展名不同）
                current_image_name = filename.replace('.json', os.path.splitext(new_image_name)[1])
                current_image_path = os.path.join(directory, current_image_name)
                new_image_path = os.path.join(directory, new_image_name)

                # 重命名图像文件
                if os.path.exists(current_image_path):
                    os.rename(current_image_path, new_image_path)
                    print(f"Renamed image: {current_image_path} to {new_image_path}")
                else:
                    print(f"Image file not found: {current_image_path}")

                # 重命名JSON文件
                new_json_name = os.path.splitext(new_image_name)[0] + '.json'
                new_json_path = os.path.join(directory, new_json_name)
                os.rename(json_file_path, new_json_path)
                print(f"Renamed JSON: {json_file_path} to {new_json_path}")


# 设置目标目录
target_directory = "/work/datasets/RUBBY/20240718_purchase/20240722/labelme/house"

# 执行重命名操作
rename_files_with_json(target_directory)
