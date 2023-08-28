import re
import os


def replace_img_url_1(input_path, out_path):
    # 读取文件内容
    with open(input_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 对于 ![image name](Notes.assets/image-000.png) 格式的替换
    pattern1 = r'!\[.*?\]\((Notes\.assets/.*?\.(png|jpg|jpeg|gif|bmp))\)'
    replacement1 = r'!\[.*?\](/assets/images/\1)'
    content = re.sub(pattern1, replacement1, content)

    # 对于 <img src="Notes.assets/image-20230826111831637.png" ... /> 格式的替换
    pattern2 = r'<img src="(Notes\.assets/.*?\.(png|jpg|jpeg|gif|bmp))"'
    replacement2 = r'<img src="/assets/images/\1"'
    content = re.sub(pattern2, replacement2, content)

    # 将修改后的内容写入新文件
    with open(out_path, "w", encoding="utf-8") as file:
        file.write(content)


def replace_img_url_2(input_path, out_path):
    # 读取文件内容
    with open(input_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 对于 ![image name](Notes.assets/image-000.png) 格式的替换
    pattern1 = r'(!\[.*?\])\((/assets/images/Notes\.assets/.*?\.(png|jpg|jpeg|gif|bmp))\)'
    replacement1 = r'\1(/BrainPy-course-notes\2)'
    content = re.sub(pattern1, replacement1, content)

    # 对于 <img src="Notes.assets/image-20230826111831637.png" ... /> 格式的替换
    pattern2 = r'<img src="(/assets/images/Notes\.assets/.*?\.(png|jpg|jpeg|gif|bmp))"'
    replacement2 = r'<img src="/BrainPy-course-notes\1"'
    content = re.sub(pattern2, replacement2, content)

    # 将修改后的内容写入新文件
    with open(out_path, "w", encoding="utf-8") as file:
        file.write(content)


if __name__ == "__main__":
    # 打印当前工作目录
    # print(os.getcwd())
    input = "_posts/2023-08-27-BrainPy-course-notes.md"
    out = "_posts/2023-08-27-BrainPy-course-notes_replace.md"
    replace_img_url_2(input, out)
