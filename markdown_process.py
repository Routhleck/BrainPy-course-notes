import argparse
import re
import os

def prepend_markdown_header(content):
    header = """---
title: BrainPy course notes
tags:
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script> 
MathJax = {
  tex: {
    inlineMath: [['$', '$']],
    processEscapes: true
  }
};
</script>

"""
    return header + content


def replace_img_url_1(content):
    # 对于 ![image name](Notes.assets/image-000.png) 格式的替换
    pattern1 = r'(!\[.*?\])\((Notes\.assets/.*?\.(png|jpg|jpeg|gif|bmp))\)'
    replacement1 = r'\1(/assets/images/\2)'
    content = re.sub(pattern1, replacement1, content)

    # 对于 <img src="Notes.assets/image-20230826111831637.png" ... /> 格式的替换
    pattern2 = r'<img src="(Notes\.assets/.*?\.(png|jpg|jpeg|gif|bmp))"'
    replacement2 = r'<img src="/assets/images/\1"'
    content = re.sub(pattern2, replacement2, content)

    return content


def replace_img_url_2(content):
    # 对于 ![image name](Notes.assets/image-000.png) 格式的替换
    pattern1 = r'(!\[.*?\])\((/assets/images/Notes\.assets/.*?\.(png|jpg|jpeg|gif|bmp))\)'
    replacement1 = r'\1(/BrainPy-course-notes\2)'
    content = re.sub(pattern1, replacement1, content)

    # 对于 <img src="Notes.assets/image-20230826111831637.png" ... /> 格式的替换
    pattern2 = r'<img src="(/assets/images/Notes\.assets/.*?\.(png|jpg|jpeg|gif|bmp))"'
    replacement2 = r'<img src="/BrainPy-course-notes\1"'
    content = re.sub(pattern2, replacement2, content)

    return content


def add_newlines_around_formula(content):
    # 在 $$ 前后添加空行，确保不重复添加空行，并且公式块内部的内容不发生改变
    pattern_start = r'(?<!\n)\n\$\$'
    replacement_start = r'\n\n$$'
    content = re.sub(pattern_start, replacement_start, content)
    
    pattern_end = r'\$\$(?!\n)'
    replacement_end = r'$$\n'
    content = re.sub(pattern_end, replacement_end, content)

    return content


def main(input_path, out_path):
    # 打印当前工作目录
    print(os.getcwd())
    # input_path = "_posts/2023-08-27-BrainPy-course-notes.md"
    # out_path = "_posts/2023-08-27-BrainPy-course-notes_replace.md"
    
    with open(input_path, "r", encoding="utf-8") as file:
        md_content = file.read()
    
    md_content = prepend_markdown_header(md_content)
    md_content = replace_img_url_1(md_content)
    md_content = replace_img_url_2(md_content)
    md_content = add_newlines_around_formula(md_content)

    # 将修改后的内容写入新文件
    with open(out_path, "w", encoding="utf-8") as file:
        file.write(md_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process markdown file.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input markdown file.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output markdown file.')

    args = parser.parse_args()
    main(args.input, args.output)
