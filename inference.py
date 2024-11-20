import os
import cv2
import numpy as np
from paddlex import create_pipeline
import tempfile
from typing import List, Optional
import requests
import urllib.request



FONT_PATH = ""
PIPELINE_CONFIG = "D:/Desktop/OCR.yaml"
OUTPUT_DIR = "./output"
OUTPUT_DIR_PADDEX = "./output_paddlex"
VISUALIZE = False  # 根据需要设置是否可视化

# 定义模型存储路径
model_dir = "/root/.paddlex/official_models/"
det_model_name = "PP-OCRv4_mobile_det.pdmodel"
rec_model_name = "PP-OCRv4_mobile_rec.pdmodel"
det_model_url = "https://1111.com/PP-OCRv4_mobile_det.pdmodel"
rec_model_url = "https://1111.com/PP-OCRv4_mobile_rec.pdmodel"


# 检查并下载模型
def check_and_download_model(model_name, model_url, save_dir):
    model_path = os.path.join(save_dir, model_name)

    if not os.path.exists(model_path):
        os.makedirs(save_dir, exist_ok=True)
        download_model(model_url, model_path)

# 下载模型函数
def download_model(model_url, save_path):
    try:
        # 从指定URL下载模型文件
        response = requests.get(model_url)

        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
        else:
            pass
    except Exception:
        pass


# 使用HSV颜色空间查找绿色区域
def find_digit_display_regions(img: np.ndarray, visualize: bool = False) -> Optional[List[List[int]]]:
    if img is None:
        return None

    # 转换HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义绿色HSV范围
    lower_green = np.array([40, 50, 50])  # 绿色低阈值
    upper_green = np.array([80, 255, 255])  # 绿色高阈值

    # 筛选出绿色区域
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 去除噪声
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # 查找所有绿色区域轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 提取所有轮廓的边界框
    bounding_boxes = [cv2.boundingRect(i) for i in contours]

    # 按照y坐标排序，找出最上面的显示框
    bounding_boxes.sort(key=lambda x: x[1])

    if visualize:
        # 可视化绿色区域和边界框
        img_copy = img.copy()
        for box in bounding_boxes:
            x, y, w, h = box
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Green Display Regions", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bounding_boxes

# 根据边界框裁剪图像
def crop_image(img: np.ndarray, bbox: List[int]) -> np.ndarray:
    x_min, y_min, w, h = bbox
    x_max = x_min + w
    y_max = y_min + h

    # 确保裁剪区域不超出图像范围
    img_height, img_width = img.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)

    # 确保裁剪区域的宽度和高度都大于0
    if x_max <= x_min or y_max <= y_min:
        print("裁剪区域无效！")
        return None

    # 裁剪并返回图像
    cropped = img[y_min:y_max, x_min:x_max]
    return cropped

# 使用 PaddleX 模型进行 OCR 识别，返回识别的文本列表
def perform_paddlex_ocr(image_path: str, pipeline_config: str, output_dir: str) -> Optional[List[str]]:
    pipeline = create_pipeline(pipeline=pipeline_config)
    output = pipeline.predict(image_path)

    texts = []
    for res in output:
        # res.save_to_img(output_dir)
        if 'rec_text' in res and res['rec_text']:
            texts.extend(res['rec_text'])

    if texts:
        return texts
    else:
        return None

# 处理输入图像，进行裁剪和 OCR 识别，返回识别的文本列表
def process_image(image: np.ndarray, font_path: str, pipeline_config: str, output_dir: str = "./output",
                  output_dir_paddlex: str = "./output_paddlex", visualize: bool = False) -> Optional[List[str]]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_paddlex, exist_ok=True)

    # 查找绿色数码管显示屏区域
    bounding_boxes = find_digit_display_regions(image, visualize=visualize)
    if bounding_boxes is None:
        return None

    # 选择最上面的第一个框
    top_bbox = bounding_boxes[0]
    # 裁剪最上面数码管区域颜色不变
    cropped_img = crop_image(image, top_bbox)

    if cropped_img is None:
        print("裁剪后的图像为空！")
        return None

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        ori_image_path = tmp_file.name
        cv2.imwrite(ori_image_path, cropped_img)
    try:
        recognized_texts = perform_paddlex_ocr(ori_image_path, pipeline_config, output_dir_paddlex)
    finally:
        os.remove(ori_image_path)
    return recognized_texts


# 读取本地图片并进行推理，返回识别的文本
def inference(image_path: str, font_path: str = FONT_PATH, pipeline_config: str = PIPELINE_CONFIG,
              output_dir: str = OUTPUT_DIR, output_dir_paddlex: str = OUTPUT_DIR_PADDEX,
              visualize: bool = VISUALIZE) -> Optional[str]:

    # # 检查并下载模型
    # check_and_download_model(det_model_name, det_model_url, model_dir)
    # check_and_download_model(rec_model_name, rec_model_url, model_dir)

    try:
        res = urllib.request.urlopen(image_path)
    except Exception:
        return None

    img_array = np.asarray(bytearray(res.read()), dtype="uint8")
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # # 读取本地图片
    # img = cv2.imread(image_path)
    # if img is None:
    #     return None

    texts = process_image(
        image=img,
        font_path=font_path,
        pipeline_config=pipeline_config,
        output_dir=output_dir,
        output_dir_paddlex=output_dir_paddlex,
        visualize=visualize
    )

    return ''.join(texts) if texts else None



# def main():
#     # 输入本地图像文件的路径
#     image_path = "E:/electronic_scale/test3.png"
#
#     # 调用inference函数进行OCR识别
#     recognized_text = inference(image_path)
#
#     # 输出识别结果
#     if recognized_text:
#         print(f"识别的文本：{recognized_text}")
#     else:
#         print("没有识别到文本")
#
#
# if __name__ == "__main__":
#     main()