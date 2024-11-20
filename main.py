from inference import inference
def main():
    # 输入本地图像文件的路径
    image_path = "D:/shiyantupian/test3.png"

    # 调用inference函数进行OCR识别
    recognized_text = inference(image_path)

    # 输出识别结果
    if recognized_text:
        print(f"识别的文本：{recognized_text}")
    else:
        print("没有识别到文本")


if __name__ == "__main__":
    main()
