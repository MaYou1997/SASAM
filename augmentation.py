from PIL import Image
import os

def convert_heic_to_jpg(input_folder, output_folder):
    """
    批量将一个文件夹中的 HEIC 图片转换为 JPG 格式
    :param input_folder: 包含 HEIC 图片的输入文件夹路径
    :param output_folder: 转换后的 JPG 图片的输出文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件扩展名是否为 HEIC（支持大小写）
        if filename.lower().endswith(".heic"):
            # 构建完整的输入文件路径
            input_path = os.path.join(input_folder, filename)
            # 构建输出文件路径（将 HEIC 替换为 JPG）
            output_filename = filename[:-5] + ".jpg"  # 去掉.HEIC后缀，添加.jpg后缀
            output_path = os.path.join(output_folder, output_filename)

            try:
                # 打开 HEIC 图片
                with Image.open(input_path) as img:
                    # 转换为 RGB 模式（JPG 不支持透明度）
                    img = img.convert("RGB")
                    # 保存为 JPG 格式
                    img.save(output_path, "JPEG")
                print(f"转换完成：{filename} -> {output_filename}")
            except Exception as e:
                print(f"转换失败：{filename}，错误信息：{e}")

# 示例用法
if __name__ == "__main__":
    # 输入文件夹路径（包含 HEIC 图片）
    input_folder_path = "D:\esu\Seg"  # 替换为你的输入文件夹路径
    # 输出文件夹路径（保存 JPG 图片）
    output_folder_path = r"D:\esu\Seg\1"  # 替换为你的输出文件夹路径
    convert_heic_to_jpg(input_folder_path, output_folder_path)