from PIL import Image
from paddle.vision.transforms import ColorJitter, Grayscale, BrightnessTransform
import os
import json

def check_and_filter_images(json_file_path, image_dir, new_json_file):
    # 读取原始JSON文件
    with open(json_file_path, "r") as f:
        json_data = json.load(f)
    
    new_data = {"images": [], "type": json_data["type"], "annotations": [], "categories": json_data["categories"]}
    
    for image_info in json_data["images"]:
        file_name = image_info["file_name"]
        image_path = os.path.join(image_dir, file_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {file_name} not found in {image_dir}. Skipping.")
            continue
        
        file_size = os.path.getsize(image_path)
        large_edge = max(image_info["height"], image_info["width"])
        
        if file_size > 14 * 1024 * 1024 or large_edge > 4096:
            new_data["images"].append(image_info)
            
            image_id = image_info["id"]
            image_annotations = [anno for anno in json_data["annotations"] if anno["image_id"] == image_id]
            new_data["annotations"].extend(image_annotations)
    
    with open(new_json_file, "w") as f:
        json.dump(new_data, f)

def augment_and_save_images(json_file_path, filtered_json_file, image_dir):
    # Read filtered JSON file
    with open(filtered_json_file, "r") as f:
        filtered_json_data = json.load(f)
    # 7.23 Read original JSON file
    with open(json_file_path, "r") as f:
        json_data = json.load(f)
    
    # Initialize new image_id and annotation_id to increment from the last available IDs
    last_image_id = max(image_info["id"] for image_info in json_data["images"])
    last_annotation_id = max(anno["id"] for anno in json_data["annotations"])
    
    for image_info in filtered_json_data["images"]:
        file_name = image_info["file_name"]
        image_path = os.path.join(image_dir, file_name)
        img = Image.open(image_path).convert('RGB')
            
        # Augmentation 1: ColorJitter
        transform = ColorJitter(0.5, 0.5, 0.5, 0.5)
        augmented_img = transform(img)
        out_path = os.path.join(image_dir, 'ColorJitter_' + file_name)
        augmented_img.save(out_path, quality=80)
        
        # Augmentation 2: Grayscale
        transform = Grayscale()
        augmented_img = transform(img)
        out_path = os.path.join(image_dir, 'Grayscale_' + file_name)
        augmented_img.save(out_path, quality=80)
        
        # Augmentation 3: BrightnessTransform
        transform = BrightnessTransform(0.4)
        augmented_img = transform(img)
        out_path = os.path.join(image_dir, 'BrightnessTransform_' + file_name)
        augmented_img.save(out_path, quality=80)
        
        # Update "images" field in the JSON data with new image_info
        new_image_info1 = {
            "file_name": 'ColorJitter_' + file_name,
            "height": image_info["height"],
            "width": image_info["width"],
            "id": last_image_id + 1
        }
        json_data["images"].append(new_image_info1)
        
        new_image_info2 = {
            "file_name": 'Grayscale_' + file_name,
            "height": image_info["height"],
            "width": image_info["width"],
            "id": last_image_id + 2
        }
        json_data["images"].append(new_image_info2)
        
        new_image_info3 = {
            "file_name": 'BrightnessTransform_' + file_name,
            "height": image_info["height"],
            "width": image_info["width"],
            "id": last_image_id + 3
        }
        json_data["images"].append(new_image_info3)
        
        # Increment the last_image_id after adding 3 augmented images
        last_image_id += 3

        # Get the image_id for the current image_info
        image_id = image_info["id"]

        # Update "annotations" field in the JSON data with new annotations for the augmented images
        for anno in filtered_json_data["annotations"]:
            if anno["image_id"] == image_id:
                # Create new annotations for the augmented images
                new_annotation1 = {
                    "area": anno["area"],
                    "iscrowd": anno["iscrowd"],
                    "bbox": anno["bbox"],
                    "category_id": anno["category_id"],
                    "ignore": anno["ignore"],
                    "image_id": last_image_id - 2,  # Image ID for ColorJitter augmented image
                    "id": last_annotation_id + 1
                }
                json_data["annotations"].append(new_annotation1)
                last_annotation_id += 1

                new_annotation2 = {
                    "area": anno["area"],
                    "iscrowd": anno["iscrowd"],
                    "bbox": anno["bbox"],
                    "category_id": anno["category_id"],
                    "ignore": anno["ignore"],
                    "image_id": last_image_id - 1,  # Image ID for Grayscale augmented image
                    "id": last_annotation_id + 1
                }
                json_data["annotations"].append(new_annotation2)
                last_annotation_id += 1

                new_annotation3 = {
                    "area": anno["area"],
                    "iscrowd": anno["iscrowd"],
                    "bbox": anno["bbox"],
                    "category_id": anno["category_id"],
                    "ignore": anno["ignore"],
                    "image_id": last_image_id,  # Image ID for BrightnessTransform augmented image
                    "id": last_annotation_id + 1
                }
                json_data["annotations"].append(new_annotation3)
                last_annotation_id += 1

    # Save the updated JSON data to the original filtered_json_file
    with open(json_file_path, "w") as f:
        json.dump(json_data, f)

# 输入的json文件路径和图像目录
json_file_path = "/home/aistudio/offline_transforms/grid_train_720.json"
image_directory = "/home/aistudio/data/data212110/JPEG"
filtered_json_file = "/home/aistudio/offline_transforms/filtered.json"

# 使用check_and_filter_images()函数来过滤并保存满足条件的图片信息到filtered.json
check_and_filter_images(json_file_path, image_directory, filtered_json_file)

# 对满足条件的图片进行增强并更新json_file_path.json
augment_and_save_images(json_file_path, filtered_json_file, image_directory)

print("Done. Augmented images saved and JSON file updated.")
