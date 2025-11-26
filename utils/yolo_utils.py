import os
import yaml
import cv2
import random
import glob
from tqdm import tqdm

def create_yolo_labels(dataset_root, folder_id='02'):
    """
    Generate .txt files for YOLO starting from LINEMOD's gt.yml.
    """
    # Instead of creating a separate “labels” folder, we put the txt files 
    # in the same folder as the images (“rgb”). YOLO will definitely find them.
    labels_dir = os.path.join(dataset_root, 'data', folder_id, 'rgb')
    images_dir = os.path.join(dataset_root, 'data', folder_id, 'rgb')
    yaml_path = os.path.join(dataset_root, 'data', folder_id, 'gt.yml')

    # Quick check: let's count the .txt files in the rgb folder
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if len(txt_files) > 100:
        print(f"[INFO] Labels già presenti in {labels_dir}. Salto generazione.")
    else:
        print(f"[INFO] Generazione labels YOLO da {yaml_path}...")
        
        # Map ID LINEMOD -> ID YOLO
        id_map = {
            1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 8: 5, 
            9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12
        }

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        count = 0
        for img_id, objects in tqdm(data.items(), desc="Creazione .txt"):
            filename = f"{img_id:04d}"
            
            # Paths
            img_path = os.path.join(images_dir, filename + ".png")
            txt_path = os.path.join(labels_dir, filename + ".txt")
            
            if not os.path.exists(img_path): continue
            #hard coded to save time
            w_img, h_img = 640, 480 

            yolo_lines = []
            for obj in objects:
                obj_id = obj['obj_id']
                if obj_id not in id_map: continue

                x_min, y_min, w_box, h_box = obj['obj_bb']

                x_center = (x_min + w_box / 2) / w_img
                y_center = (y_min + h_box / 2) / h_img
                w_norm = w_box / w_img
                h_norm = h_box / h_img

                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))

                yolo_lines.append(f"{id_map[obj_id]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            if yolo_lines:
                with open(txt_path, 'w') as f_out:
                    f_out.write("\n".join(yolo_lines))
                    count += 1
        print(f"Labels generated in {labels_dir}.")

def create_yolo_config(dataset_root, folder_id='02', train_size=1000):
    """
    1. Create two .txt files (path lists) to split train and val.
    2. Create the data.yaml file that points to these lists.
    """
    images_dir = os.path.join(dataset_root, 'data', folder_id, 'rgb')
    
    #Find all .png images
    search_path = os.path.join(images_dir, "*.png")
    all_images = glob.glob(search_path)
    all_images = sorted(all_images)
    
    total_imgs = len(all_images)
    print(f"find {total_imgs} images in {folder_id}.")
    
    if total_imgs == 0:
        raise ValueError("No images found.Check the path")

    #Shuffle
    random.seed(42)
    random.shuffle(all_images)
    
    limit = train_size if total_imgs > train_size else int(total_imgs * 0.8)
    
    train_imgs = all_images[:limit]
    val_imgs = all_images[limit:]
    
    print(f"[INFO] Split created: {len(train_imgs)} Train, {len(val_imgs)} Validation.")

    # Save the lists in two .txt files
    train_list_path = os.path.join(dataset_root, 'autosplit_train.txt')
    val_list_path = os.path.join(dataset_root, 'autosplit_val.txt')
    
    with open(train_list_path, 'w') as f:
        f.write('\n'.join(train_imgs))
        
    with open(val_list_path, 'w') as f:
        f.write('\n'.join(val_imgs))
        
    config = {
        'path': dataset_root, 
        'train': train_list_path,
        'val': val_list_path,
        'names': {
            0: 'Ape',         
            1: 'Benchvise',   
            2: 'Cam',          
            3: 'Can',         
            4: 'Cat',          
            5: 'Driller',     
            6: 'Duck',         
            7: 'Eggbox',       
            8: 'Glue',        
            9: 'Holepuncher',  
            10: 'Iron',        
            11: 'Lamp',        
            12: 'Phone'       
        }

    }


    
    config_path = os.path.join(dataset_root, 'linemod_yolo_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"[INFO] Configurazione salvata in: {config_path}")
    return config_path