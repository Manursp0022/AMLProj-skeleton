import os
import yaml
import cv2
import random
import glob
from tqdm import tqdm

def create_yolo_labels(dataset_root, folder_id='02'):
    """
    Genera i file .txt per YOLO partendo dal gt.yml di LINEMOD.
    """
    labels_dir = os.path.join(dataset_root, 'data', folder_id, 'labels')
    images_dir = os.path.join(dataset_root, 'data', folder_id, 'rgb')
    yaml_path = os.path.join(dataset_root, 'data', folder_id, 'gt.yml')

    os.makedirs(labels_dir, exist_ok=True)

    # Controllo rapido
    if len(os.listdir(labels_dir)) > 100:
        print(f"[INFO] Labels già presenti in {labels_dir}. Salto generazione.")
    else:
        print(f"[INFO] Generazione labels YOLO da {yaml_path}...")
        
        # Mappa ID LINEMOD -> ID YOLO
        id_map = {
            1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 8: 5, 
            9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12
        }

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        count = 0
        for img_id, objects in tqdm(data.items(), desc="Creazione .txt"):
            filename = f"{img_id:04d}"
            
            # Percorsi
            img_path = os.path.join(images_dir, filename + ".png")
            txt_path = os.path.join(labels_dir, filename + ".txt")
            
            if not os.path.exists(img_path): continue
            
            # Ottimizzazione: Leggiamo dimensioni solo se necessario
            # Assumiamo 640x480 standard Linemod per velocità, 
            # se variano decommenta cv2.imread
            # img = cv2.imread(img_path)
            # h, w = img.shape[:2]
            w_img, h_img = 640, 480 

            yolo_lines = []
            for obj in objects:
                obj_id = obj['obj_id']
                if obj_id not in id_map: continue

                x_min, y_min, w_box, h_box = obj['obj_bb']

                # Calcoli YOLO
                x_center = (x_min + w_box / 2) / w_img
                y_center = (y_min + h_box / 2) / h_img
                w_norm = w_box / w_img
                h_norm = h_box / h_img

                # Clipping
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))

                yolo_lines.append(f"{id_map[obj_id]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            if yolo_lines:
                with open(txt_path, 'w') as f_out:
                    f_out.write("\n".join(yolo_lines))
                    count += 1
        print(f"[INFO] Label generate.")

def create_yolo_config(dataset_root, folder_id='02', train_size=1000):
    """
    1. Crea due file .txt (liste di percorsi) per splittare train e val.
    2. Crea il data.yaml che punta a queste liste.
    """
    images_dir = os.path.join(dataset_root, 'data', folder_id, 'rgb')
    
    # 1. Troviamo tutte le immagini .png
    # Usiamo glob per trovare i file reali
    search_path = os.path.join(images_dir, "*.png")
    all_images = glob.glob(search_path)
    all_images = sorted(all_images) # Ordiniamo prima per riproducibilità
    
    total_imgs = len(all_images)
    print(f"[INFO] Trovate {total_imgs} immagini in {folder_id}.")
    
    if total_imgs == 0:
        raise ValueError("Nessuna immagine trovata! Controlla il path.")

    # 2. Mischiamo e dividiamo (Shuffle & Split)
    random.seed(42) # FONDAMENTALE: il seed deve essere fisso così lo split è sempre uguale
    random.shuffle(all_images)
    
    # Se abbiamo abbastanza immagini, usiamo il numero fisso, altrimenti percentuale
    limit = train_size if total_imgs > train_size else int(total_imgs * 0.8)
    
    train_imgs = all_images[:limit]      # Primi 1000
    val_imgs = all_images[limit:]        # Restanti 213
    
    print(f"[INFO] Split creato: {len(train_imgs)} Train, {len(val_imgs)} Validation.")

    # 3. Salviamo le liste in due file .txt dentro la cartella del dataset
    # YOLO vuole i path assoluti scritti riga per riga
    train_list_path = os.path.join(dataset_root, 'autosplit_train.txt')
    val_list_path = os.path.join(dataset_root, 'autosplit_val.txt')
    
    with open(train_list_path, 'w') as f:
        f.write('\n'.join(train_imgs))
        
    with open(val_list_path, 'w') as f:
        f.write('\n'.join(val_imgs))
        
    # 4. Creiamo il config yaml puntando ai file .txt
    config = {
        'path': dataset_root, 
        'train': train_list_path,  # <--- Ora punta al file .txt!
        'val': val_list_path,      # <--- Ora punta al file .txt!
        'names': {
            0: 'Ape', 1: 'Morsa', 2: 'Cam', 3: 'Lattina', 4: 'Gatto', 
            5: 'Trapano', 6: 'Papera', 7: 'Uova', 8: 'Colla', 
            9: 'Perforatrice', 10: 'Ferro', 11: 'Lampada', 12: 'Telefono'
        }
    }
    
    config_path = os.path.join(dataset_root, 'linemod_yolo_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"[INFO] Configurazione salvata in: {config_path}")
    return config_path