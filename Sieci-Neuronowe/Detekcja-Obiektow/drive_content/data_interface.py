import os
import yaml
from loaders import setup_collab_data, get_dataloader

def load_project_data(dataset_name, model_type='pytorch', batch_size=4):
    dataset_name = dataset_name.lower()
    PROJECT_ROOT = '/content/drive/Shareddrives/Sieci_Neuronowe/Projekt_2'
    LOCAL_DATA = f'/content/{dataset_name}_data'
    
    zip_file_names = {
        'bdd': 'bdd',
        'bccd': 'bccd',
        'visdrone': 'VisDrone'
    }
    zip_name = zip_file_names.get(dataset_name, dataset_name) # Fallback if new dataset added without mapping
    ZIP_PATH = f'{PROJECT_ROOT}/data/{zip_name}.zip'

    if not os.path.exists(LOCAL_DATA):
        setup_collab_data(ZIP_PATH, LOCAL_DATA)

    if model_type.lower() == 'yolo':
        return _create_yolo_yaml(dataset_name, LOCAL_DATA)

    return _get_pytorch_loaders(dataset_name, LOCAL_DATA, batch_size)

def _create_yolo_yaml(dataset_name, local_data):

    if dataset_name == 'bccd':
        train_path = os.path.join(local_data, 'train/img')
        val_path = os.path.join(local_data, 'val/img')
    elif dataset_name == 'visdrone':
        train_path = os.path.join(local_data, 'train/images')
        val_path = os.path.join(local_data, 'val/images')
    else: # bdd
        train_path = os.path.join(local_data, 'images/train')
        val_path = os.path.join(local_data, 'images/val')

    data_config = {
        'path': local_data,
        'train': train_path,
        'val': val_path,
        'names': {
            'bccd': {0: 'RBC', 1: 'WBC', 2: 'Platelets'},
            'visdrone': {0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'},
            'bdd': {0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle', 8: 'traffic light', 9: 'traffic sign'}
        }[dataset_name]
    }
    
    yaml_path = f"{local_data}/config.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    return yaml_path

def _get_pytorch_loaders(dataset_name, local_data, batch_size):
    name_map = {'bccd': 'BCCD', 'bdd': 'BDD100K', 'visdrone': 'VisDrone'}
    internal_name = name_map[dataset_name]
    if dataset_name == 'bdd':
        p = {'t_img': f'{local_data}/images/train', 't_ann': f'{local_data}/labels/bdd100k_labels_images_train.json',
             'v_img': f'{local_data}/images/val', 'v_ann': f'{local_data}/labels/bdd100k_labels_images_val.json'}
    elif dataset_name == 'visdrone':
        p = {'t_img': f'{local_data}/train/images', 't_ann': f'{local_data}/train/annotations',
             'v_img': f'{local_data}/val/images', 'v_ann': f'{local_data}/val/annotations'}
    else: # bccd
        p = {'t_img': f'{local_data}/train/img', 't_ann': f'{local_data}/train/ann',
             'v_img': f'{local_data}/val/img', 'v_ann': f'{local_data}/val/ann'}
    train_l = get_dataloader(internal_name, p['t_img'], p['t_ann'], batch_size, train=True)
    val_l = get_dataloader(internal_name, p['v_img'], p['v_ann'], batch_size, train=False)
    return train_l, val_l