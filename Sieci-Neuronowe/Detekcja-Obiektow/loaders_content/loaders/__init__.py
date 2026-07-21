from .base_loader import GenericDetectionDataset, get_transforms, collate_fn
from .visdrone_parser import VisDroneDataset
from .bdd_parser import BDD100KDataset
from .bccd_parser import BCCDDataset
from .setup_data import setup_collab_data
from torch.utils.data import DataLoader

# definiujemy co jest dostępne przy "from loaders import *"
__all__ = [
    "GenericDetectionDataset",
    "VisDroneDataset",
    "BDD100KDataset",
    "BCCDDataset",
    "get_transforms",
    "collate_fn",
    "setup_collab_data",
    "get_dataloader"
]

#meta
DATASET_METRICS = {
    "VisDrone": {"classes": 10, "type": "Aerial/Dense"},
    "BDD100K": {"classes": 10, "type": "Driving/Large"},
    "BCCD": {"classes": 3, "type": "Medical/Small"}
}

def get_dataloader(dataset_name, img_dir, annot_path, batch_size=4, train=True):

    transforms = get_transforms(train)
    
    if dataset_name == "BCCD":
        ds = BCCDDataset(img_dir, annot_path, transforms=transforms)
    elif dataset_name == "BDD100K":
        ds = BDD100KDataset(img_dir, annot_path, transforms=transforms)
    elif dataset_name == "VisDrone":
        ds = VisDroneDataset(img_dir, annot_path, transforms=transforms)
    else:
        raise ValueError(f"Nieobsługiwany dataset: {dataset_name}. Wybierz z: {list(DATASET_METRICS.keys())}")
    
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=train, 
        collate_fn=collate_fn,
        num_workers=2 # dla wydajności w Colabie
    )