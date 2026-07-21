import zipfile
import os

def setup_collab_data(zip_path, extract_to='/content/data'):
    if not os.path.exists(zip_path):
        print(f"BŁĄD: Nie znaleziono pliku {zip_path}")
        return

    print(f"Rozpakowywanie {zip_path} do {extract_to}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    print("Rozpakowywanie zakończone sukcesem!")