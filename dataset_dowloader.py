import os
from datasets import load_dataset

def download_hf_dataset():
    # Nome del repository su Hugging Face
    dataset_name = "srikanthgali/ai-text-detection-pile-cleaned"
    
    # Cartella locale dove salvare il dataset sul tuo computer/HPC
    output_dir = "./ai_text_detection_dataset"
    
    print(f" Avvio del download per il dataset: '{dataset_name}'...")
    
    try:
        # Scarica il dataset (sfrutta la cache locale di Hugging Face)
        dataset = load_dataset(dataset_name)
        
        print("\n Download completato con successo!")
        print("-" * 40)
        print("Struttura del dataset scaricato:")
        print(dataset)
        print("-" * 40)
        
        # Salva il dataset in locale sul disco
        print(f"Salvataggio del dataset in corso nella cartella: {output_dir}")
        dataset.save_to_disk(output_dir)
        
        print(f" Completato! Path assoluto del dataset: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"\n Si è verificato un errore durante il download: {e}")

if __name__ == "__main__":
    download_hf_dataset()
