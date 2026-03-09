import os 
import zipfile
import kaggle

# DEFINING DATASET NAME ON KAGGLE AND DESTINATION FOLDER
dataset_name = "vishweshsalodkar/wild-animals"
dest_folder = "./animals"

def main():
    # CREATING THE FOLDER DESTINATION IF DOESN'T EXISTS
    os.makedirs(dest_folder, exist_ok=True)

    # DOWNLOADING AND UNZIPING DATASET WITH KAGGLE API
    try:
        kaggle.api.dataset_download_files(dataset_name, path=dest_folder, unzip=True)
    except Exception as e:
        print("Error! Make sure you have your ~/.kaggle/access_token.json")
        print(f"Error details: {e}")
    
if __name__ == "__main__":
    main()