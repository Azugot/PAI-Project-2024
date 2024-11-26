import os
import scipy.io
import numpy as np
from PIL import Image

def save_images_from_mat(mat_file_path, output_folder):
    """
    Opens a .mat file, extracts images, and saves them in a specified folder.
    
    Args:
        mat_file_path (str): Path to the .mat file.
        output_folder (str): Path to the output folder where images will be saved.
    """
    # Load the .mat file
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        data_array = mat_data['data']  # Adjust key based on your specific .mat file structure
    except KeyError:
        print("The .mat file does not have a 'data' key.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for patient_index, patient_data in enumerate(data_array[0]):
        images = patient_data['images']
        for image_index, image in enumerate(images):
            # Convert the image array to a PIL Image
            image_array = np.array(image)
            pil_image = Image.fromarray(image_array)

            # Define the file name and save the image
            file_name = f"ROI_{patient_index:02}_{image_index}.png"
            file_path = os.path.join(output_folder, file_name)
            pil_image.save(file_path)
            print(f"Saved: {file_path}")

if __name__ == "__main__":
    # Input: Replace this with the actual .mat file path
    mat_file_path = "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"
    output_folder = "FullImageTestSet"

    save_images_from_mat(mat_file_path, output_folder)
    print(f"All images saved to {output_folder}.")
