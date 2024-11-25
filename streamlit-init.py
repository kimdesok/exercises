import streamlit as st
import subprocess
import json
import os

from datetime import datetime, timezone, timedelta

from PIL import Image

root_folder = "/home/work"  # Replace with the base directory you want to browse

def set_page_config():
    # Set page config to customize the "About" section
    st.set_page_config(
        page_title="Prototype of Pathology Image Inference Engine",  # Your app title
        page_icon="⛵",  # Your app favicon (optional)
        layout="wide",  # Use wide layout (optional)
        initial_sidebar_state="expanded",  # Initial state of the sidebar (optional)
        menu_items={ 
            'About': "Managed by Jk, BTRUST Co., Seoul. Contact dskim@btrust.co.kr for more info. Compute resources provided by NIPA",
            "Report a bug": "https://github.com/kimdesok/Computer-Vision-Transformers",
            "Get Help": "https://github.com/kimdesok/Computer-Vision-Transformers" 
        }
    )

from pathlib import Path


def list_files_in_directory(directory, extension=".rbln"):
    """
    List all files with a specific extension in the given directory.
    """
    # Convert directory to Path object
    dir_path = Path(directory)
    # Find all files with the specified extension
    return [f for f in dir_path.glob(f"*{extension}") if f.is_file()]

def run_rebel_script(params):
    json_path = '/tmp/params.json'
    with open(json_path, 'w') as f:
        json.dump(params, f)

    # Run the rebel-test.py script with the JSON file as input
    process = subprocess.Popen(
        ["python3", "/home/work/exercises/rebel-test.py", json_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Capture output line by line
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        output_lines.append(line.strip())  # Collect the line
        print(line.strip())  # Print it to the console or Streamlit app

    process.stdout.close()
    process.wait()  # Wait for the process to finish

    # Capture any error output
    error_output = process.stderr.read()
    process.stderr.close()

    return output_lines, error_output

def get_input():

    if "number_of_images" not in st.session_state:
        st.session_state.number_of_images = None

    # Create a row of 4 columns with equal width
    col1, col2, col3, col4 = st.columns(4)
    
    # Place a button in each column
    with col1:
        if st.button('4 images'):
            st.session_state.number_of_images = 4
    with col2:
        if st.button('16 images'):
            st.session_state.number_of_images = 16
    with col3:
        if st.button('40 images'):
            st.session_state.number_of_images = 40
    with col4:
        if st.button('200 images'):
            st.session_state.number_of_images = 200

    # Check if a valid number of images has been selected
    if st.session_state.number_of_images is not None:  # Ensure it's not None before comparing
        if st.session_state.number_of_images > 0:
            st.write(f'{st.session_state.number_of_images} images selected.')
    else:
        st.write("Waiting for your selection...")
    
    return st.session_state.number_of_images

# Start with setting page configuration
set_page_config()

# Initializing session states

if "model_path" not in st.session_state:
    st.session_state.model_path = None

# Rebellion performance test on a dataset
dataset_path = "1aurent/PatchCamelyon"  #dataset of a patch type histologic images of breast cancers 
custom_cache_dir="/home/work/exercises/data"
dataset_name = dataset_path.split("/")[-1]

# Top title
st.title("Image classification by NPU compiled models")
st.markdown(f'<h3 style="font-size:26px;color:gray;">dataset : Test on randomly selected images from "{dataset_name}" </h3>', unsafe_allow_html=True)

# Input the number of images to be analyzed 
st.title("Select the number of input images")
status_placeholder = st.empty()
no_images = get_input()

# Set the base directory (adjust as needed)
base_directory = "/home/work/exercises/models"

#if(no_images > 0) :
# Get the model to compile or for inference
st.title("Select a model file")
files = list_files_in_directory(base_directory, extension=".rbln")

# Create a dropdown or selectbox for the user to choose a file
if files:
    # Select file only if files are available
    st.session_state.model_path = st.selectbox("Select a file to load:", files)
else:
    st.write('No suitable model files found')

#Trigger inference
st.title("Start Inference")
if st.button("Start Inference"):
    if st.session_state.number_of_images and st.session_state.model_path:
        # Display selected options
        st.write(f"Number of images selected: {st.session_state.number_of_images}")
        model_name = Path(st.session_state.model_path).name
        st.write(f"Model selected: {model_name}")
        
        st.write(f"Inference is in progress.")
        st.title("Inference Results")

        # Inference performed in rebel-text.py
        params = {"no_images": no_images, "model_path": str(st.session_state.model_path), \
        "dataset_path": str(dataset_path), "custom_cache_dir" : str(custom_cache_dir)}

        # Run inference and display results
        output_lines, error = run_rebel_script(params)

        # Display captured outputs and errors from the rebel script
        for line in output_lines:
            st.write(line)
        if error:
            print("Rebel error:", error)

        st.write("------------------------------------------------------------------------------------------")

        my_current_time = datetime.now(tz=timezone(timedelta(hours=9)))
        formatted_time = my_current_time.strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"The current time", formatted_time)

        st.write("The incorrect results are highlighted as red in the image index")
        
        # Load the image from the local folder
        image_path = "/home/work/exercises/inference_test.jpg"
        if Path(image_path).exists():
            image = Image.open(image_path)
            st.image(image, caption="Inference output image", use_container_width=True)

    else:
        st.warning("Please complete all selections before starting inference.")