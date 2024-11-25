import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras import layers

import rebel
print(rebel.__version__)

from datasets import load_dataset
from datasets import load_metric

from transformers import TFAutoModelForImageClassification, AutoImageProcessor, AutoFeatureExtractor
from transformers import DefaultDataCollator
from transformers import AdamWeightDecay

import pandas as pd
import numpy as np

import streamlit as st
from PIL import Image

import time
from datetime import datetime, timedelta, timezone

from matplotlib import pyplot as plt
import seaborn as sns
import io

# TF tensor conversion along with other preprocessings
def convert_to_tf_tensor(image: Image):
    np_image = np.array(image)
    tf_image = tf.convert_to_tensor(np_image)
    # `expand_dims()` is used to add a batch dimension since
    # the TF augmentation layers operates on batched inputs.
    return tf.expand_dims(tf_image, 0)
    
def transformx(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processorx([x for x in example_batch['image']], return_tensors='tf')
    #inputs = feature_extractor([x for x in example_batch['image']], return_tensors='tf')

    # Include the labels
    inputs['labels'] = example_batch['label']
    return inputs

# Shuffle the dataset with a seed for reproducibility
def sample_some(dataset, no_sample):
    shuffled_dataset = dataset.shuffle()
    # Sample a subset of the shuffled dataset
    sampled_indices = range(no_sample)
    #return shuffled_dataset.select(sampled_indices)
    return shuffled_dataset.select(sampled_indices)

def inference_test(outputs, id2label):
    logits = outputs.logits
    #print(logits)
    
    predicted_class_idx = tf.math.argmax(logits, -1).numpy()
    #print(predicted_class_idx)
    
    predicted_label =[]
    for idx in predicted_class_idx:
        predicted_label.append(id2label[str(idx)])
    
    return predicted_label

def preprocess_test(example_batch):
    """Apply val_transforms across a batch."""
    images = [
        val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch['image']], return_tensors='tf')
    #inputs = feature_extractor([x for x in example_batch['image']], return_tensors='tf')

    # Include the labels
    inputs['labels'] = example_batch['label']
    return inputs

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    images = [
        val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

@st.cache_data
def evaluate_model(test_ds): 
    #Batching applied 
    #New transformed dataset
    test_dsx = test_ds.with_transform(transformx)
    
    #the same formatting with validation set
    test_dsx.set_transform(preprocess_val)
    test_dsx.set_transform(preprocess_test)
    
    #Convert the test dataset to tensorflow format
    test_set = test_dsx.to_tf_dataset(
        columns=["pixel_values", "label"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator
    )
        
    #Evaluate the model on the test dataset
    start = time.time()
    eval = loaded_model.evaluate(test_set)
    
    e_time = time.time()-start
    print(f"Elapsed time = {e_time:.3f}")
    print(f"Inference speed = {len(test_set)/e_time:.3f}")
    print(f"Loss = {eval[0]:.4f} Accuracy = {eval[1]:.4f}")
    return eval

@st.cache_data
def load_image(dataset_name):
    # test set for evaluation at the end
    print("Name of dataset:", dataset_name)
    test_ds = load_dataset(dataset_name, split="test[:40]", cache_dir=custom_cache_dir)
    print("No. of test sets loaded yet wait for the user selection:", len(test_ds))
    return test_ds

#Decorate this function with st.cache to only load and process images once
#@st.cache_data
def process_image(_test_dss, module):
    data_df = pd.DataFrame({"label": _test_dss["label"]})
    true_labels = test_ds["label"]

    label2id, id2label = dict(), dict()
    for i, label in enumerate(set(true_labels)):
        label2id[label] = str(i)
        id2label[str(i)] = label
 
    test_dsx = _test_dss.with_transform(transform)
    test_dsx.set_transform(preprocess_val)
    
    test_set = test_dsx.to_tf_dataset(
        columns=["pixel_values", "label"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    pred_labels, true_labels = perform_inference(module, _test_dss)
    #outputs = loaded_model.predict(test_set)
    #pred_labels = inference_test(outputs, id2label)
    
    return pred_labels, true_labels


#@st.cache_data
def display_image_gallery(_dataset_ds, list_preds):
    
    @st.cache_data
    def on_user_input():
    # Action to perform when input changes, e.g., update session state or perform a calculation
        st.session_state.user_input = ""  # Assuming 'input' is the key for the text input widget
    
    for index, row in enumerate(_dataset_ds):
        # Streamlit columns layout
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])  # Adjust the ratio as needed

        with col1:
            st.image(row['image'], use_column_width=True)

        with col2:
        # Use markdown or st.write to display attributes
            st.markdown(f"**Label:** {row['label']}")
        
        with col3:
        #Use markdown or st.write to display attributes
            st.markdown(f"**Prediction:** {list_preds[index]}")

        # Use the fourth column for user input (e.g., expert correction or labeling)
        with col4:
            if(row['label'] == list_preds[index]):
                user_input = st.text_input("Expert label", key="expert_label_"+str(index), disabled=True, value="Correct prediction")#, on_change=on_user_input)
            else:
                user_input = st.text_input("Expert label", key="expert_label_"+str(index), disabled=True, value="Pick up for retraining")
                #st.write("User Input:", user_input)
        #Display the image
        #st.image(image, caption=f"Label: {label}", use_column_width=True)
        
        #Display attributes using markdown with markup parameters
        #st.markdown(f""" 
        #**Label:** {label}
        #""")
        
        #Example of adding more attributes
        #If you have more attributes, you can add them in a similar manner
        #st.markdown(f"**Another Attribute:** {row['another_attribute']}")

#@st.cache_data
def get_input():

    # Initialize session state for number_of_images if it doesn't exist
    if 'number_of_images' not in st.session_state:
        st.session_state.number_of_images = 0
        
    # Define a list of integer values you want buttons for
    #options = [5, 20, 40, 200]
    # Create a row of 4 columns with equal width
    col1, col2, col3, col4 = st.columns(4)
    
    # Create a button for each option
    # if st.button(f'Input {option} Images'):
    # Place a button in each column
    with col1:
        if st.button('Select 4'):
            st.session_state.number_of_images = 4
    with col2:
        if st.button('Select 16'):
            st.session_state.number_of_images = 16
    with col3:
        if st.button('Select 40'):
            st.session_state.number_of_images = 40
    #with col4:
    #    if st.button('Select 200'):
    #        st.session_state.number_of_images = 200
    
    # Proceed with processing if a selection has been made
    if st.session_state.number_of_images > 0:
        st.write(f'{st.session_state.number_of_images} images selected')

    return st.session_state.number_of_images

# Display and label printing

def show_images(test_dataset, pred_text, num_images = 16):
    num_cols = 4
    num_rows = int(num_images/num_cols) + num_images%num_cols  # takes care of the figure with num_cols columns
    #print(num_rows)
    size_x = 24                                   # takes care of the frame size
    size_y = 5* num_rows
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(size_x, size_y))
    axs = axs.flatten()
    no_false = 0
    for i in range(num_images):
        
        image = test_dataset[i]['image']
        label = test_dataset[i]['label']
        label_str = str(label) + " predicted as " +  str(pred_text[i])
        axs[i].imshow(image)
        axs[i].set_xlabel(label_str, fontsize = 13, fontweight ='bold', horizontalalignment ='center')
        #print(label, pred_text[i])
        if(label == pred_text[i]) :
            axs[i].text(5, 10, str(i+1), fontsize = 20, fontweight ='bold', color = 'white', \
                        bbox={'facecolor': 'black', 'alpha': 0.7, 'pad': 10})
        else:
            axs[i].text(5, 10, str(i+1), fontsize = 20, fontweight ='bold', color = 'red', \
                        bbox={'facecolor': 'black', 'alpha': 0.7, 'pad': 10})
            no_false += 1
            
    plt.savefig("inference_test.jpg", bbox_inches='tight', dpi=60)
    # Save it to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', dpi=60)
    buf.seek(0)
    
    return buf, no_false

def softmax(logits):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def perform_inference(runtime_module, dataset):
    predictions = []
    labels = []
    start_time = time.time()
    
    for batch in dataset:
        # Extract the inputs and labels from the batch
        input_images = batch['pixel_values']  # Ensure the input key aligns with your dataset processing
        #input_images = batch['input_1'] 
        batch_labels = batch['labels']

        #print(type(input_images), input_images.shape)
        #print(type(batch_labels), batch_labels)

        img_array = np.transpose(input_images, (0, 3, 2, 1))  # Convert to HWC
        # Normalize the image
        #img_array = img_array / 127.5 - 1.0
        #img_array = np.expand_dims(img_array, axis=0)
        img_array = np.ascontiguousarray(img_array) # Ensure the image array is contiguous before passing to the model
        #print(type(img_array), img_array.shape)
        
        # Run inference
        output = runtime_module.run(img_array)
        #print(output)
        
        #probabilities = softmax(output)
        #print(probabilities)
        
        # Use argmax to determine the most likely class
        #predicted_labels = np.argmax(probabilities, axis=-1) 
        predicted_labels  = (output > 0.5).astype(int)

        #print(batch_labels, predicted_labels)

        # Collect predictions and labels for evaluation
        predictions.extend(predicted_labels)
        labels.extend(batch_labels)
    
    #print(f"Inference time : {1000 * (time.time()-start_time)/len(labels):.4f} msecs")
    print(np.array(predictions).shape, np.array(labels).shape)
    return np.array(predictions), np.array(labels)
    
# setting hyperparameters
learning_rate = 5e-5
weight_decay = 0.01
batch_size = 64
epochs = 5

checkpoint = 'google/vit-base-patch16-224-in21k'
model_path = "/home/ubuntu/exercises/models/Resnet50.rbln"
dataset_path = "1aurent/PatchCamelyon"  #dataset of a patch type histologic images of breast cancers 
custom_cache_dir="/data"

optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)

# Example usage
st.title("Fine-tuning showcase : image classification")
model_name = model_path.split("/")[-1]
dataset_name = dataset_path.split("/")[-1]

st.markdown(f'<h3 style="font-size:26px;color:gray">model : Visual Transformer</h3>', unsafe_allow_html=True)

st.markdown(f'<h3 style="font-size:26px;color:gray;">dataset : Test split of "{dataset_name}" </h3>', unsafe_allow_html=True)
st.write("------------------------------------------------------------------------------------------")

st.write(f'Loading the fine-tuned model, "{model_name}",  from the local repository')
st.write("GPU device 4 in service ")

st.write("Model loading is in progress.")

# Assuming `test_dsx` is already a TensorFlow dataset ready for iteration
module = rebel.Runtime(model_path)  
print(module)

# size variable set according to the processor attributes
size = [224, 224]

data_collator = DefaultDataCollator(return_tensors="np")
#print(processor)

print("Model : ", checkpoint)
print("Learning rate : ", learning_rate)
print("Weight decay : ", weight_decay)
print("Epochs : ", epochs)
print("Batch size :",  batch_size)

#print(eval)

st.write(f'Loading the test images of breast cancer from "{custom_cache_dir}"' )
test_ds = load_image(dataset_path)

# Select the number of test images
st.markdown(f'<h3 style="color:gray;">Select the number of input images </h3>', unsafe_allow_html=True)
no_images = get_input() 
if no_images > 0 :
    
    if st.button(f'Classify {no_images} images?'):

        my_current_time = datetime.now(tz=timezone(timedelta(hours=3)))
        print(my_current_time.__str__())

        st.write(f"Sampling random {no_images} images...")
        test_dss = sample_some(test_ds, no_images)

        start_time = time.time()
        
        st.write(f"Processing images...")
        predictions, true_labels = process_image(test_dss, module)

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        e_time = time.time()-start_time
        st.write(f"Elapsed time = {e_time:.3f} secs for {no_images} images")
        st.write(f"Inference speed = {no_images/e_time:.1f} images per sec")
        st.write("------------------------------------------------------------------------------------------")
        # Display the plot with labels on the image 
        st.write(f"Generating the gallery of {no_images} images....")
        buf, no_false = show_images(test_dss, predictions, no_images)
        
        # Display the plot with labels in the side cells
        st.image(buf, caption='Prediction gallery')

        st.write(f"Out of {no_images}, {no_images-no_false} correctly predicted.")
        st.write(f"Accuracy = {(no_images-no_false)/no_images:.4f}")
        st.write("------------------------------------------------------------------------------------------")
        
        # Call the function to display the gallery
        st.write(f"Generating the gallery of a mockup annotation column....")
        display_image_gallery(test_dss, predictions)
        
        
        
        