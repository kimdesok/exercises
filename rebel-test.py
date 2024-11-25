import rebel
#print("Rebel imported successfully")

import pandas as pd
import numpy as np
import time
import io

import tensorflow as tf
#print(tf.__version__)

# Download the model weights into this folder
import os
os.environ['KERAS_HOME'] = '/home/work/data/keras'
os.environ["HF_HOME"] = "/home/work/models/huggingface"
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0': INFO, '1': WARNING, '2': ERROR, '3': FATAL
tf.get_logger().setLevel('ERROR')  # Suppress other TensorFlow warnings


from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

from tensorflow.keras.models import load_model

from sklearn.metrics import precision_score, \
recall_score, f1_score, accuracy_score, \
confusion_matrix, classification_report

from sklearn.metrics import confusion_matrix

from datasets import load_dataset
#from datasets import load_metric

from transformers import TFAutoModelForImageClassification, AutoImageProcessor, AutoFeatureExtractor
from transformers import DefaultDataCollator
from transformers import AdamWeightDecay

from PIL import Image
from matplotlib import pyplot as plt
from datetime import datetime, timezone, timedelta

# size variable set according to the processor attributes
size = [224, 224]

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

val_data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.CenterCrop(size[0], size[1]),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ],
    name="val_data_augmentation",
)

def preprocess_test(example_batch):
    """Apply val_transforms across a batch."""
    images = [
        val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["images"] = [tf.transpose(tf.squeeze(image)) for image in images]
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
    example_batch["images"] = [tf.transpose(tf.squeeze(image)) for image in images]
    return example_batch
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def evaluate_model(test_ds): 
    #Batching applied 
    #New transformed dataset
    test_dsx = test_ds.with_transform(transformx)
    
    #the same formatting with validation set
    test_dsx.set_transform(preprocess_val)
    test_dsx.set_transform(preprocess_test)
    
    #Convert the test dataset to tensorflow format
    test_set = test_dsx.to_tf_dataset(
        columns=["images", "label"],
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

def load_image(dataset_name):
    # test set for evaluation at the end
    #print(f"Name of dataset: {dataset_name} at {custom_cache_dir}")
    test_ds = load_dataset(dataset_name, split="test", cache_dir=custom_cache_dir)
    #print("No. of test sets loaded yet wait for the user selection:", len(test_ds))
    return test_ds

def softmax(logits):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# This is the relevant inference operation. the rest is not used.
def perform_inference(runtime_module, dataset):
    predictions = []
    labels = []
    start_time = time.time()
    
    for batch in dataset:
        # Extract the inputs and labels from the batch
        img_array = batch['image']  # Ensure the input key aligns with your dataset processing
        img_label = [batch['label']]

        # Resize the image
        img_array = img_array.resize((size[0], size[1]))

        #print(type(img_array), img_array.size, img_array.mode)
        #print(type(batch_label), batch_label)
        
        # Change it to numpy array and then apply type change & arithmetic operations.
        img_array = np.array(img_array).astype('float32')
        #print("Data type:", img_array.dtype) 

        #print("Min value:", img_array.min())
        #print("Max value:", img_array.max())

        # Normalize the image
        img_array = img_array / 127.5 - 1.0
        
        # Change the type
        img_array = img_array.astype('float32')
       
        # Ensure the image array is contiguous before passing to the model
        # img_array = np.ascontiguousarray(img_array) 

        img_array = np.expand_dims(img_array, axis=0)
        
        #print(type(img_array), img_array.shape)
        
        # Run inference
        output = runtime_module(img_array)
        #print(f"\n Output: {output}")
        
        # Use argmax to determine the most likely class
        predicted_label  = (output > 0.5).astype(int)
        
        #print(img_label, predicted_label)

        # Collect predictions and labels for evaluation
        predictions.extend(predicted_label)
        labels.extend(img_label)
    
    #print(f"Inference time : {1000 * (time.time()-start_time)/len(labels):.4f} msecs")
    #print(np.array(predictions).flatten().shape, np.array(labels).shape)
    return np.array(predictions).flatten(), np.array(labels)

def process_image(_test_dss, module):
    data_df = pd.DataFrame({"label": _test_dss["label"]})

    label2id, id2label = dict(), dict()
    for i, label in enumerate(set(true_labels)):
        label2id[label] = str(i)
        id2label[str(i)] = label
 
    test_dsx = _test_dss.with_transform(transform)
    test_dsx.set_transform(preprocess_val)
    
    test_set = test_dsx.to_tf_dataset(
        columns=["image", "label"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    outputs = module.run(test_set)
    pred_labels = inference_test(outputs, id2label)
    
    return pred_labels

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

import json
import sys
from pathlib import Path

# Inference performed in rebel-text.propert
if len(sys.argv) > 1:
    json_path = sys.argv[1]
    with open(json_path, "r") as f:
        params = json.load(f)

    # Access the parameters
    no_images = params.get("no_images")
    model_path = params.get("model_path")
    dataset_path = params.get("dataset_path")
    custom_cache_dir = params.get("custom_cache_dir")

    #print(f"Received parameters: {no_images}, {model_path}, {dataset_path}, {custom_cache_dir}\n")
else:
    print("No parameters provided!\n")

# setting hyperparameters
learning_rate = 5e-5
weight_decay = 0.01
batch_size = 512
epochs = 5


#checkpoint = 'google/vit-base-patch16-224-in21k'
#dataset_path = "1aurent/PatchCamelyon"  #dataset of a patch type histologic images of breast cancers 
#custom_cache_dir="/home/work/data"

#*streamlit parameters
optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)

# Rebellion's model 
# Load the compiled model if it is available
#model_path = "/home/work/exercises/models/VGG16.rbln" #*streamlit parameter
if model_path.endswith('.rbln'):
    module = rebel.Runtime(model_path)  
    #print(f"Rebel module loaded is {module}\n")

elif model_path.endswith('.h5') or model_path.endswith('.keras'):
    #print(f"Upload the model from {model_path}\n")
    module = load_model(model_path)
    
    # Compile a TF model
    func = tf.function(lambda input_img : module(input_img))
    input_info = [('input_img', [1, 224, 224, 3], tf.float32)]

    #print(f"Compile the model from {model_path}\n")
    compiled_model = rebel.compile_from_tf_function(func, input_info)

    # Save the compiled model
    model_path = Path(model_path)
    #model_path = model_path.split('.')[0] + '.rbln'
    model_path = model_path.with_suffix('.rbln')
    compiled_model.save(model_path)
    print(f"Save the model to {model_path}\n")


data_collator = DefaultDataCollator(return_tensors="np")
#model_name = model_path.split('/')[-1]
#model_name = model_path.name

#print(f"\nModel is {model_name}\n")
#print("Learning rate : ", learning_rate)
#print("Weight decay : ", weight_decay)
#print("Epochs : ", epochs)
#print("Batch size :",  batch_size)

# Load the image dataset
test_ds = load_image(dataset_path)

#print(test_ds)
#print("Type of the dataset", type(test_ds))
#print("Dataset shape", test_ds.shape)
#print("Dataset column names", test_ds.column_names)

my_current_time = datetime.now(tz=timezone(timedelta(hours=9)))
#print(f"The current time is {my_current_time.__str__()}\n")

#no_images = 60  ### streamlit parameter
test_dss = sample_some(test_ds, no_images)
#print("Sampled Dataset shape", test_dss.shape)
#print("Sampled Dataset column names", test_dss.column_names)
true_labels = test_dss['label']

if no_images > 0 :
    start_time = time.time()
    
    #print(f"Processing images...\n")
    predictions, true_labels = perform_inference(module, test_dss)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')

    #print(f"Inference Results")
    print("------------------------------------------------------------------------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("------------------------------------------------------------------------------------------")
    
    e_time = time.time()-start_time
    print(f"Elapsed time = {e_time:.3f} secs for {no_images} images")
    print(f"Inference speed = {no_images/e_time:.1f} images per sec")
    print(f"Inference time per image = {e_time/no_images*1000:.1f} msec per image")
    print("------------------------------------------------------------------------------------------")
    # Display the plot with labels on the image 
    #print(f"Generating the gallery of {no_images} images....")
    buf, no_false = show_images(test_dss, predictions, no_images)
    
    # Display the plot with labels in the side cells
    #st.image(buf, caption='Prediction gallery')

    # Display the image using PIL (useful in VS Code or notebooks)
    #img = Image.open(buf)
    #img.show()  # Displays in VS Code if `code-server` has image support

    #print(f"Accuracy = {(no_images-no_false)/no_images:.4f}")
    print(f"Out of {no_images}, {no_images-no_false} correctly predicted.")
    #print("------------------------------------------------------------------------------------------")
    
    # Call the function to display the gallery
    #print(f"Generating the gallery of a mockup annotation column....")
    #display_image_gallery(test_dss, predictions)

    tf.keras.backend.clear_session()

