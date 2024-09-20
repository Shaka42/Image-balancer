import os
import skimage.io
import skimage.transform
import skimage.util

#Enter the Director where the Folders will be Created
data_path = ""
files = os.listdir(data_path)

# Create dataset directories
path = data_path + "/Dataset"
os.makedirs(path, exist_ok=True)

folders = ['train', 'Test', 'Validation']
for i in folders:
    os.makedirs(os.path.join(path, i), exist_ok=True)

# Image resize settings and dataset splits
newsize = (1000,1000, 3)
train_size = 25
validation_size = 10
test_size = 20

# Iterate through image categories and process images
for i in files:
    image_files = [f for f in os.listdir(os.path.join(data_path, i)) if f.lower().endswith(".jpg")]
    
    # Train set
    for im in image_files[:train_size]:
        image_path = os.path.join(data_path, i, im)
        image = skimage.io.imread(image_path)
        resized_image = skimage.transform.resize(image, newsize, anti_aliasing=True)
        
        os.makedirs(os.path.join(path, "train", i), exist_ok=True)
        resized_image_uint8 = skimage.util.img_as_ubyte(resized_image)
        skimage.io.imsave(os.path.join(path, "train", i, im), resized_image_uint8)
    
    # Validation set
    for im in image_files[train_size:train_size + validation_size]:
        image_path = os.path.join(data_path, i, im)
        image = skimage.io.imread(image_path)
        resized_image = skimage.transform.resize(image, newsize, anti_aliasing=True)
        
        os.makedirs(os.path.join(path, "Validation", i), exist_ok=True)
        resized_image_uint8 = skimage.util.img_as_ubyte(resized_image)
        skimage.io.imsave(os.path.join(path, "Validation", i, im), resized_image_uint8)
    
    # Test set
    for im in image_files[train_size + validation_size:]:
        image_path = os.path.join(data_path, i, im)
        image = skimage.io.imread(image_path)
        resized_image = skimage.transform.resize(image, newsize, anti_aliasing=True)
        
        os.makedirs(os.path.join(path, "Test", i), exist_ok=True)
        resized_image_uint8 = skimage.util.img_as_ubyte(resized_image)
        skimage.io.imsave(os.path.join(path, "Test", i, im), resized_image_uint8)
