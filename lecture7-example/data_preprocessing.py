from glob import glob
import multiprocessing as mp

import numpy as np
from PIL import Image
from tqdm import tqdm

N_PER_CATEGORY = 150

def load_images_for_class(class_info):
    dog_index, dog_breed = class_info

    class_images = glob(f"{dog_breed}/*")
    if len(class_images) < N_PER_CATEGORY:
        return None
    class_images = class_images[:N_PER_CATEGORY]

    images = []
    labels = []
    for image_file in class_images:
        image = Image.open(image_file)
        image = image.resize((224, 224))
        image = image.convert("RGB")
        image = np.array(image)
        images.append(image)
        labels.append(dog_index)

    return np.array(images), np.array(labels)

def load_images():
    images = []
    labels = []
    label_mapping = {}

    classes = glob("Images/*")
    loading_information = list(enumerate(classes))
    
    for (index, _) in loading_information:
        label_mapping[index] = classes[index].split("/")[-1]

    with mp.Pool() as pool:
        results = list(tqdm(pool.imap(load_images_for_class, loading_information), total=len(loading_information)))

    for result in results:
        if result is not None:
            result_images, result_labels = result
            images.append(result_images)
            labels.append(result_labels)

    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0), label_mapping

images, labels, label_mapping = load_images()
np.save("images.npy", images)
np.save("labels.npy", labels)
np.save("label_mapping.npy", label_mapping)
