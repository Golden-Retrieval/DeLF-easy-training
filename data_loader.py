# -*- coding: utf_8 -*-
from sklearn.model_selection import train_test_split
import time
import os 
import tensorflow as tf


def time_checker(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('[ {:s} ] function : {:.3f} s'.format(f.__name__, (time2-time1)))

        return ret
    return wrap


@time_checker
def check_train_dataset(data_path):
    train_data_dir = os.path.join(data_path, "train")
    labels = os.listdir(train_data_dir)
    for label in labels:
        label_path = os.path.join(train_data_dir, label)
        if not os.path.isdir(label_path):
            print("ERROR: check this file : " + label_path)
            raise Exception("data_path should follow guide line : Please read train tutorial file")
    return 

        
@time_checker
def check_infer_dataset(data_path):
    test_data_dir = os.path.join(data_path, "test")
    imgs = os.listdir(test_data_dir)
    imgs = list(filter(lambda name: ".ipy" not in name, imgs))
    for img in imgs:
        img_path = os.path.join(test_data_dir, img)
        if not os.path.isfile(img_path):
            print("ERROR: check this file : " + img_path)
            raise Exception("data_path should follow guide line : Please read train tutorial file")
    return 


@time_checker
def _ReadImageList(list_path):
    """Helper function to read image paths.

    Args:
      list_path: Path to list of images, one image path per line.

    Returns:
      image_paths: List of image paths.
    """
    with tf.gfile.GFile(list_path, 'r') as f:
        image_paths = f.readlines()
    image_paths = [entry.rstrip() for entry in image_paths]
    return image_paths


@time_checker
def _label_to_int(labels):
    """
    Get all the labels and convert it with ascending numbers.
    """
    result = []
    label_book = {}
    numbering = 0
    for label in labels:
        try:
            result.append(label_book[label])
        except:
            label_book[label] = numbering
            result.append(numbering)
            numbering += 1
    return result


@time_checker
def load_dataset(directory, no_label=False):
    """
    Get all the images and labels in data_path/train/classes/*.jpg
    """
    if no_label:
        filenames = [os.path.join(directory, file) for file in os.listdir(directory)]
        labels = [0 for i in filenames]
        return filenames, labels
    else:
        labels = os.listdir(directory)
        files_and_labels = []
        for label in labels:
            for f in os.listdir(os.path.join(directory, label)):
                if '.ipy' not in f:
                    files_and_labels.append(
                        (os.path.join(directory, label, f), label))

        filenames, labels = zip(*files_and_labels)
        filenames = list(filenames)
        labels = list(labels)
        int_labels = _label_to_int(labels)
        return filenames, int_labels




@time_checker
def _parse_function(filename, label, size=(224, 224)):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_decoded = tf.image.resize_images(image_decoded, size)
    image = tf.cast(image_decoded, tf.float32)
    return image, label


def pipe_data(config, data_path, validation_size=0.1):

    ############### [ read dataset ] ##################
    batch_size = config.batch_size
    filenames, labels = load_dataset(data_path)
    print("[ number of file names ] = %d" % len(filenames))

    ########## [ split train, validation set ] ######### 
    train_filenames, val_filenames, \
    train_labels, val_labels = train_test_split(filenames,
                                                labels,
                                                test_size=validation_size,
                                                random_state=42)

    # === spliting set should not reduce train class ==== 
    assert len(set(labels)) == len(set(train_labels))
    config.num_classes = len(set(labels))
    print("[ classes ] = " + str(config.num_classes))

    ############### [ calulate num batches ] ############
    num_train_data = len(train_filenames)
    num_train_batches = int(num_train_data / config.batch_size)
    num_val_data = len(val_filenames)
    num_val_batches = int(num_val_data / config.batch_size)

    ############## [ pipeline dataset  ] ##############
    # pipeline training data 
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_filenames, train_labels))
    train_dataset = train_dataset.shuffle(
        buffer_size=len(train_filenames))
    train_dataset = train_dataset.map(_parse_function,
                                      num_parallel_calls=config.num_preprocess_threads).prefetch(
        batch_size)

    # pipeline validation data 
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (val_filenames, val_labels))
    validation_dataset = validation_dataset.shuffle(
        buffer_size=len(train_filenames))
    validation_dataset = validation_dataset.map(_parse_function,
                                                num_parallel_calls=config.num_preprocess_threads).prefetch(batch_size)

    ################ [ make batch ] ####################
    batched_train_dataset = train_dataset.batch(batch_size)
    batched_validation_dataset = validation_dataset.batch(
        batch_size)

    ############### [ make iterator ] ##################
    iterator = batched_train_dataset.make_initializable_iterator()
    config.train_init_op = iterator.make_initializer(
        batched_train_dataset)
    config.validation_init_op = iterator.make_initializer(
        batched_validation_dataset)

    ############# [ generate input data ] ################
    images, labels = iterator.get_next()
    return images, labels, num_train_batches, num_val_batches

