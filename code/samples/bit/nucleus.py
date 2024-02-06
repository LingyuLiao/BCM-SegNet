"""

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import cv2
import csv
from imgaug import augmenters as iaa
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print('pretrained model:',COCO_WEIGHTS_PATH)

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/bit/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.


############################################################
#  Configurations
############################################################

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 8 // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, 8 // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_ANCHOR_SCALES = (8, 32, 64, 128, 256)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 3000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128        #64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([102.2, 102.2, 102.2])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 512         #256

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 800

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 800


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7 #0.7


############################################################
#  Dataset
############################################################

class NucleusDataset(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "nucleus")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "test", "stage1_train", "stage1_test", "stage2_test"]
                
        if subset == "val":
            self.dataset_dir_imgs = os.path.join(dataset_dir, "images")
  
            image_ids = os.listdir(self.dataset_dir_imgs)
            image_ids = [ele for ele in image_ids if '.png' in ele]
            image_ids = [ele.split('.png')[0] for ele in image_ids]
        else:
            self.dataset_dir_imgs = os.path.join(dataset_dir, "images")

            image_ids = os.listdir(self.dataset_dir_imgs)
            image_ids = [ele for ele in image_ids if '.png' in ele]
            image_ids = [ele.split('.png')[0] for ele in image_ids]
            

        # Add images
        for image_id in image_ids:
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=os.path.join(self.dataset_dir_imgs, image_id + ".png"))


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks",os.path.basename(info['path']).split('.png')[0])


        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Iou
############################################################
def calculate_iou(mask_gt, mask_pred):
    intersection = np.logical_and(mask_gt, mask_pred)
    union = np.logical_or(mask_gt, mask_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def calculate_image_metrics(iou_matrix, num_gt_masks):
    """Calculate TP, FP, FN, and derived metrics for a given IoU matrix."""
    image_TP = np.sum(iou_matrix >= 0.5)
    image_FP = np.sum((iou_matrix > 0) & (iou_matrix < 0.5))
    image_FN = num_gt_masks - image_TP

    image_accuracy = image_TP / (image_TP + image_FP + image_FN) if (image_TP + image_FP + image_FN) > 0 else 0
    image_precision = image_TP / (image_TP + image_FP) if (image_TP + image_FP) > 0 else 0
    image_recall = image_TP / (image_TP + image_FN) if (image_TP + image_FN) > 0 else 0
    image_f1_score = 2 * (image_precision * image_recall) / (image_precision + image_recall) if (
                                                                                                            image_precision + image_recall) > 0 else 0

    return (image_accuracy, image_precision, image_recall, image_f1_score)


def print_metrics(image_id, desc, metrics):
    """Print the metrics for the image."""
    print(
        f"Image {image_id} - {desc} - Accuracy: {metrics[0]}, Precision: {metrics[1]}, Recall: {metrics[2]}, F1 Score: {metrics[3]}")


def save_metrics_to_csv(submit_dir, filename, metrics_list):
    """Save the metrics to a CSV file."""
    metrics_file = os.path.join(submit_dir, filename)
    with open(metrics_file, "w", newline='') as f:
        metrics_writer = csv.writer(f)
        metrics_writer.writerow(["Accuracy", "Precision", "Recall", "F1 Score"])
        for metrics in metrics_list:
            metrics_writer.writerow(metrics)

    print(f"Metrics saved to {metrics_file}")

############################################################
# 计算总体的TP, FP, FN
############################################################
def compute_overall_tp_fp_fn(dataset, iou_threshold=0.5):
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        results = model.detect([image], verbose=0)
        r = results[0]
        for i in range(gt_mask.shape[-1]):
            gt_mask_i = gt_mask[:, :, i]
            max_iou = np.max([calculate_iou(gt_mask_i, r["masks"][:, :, j]) for j in range(r["masks"].shape[-1])])
            if max_iou >= iou_threshold:
                TP += 1
            else:
                FN += 1
        FP += r["masks"].shape[-1] - TP

    return TP, FP, FN

############################################################
#  Training
############################################################

def train(model, dataset_dir):
    """Train the model."""
    # Training dataset.
    dataset_train = NucleusDataset()
    dataset_train.load_nucleus((os.path.join(dataset_dir, "train")), "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleusDataset()
    dataset_val.load_nucleus((os.path.join(dataset_dir, "val")), "val")
    dataset_val.prepare()

    # Image augmentation
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)])
        # iaa.Multiply((0.8, 1.5)),    # 改变图像的亮度
        # iaa.GaussianBlur(sigma=(0.0, 5.0))     # 应用高斯模糊
    ])

    # 定义EarlyStopping和ModelCheckpoint
    early_stop = EarlyStopping(monitor='val_loss', patience=50, mode='min')
    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1,
                                 save_best_only=True, save_weights_only=True, mode='min')

    # Add TensorBoard callback
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=logdir)

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1500,
                augmentation=augmentation,
                layers='heads',
                optimizer_name=config.OPTIMIZER,
                custom_callbacks=[early_stop, checkpoint, tensorboard_callback])

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1500,
                augmentation=augmentation,
                layers='all',
                custom_callbacks=[early_stop, checkpoint, tensorboard_callback])


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset()
    dataset.load_nucleus(os.path.join(dataset_dir, subset), subset)
    dataset.prepare()

    # Lists to store metrics for each image, without IoU filtering and with IoU filtering
    image_metrics_no_filter = []
    image_metrics_filter = []

    # Load over images
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        gt_masks, _ = dataset.load_mask(image_id)  # Assuming this method exists and returns masks

        # Detect objects
        r = model.detect([image], verbose=0)[0]

        # Initialize the IoU matrix
        iou_matrix = np.zeros((gt_masks.shape[-1], r["masks"].shape[-1]))

        # Compute IoU for each ground truth and prediction pair
        for i in range(gt_masks.shape[-1]):
            for j in range(r["masks"].shape[-1]):
                iou_matrix[i, j] = calculate_iou(gt_masks[:, :, i], r["masks"][:, :, j])

        # Calculate metrics without IoU filtering
        image_metrics_no_filter.append(calculate_image_metrics(iou_matrix, gt_masks.shape[-1]))

        # Filter out low IoU values
        high_iou_indices = iou_matrix >= 0.15
        iou_matrix_filtered = np.where(high_iou_indices, iou_matrix, 0)

        # Calculate metrics with IoU filtering
        image_metrics_filter.append(calculate_image_metrics(iou_matrix_filtered, gt_masks.shape[-1]))

        # Print the metrics for this image without and with IoU filtering
        print_metrics(image_id, "No IoU Filter", image_metrics_no_filter[-1])
        print_metrics(image_id, "IoU Filter >= 0.15", image_metrics_filter[-1])

        """
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
         """

    # Save the computed metrics to CSV files
    save_metrics_to_csv(submit_dir, "metrics_no_filter.csv", image_metrics_no_filter)
    save_metrics_to_csv(submit_dir, "metrics_filter.csv", image_metrics_filter)

    print("Results saved to ", submit_dir)
############################################################
#  Command Line
############################################################

def get_activation_map(model_path, image_path, layer_name):
    # 创建一个新的推理配置
    class InferenceConfig(NucleusInferenceConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()

    # 重建模型
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)

    # 加载权重
    model.load_weights(model_path, by_name=True)

    # 加载图像并进行处理
    image = skimage.io.imread(image_path)
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = skimage.color.gray2rgb(image)
    molded_image = modellib.mold_image(image, config)
    molded_image = np.expand_dims(molded_image, 0)

    # 获取指定层的输出
    layer_output = model.keras_model.get_layer(layer_name).output
    intermediate_model = tf.keras.models.Model(inputs=model.keras_model.input, outputs=layer_output)

    # 获取特征图
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        features = intermediate_model.predict(molded_image)

    # 选择一个特征图进行可视化
    feature_map = features[0, :, :, 0]

    # 可视化特征图
    plt.imshow(feature_map, cmap='viridis')
    plt.axis('off')
    plt.show()




if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NucleusConfig()
    else:
        config = NucleusInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    if not os.path.exists("data/detect_test"):
        os.makedirs("data/detect_test")

    # Train or detect
    if args.command == "train":
        train(model, args.dataset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
