import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
from glob import glob

image_folder = r'F:\TaoFan_Beikeda\USTB_mask_r-cnn\data\train\images'
mask_folder = r'F:\TaoFan_Beikeda\USTB_mask_r-cnn\data\train\masks'
augmented_folder = r'F:\TaoFan_Beikeda\USTB_mask_r-cnn\data\augmentation'

# Define the augmentation sequence
augmentations = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.Affine(rotate=(-10, 10))),
    iaa.Sometimes(0.5, iaa.Fliplr(0.5)),
    iaa.Sometimes(0.5, iaa.Flipud(0.5)),
    iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=8, sigma=3))
])

image_paths = glob(os.path.join(image_folder, '*.png'))

for image_path in image_paths:
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Get the corresponding mask files
    base_name = os.path.basename(image_path).split('.')[0]
    mask_paths = glob(os.path.join(mask_folder, base_name, '*.png'))

    for idx in range(4):  # To get 4 augmented images for each original image
        masks = [cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) for mask_path in mask_paths]

        # Make the augmentations deterministic
        deterministic_augmentations = augmentations.to_deterministic()

        # Apply the augmentations
        image_aug = deterministic_augmentations(image=image)
        masks_aug = [deterministic_augmentations(image=mask) for mask in masks]

        # Save the augmented image
        image_aug_path = os.path.join(augmented_folder, 'images', f'{base_name}_{idx}.png')
        os.makedirs(os.path.dirname(image_aug_path), exist_ok=True)
        cv2.imwrite(image_aug_path, image_aug)

        # Save the augmented masks
        mask_aug_folder = os.path.join(augmented_folder, 'masks', f'{base_name}_{idx}')
        os.makedirs(mask_aug_folder, exist_ok=True)

        for i, mask_aug in enumerate(masks_aug):
            mask_aug_path = os.path.join(mask_aug_folder, f'{i}.png')
            cv2.imwrite(mask_aug_path, mask_aug)
