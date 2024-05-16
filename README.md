# Applying Mask R-CNN for Segmenting Backscatter Electron Images of Archaeological Bronze Casting Moulds.

This repository contains deep learning code for instance segmentation of Bronze Casting Moulds particles and detection of their morphological parameters, based on Mask R-CNN. For detailed information, please refer to the paper "Applying Mask R-CNN for Segmenting Backscatter Electron Images of Archaeological Bronze Casting Moulds."

## Dataset

The datasets of bronze casting moulds and cores used in this work were obtained from a middle Shang (14th-13th century BC) bronze casting workshop at Taijiasi. Samples were prepared using standard procedures and subjected to scanning electron microscopy using a Tescan Vega III scanning electron microscope equipped with a Bruker XFlash 6|10 EDS. The BSE-EDS parameters were set to an accelerating voltage of 20 kV, working distance of 15 mm, and live collection duration of 60 seconds. A backscattered electron (BSE) image detector was used to study the microstructures of the samples. Ultimately, we acquired 75 BSE images at 500× magnification. Labeling was guided by a professional archaeologist using the polygon annotation tool in LabelMe software to outline all feret particles > 10μm in each image.

## Installation

If you wish to replicate this project, you can follow these steps:

### Method 1: Using Virtual Environment

Step 1. Create the virtual environment required by the project:
      ```bash
      conda create -n BCM-Segnet python=3.7.9
      ```
Step 2. Activate the virtual environment:
    - On Linux/macOS:
        ```bash
        source activate BCM-Segnet
        ```
    - On Windows:
        ```cmd
        activate BCM-Segnet
        ```
Step 3. Install the required packages:
      ```bash
      python -m pip install -r requirements.txt
      ```  
### Method 2: Using Docker
Step 1. Ensure you have Docker installed on your system. For installation instructions, refer to the [Docker documentation](https://docs.docker.com/get-docker/).

Step 2. Dockerfile：
    ```bash
    FROM python:3.7.9
    LABEL Author=Stars
    WORKDIR /home/project
    RUN apt-get update && apt-get install -y libgl1-mesa-glx
    COPY . .
    RUN python -m pip install --upgrade pip && pip install -r requirements.txt
    ```
    
Step 3. Biuld the Docker
    ```bash
    docker build -t bcm .
    ```
    
    
After setting up the environment, you can choose to proceed with either of the following sections:

 1. For users who wish to use our pre-trained model to process SEM images and obtain data.
 2. For users who want to train the model based on the data we provide or your own dataset.

## 1. Use our pre-trained model to process SEM images and obtain data

Here, we provide two methods for processing SEM images: one using the command line and the other using Jupyter notebook.

### Method 1: Command line

#### Step 1

Navigate to the directory containing the `measure_instance.py` file:

- On Linux/macOS:
    ```bash
    cd path/to/measure_instance.py
    ```
- On Windows (Command Prompt):
    ```cmd
    cd path/to/measure_instance.py
    ```
- On Windows (PowerShell):
    ```powershell
    cd path/to/measure_instance.py
    ```

Replace `path/to/measure_instance.py` with the actual path to the directory containing `measure_instance.py`.

#### Step 2

Test the trained model using `/samples/bit/measure_instance.py` with the following command:

- On Linux/macOS:
    ```bash
    python measure_instance.py <image_path> <model_path> <output_path>
    ```
- On Windows (Command Prompt):
    ```cmd
    python measure_instance.py <image_path> <model_path> <output_path>
    ```
- On Windows (PowerShell):
    ```powershell
    python measure_instance.py <image_path> <model_path> <output_path>
    ```

Please note:

**<image_path>**: This is the image you wish to segment. We have provided three images in the `./sample/bit` folder: `test0`, `test1`, and `test2`. You only need to replace `<image_path>` with the path where your image is located.

**<model_path>**`: The `./sample/bit/model.h5` is our pre-trained model. You only need to replace `<model_path>` with the path where `model.h5` is located. 

 **<output_path>**: This is the location where you want to save the segmentation results. Please specify a folder, such as `./sample/bit/results`.

*For example (please replace the paths accordingly):*
```powershell
python measure_instance.py C:/Users/pc/BCM-Segnet/sample/bit/test1.png C:/Users/pc/BCM-Segnet/sample/bit/model.h5 C:/Users/pc/BCM-Segnet/sample/bit/results
```

#### Step 3

After completing the previous steps, you will have obtained the SEM image segmentation results and data, which are saved in the `output_path` you specified.

The `results` (your output_path) directory contains:

-   `Original BSE image.jpg`, `Segmented image.jpg`, `Sand_Particle.jpg`, `Voids.jpg`, `Matrix.jpg`, and `Proportion_of_Area.png` corresponding to **Figure 9** in the paper.
-   A box plot image named `custom_box_plots.jpg`, which includes the statistical distributions of area, Feret diameter, roundness, and solidity (with the additional Orientation) for all particles, as shown in **Figure 10** in the paper.

### Method 2: Douker
Here $(pwd) indicates the current host directory, which should contain all the contents of the `./sample` folder
The usage is similar to Method 1: Command line.
*For example (please replace the paths accordingly):*
```
docker run --rm -it -v $(pwd):/BCM -w /BCM/sample/bit bcm sh -c "python measure_instance.py /BCM/sample/bit/test1.png  /BCM/sample/bit/model.h5 /BCM/sample/bit/results"
```

### Method 3: Jupyter Notebook

### Step 1

Please use the `/sample/bit/measure.ipynb` notebook with the BCM-Segnet kernel.

### Step 2

Jupyter is very user-friendly. After running the notebook, it will display the predicted image segmentation results, including:

- An Excel file `measure_data.xlsx` generated in the `./sample/bit` directory, which contains data on all particles' area, Feret diameter, roundness, and solidity (with the additional Orientation). You can open this Excel file to check it. With this data, you can perform any analysis you desire.

- Six segmented result images are printed under the `if __name__ == "__main__":` function block, which includes `Original BSE image.jpg`, `Segmented image.jpg`, `Sand_Particle.jpg`, `Voids.jpg`, `Matrix.jpg`, and `Proportion_of_Area.png` corresponding to Figure 9 in the paper.

-  A box plot image named `custom_box_plots.jpg` printed under the `if __name__ == "__main__":` function block, which includes the statistical distributions of area, Feret diameter, roundness, and solidity (with the additional Orientation) for all particles corresponding to Figure 10 in the paper.


## 2. Train the model based on the data we provide or your own data set

### Ⅰ. Data Augmentation

- To augment and save your data and masks, we provide a Jupyter notebook to show the augmented effects. Please use `/sample/bit/Data augmentation.ipynb`.

- If you need to process your own dataset, replace the paths in Module 2 with the paths to your dataset:
    ```python
    image_folder = r'/path/to/your/train/images'
    mask_folder = r'/path/to/your/train/masks'
    augmented_folder = r'/path/to/your/save/folder'
    ```

- If you just want to view the data augmentation effects, we have set up the code to randomly show the augmented effects of three images (`test0`, `test1`, `test2`) in the `/sample/bit` folder, and display them in the Jupyter notebook.

### Ⅱ. Retrain the model
### Methods 1:Command line
#### Step 1
Divide the datasets into three categories: `train`, `val`, and `test`. (Or use the data that we've already processed in the `./data` folder.)

#### Step 2
Navigate to the `/sample/bit` folder using the command line.

#### Step 3
Train the model using `/samples/bit/nucleus.py` with the following command:
```bash
python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=coco
```
Or, to continue training based on a previously trained model, use:
```bash
python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last
```
### Step 4
The trained model will be saved in the `/sample/bit` folder. Detailed training logs will be saved in the `./samples/bit/logs` folder. You can monitor the training process using TensorBoard. The specific definitions of the loss function and evaluation metrics can be found in the paper.
### Step 5
The trained model `model.h5` will be saved in the `/sample/bit` folder.

### Methods 2:Docker
The usage is similar to Method 1: Command line.
*For example (please replace the paths accordingly):*
```
docker run --rm -it -v $(pwd):/BCM -w /BCM/sample/bit bcm sh -c "python3 nucleus.py train --dataset=/BCM/data --subset=train --weights=coco"
```
