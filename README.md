# BCM-SegNet
This is a deep learning code for instance segmentation of Bronze Casting Moulds particles and detection of their morphological parameters, based on Mask R-CNN. For detailed information, please refer to the paper "Applying Mask R-CNN for Segmenting Backscatter Electron Images of Archaeological Bronze Casting Moulds."
# Dataset
The datasets of bronze casting moulds and cores used in this work were from a middle Shang (14th-13th century BC) bronze casting workshop at Taijiasi.  Samples were prepared using standard procedures, then subjected to scanning electron microscopy using a Tescan Vega III scanning electron microscope equipped with a Bruker XFlash 6|10 EDS.  The BSE-EDS parameters were set to an accelerating voltage of 20 kV, working distance of 15 mm, and live collection duration of 60s.  A backscattered electron (BSE) image detector was used to study the microstructures of the samples. Ultimately, we acquired 75 BSE images at 500× magnification.  Labeling is guided by a professional archaeologist, using the polygon annotation tool in LabelMe software to outline all feret particles > 10μm in each image.
# Dataset Download
Here we provide the complete annotated dataset, which contains two folders, origin_image&groundtruth (75 images) and image_augmentation (331 images), which you can download and view：https://1drv.ms/f/s!Am4YApEyA-XwqA6Zl8yqqIuyJcwZ?e=b9YvsF
# Install
If you wish to replicate this project, you can follow these steps:
1. Create the virtual environment required in requirements.
2. Divide the datasets into three categories: train, val, test.
3. Train the model using code/samples/bit/nucleus.py with the following command:
python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=coco
Or, to continue training based on a previously trained model, use:
python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last
4. Test with the trained model using code/samples/bit/measure_instances.py. You will get a predicted image, a table containing all instance Areas, Feret Diameter, Roundness, and Solidity, and the area ratios of Particles, Matrix, and Voids:
python measure_instances.py path/to/your_pic path/to/your_model.h5 path/to/your_excel.xlsx show_image=True
# Data Show 
Some of the original images and the corresponding ground truth are shown below：
![01_0](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/5063822d-fa54-4c0f-bc1f-ec6003aed800)
![01_overlay](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/771a6cf3-fb88-4e69-8cc4-fd6ed6c21fa4)
![1-1](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/0f9a2db5-523b-4d87-87b3-f0e1d90a32ab)
![1-1_overlay](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/cd81be0e-81dc-45bb-9ef2-84bd9c2d23d7)
![3-2](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/636744ef-cdb1-4fc1-9c27-2ffb762e3c1b)
![3-2_overlay](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/40d9100d-99cd-4ccf-a783-30ec7ab71fcb)
# Data augmentation
The image_augmentation is generated using the data_augmentation.py script from the code directory, employing operations such as Affine, Fliplr, Flipud, and ElasticTransformation for data augmentation.
Take an image for example, after it is augmented:
![07](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/aae2dabb-1007-44c6-9e10-93fa2efe78a7)
![07_0](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/0a56e4d3-9338-4e65-8db8-873ead89b1f8)
![07_1](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/0863f40d-b14d-4a87-8656-626800d62866)
![07_2](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/0801cca4-77b9-45bc-9d0e-52858fca2be0)
# The result of our segmentation
![73_1](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/eb4ef8f7-b73e-4d74-81ce-9d6774c6fa3f)
![73_1_seg](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/aed7bb07-0cfc-4a55-9b0a-a6c5a736d884)
![D_mould](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/a17440bf-83ee-446c-ab42-05d62b1e5b38)
![微信图片_20240207154734](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/5bf42c90-8174-41cd-95e8-f4e7e4fa0804)
# To calculate the area ratios of Particles, Matrix, and Voids:
![图8(a)](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/62f2f620-464b-4fc3-abb0-83f9d8131fa9)
![图8(b)](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/5f18e612-1e9b-4e84-a7cc-5bce098f57bf)
# To calculation and statistics of all the instance Areas, Feret Diameter, Roundness, and Solidity
![图9](https://github.com/LingyuLiao/BCM-SegNet/assets/159132984/9d052592-023a-413e-8e15-38383fe1c44e)







