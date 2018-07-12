# multistream-re-id
This repo implements a salience and semantic-parsing based method for person re-identification.

TODO:
IN THE CODE
comment source code for parts that where based on other people references as well as comment what other people does

UPLOAD TRAINED MODELS
UPLOAD SALIENCE AND SEMANTIC PARSING IMAGES

IN THE DOCUMENTATION
- Show references to saliency repository
- Show references to semantic parsing repository
- show how to prepare data
- show how to run for TRAIN/TEST code for independent subnet
- show how to run for TRAIN/TEST code for whole net
- show how to plot results of ranking
- add Makefile to train-all nets
- add image of architecture
- add results summary
- show dependencies
- decide whether we should divide readme in train and test
### Results
All models are initialized with imagenet pretrained weights.
Results are presented using the format of `Rank1 (mAP)` in the following tables

CUHK03 uses the [new protocol](https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP#State-of-the-art) (767/700)

| Model | # param (M) | Loss | Market1501 | CUHK03 - Detected | CUHK03 - Labeled| DukeMTMC-reID |
| :--------- |     :---:   | :---:|  :---: 	  |      :---:	  | 	  :---: 	  | 	:---:       |
| ResNet50 | 25.04 | xent + htri 				| 88.1 (72.9) | 55.6 (52.9) | 58.8 (56.7) | 77.7 (62.1) |
| ResNet50 + S-ReID| 25.81 | xent + htri 		| 87.6 (73.0) | 56.0 (53.4) | 55.9 (54.4) | 78.9 (63.1) |
| ResNet50 + SP-ReID| 28.89 | xent + htri 		| 87.8 (72.4) | 56.2 (53.5) | 57.3 (55.5) | 78.0 (62.7) |
| ResNet50 + SSP-ReID| 54.7 | xent + htri 		| 89.3 (75.9) | 59.4 (57.1) | 60.6 (58.9) | 80.1 (66.1) |
| ResNet50 + SSP-ReID + RR| 54.7 | xent + htri 	| 91.5 (88.2) | 67.6 (71.1) | 68.4 (72.4) | 84.8 (81.4) |
| ResNet50M | 30.01 | xent + htri 				| 91.2 (77.5) | 58.7 (56.3) | 61.1 (58.9) | 78.8 (63.5) |
| ResNet50M + S-ReID| 30.78 | xent + htri 		| 91.2 (77.6) | 59.4 (56.7) | 62.1 (59.7) | 80.6 (65.2) |
| ResNet50M + SP-ReID| 33.85 | xent + htri 		| 90.9 (76.6) | 59.9 (57.3) | 61.4 (59.7) | 79.6 (64.9) |
| ResNet50M + SSP-ReID| 64.63 | xent + htri 	| 92.5 (80.1) | 63.1 (60.5) | 65.6 (63.3) | 81.8 (68.6) |
| ResNet50M + SSP-ReID + RR| 64.63 | xent + htri| 93.7 (90.8) | 72.4 (75.0) | 74.6 (77.5) | 86.4 (83.7) |
| DenseNet | 7.72 | xent + htri 				| 89.3 (72.0) | 44.1 (42.2) | 47.4 (45.6) | 79.7 (62.5) |
| DenseNet + S-ReID| 7.91 | xent + htri 		| 89.7 (72.3) | 44.9 (43.1) | 46.7 (44.3) | 80.3 (62.6) |
| DenseNet + SP-ReID| 8.68 | xent + htri 		| 89.6 (72.9) | 44.6 (43.3) | 44.9 (44.3) | 79.8 (62.9) |
| DenseNet + SSP-ReID| 16.59 | xent + htri 		| 90.9 (76.7) | 48.1 (48.1) | 49.1 (49.5) | 82.2 (67.1) |
| DenseNet + SSP-ReID + RR| 16.59 | xent + htri | 93.3 (89.9) | 58.4 (63.1) | 59.9 (64.7) | 86.2 (83.3) |
| InceptionV4 | 42.29 | xent 					| 81.9 (64.0) | 38.7 (38.7) | 42.4 (40.7) | 71.9 (49.6) |
| InceptionV4 + S-ReID| 43.06 | xent 			| 81.4 (62.8) | 43.1 (41.2) | 42.4 (42.5) | 70.6 (49.1) |
| InceptionV4 + SP-ReID| 46.14 | xent 			| 80.6 (62.1) | 35.6 (34.7) | 37.4 (36.1) | 70.6 (49.0) |
| InceptionV4 + SSP-ReID| 89.2 | xent 			| 85.4 (67.7) | 46.4 (45.5) | 45.2 (45.5) | 75.5 (55.0) |
| InceptionV4 + SSP-ReID + RR| 89.2 | xent 		| 86.8 (79.8) | 53.4 (57.0) | 50.6 (54.8) | 79.4 (71.0) |
| Xception | 22.39 | xent 						| 69.9 (50.1) | 26.3 (26.0) | 25.2 (25.2) | 55.4 (36.1) |
| Xception + S-ReID| 22.39 | xent 				| 68.2 (49.8) | 23.4 (23.7) | 24.1 (24.2) | 52.9 (33.6) |
| Xception + SP-ReID| 25.12 | xent 				| 70.9 (47.5) | 22.9 (20.7) | 21.6 (20.8) | 56.6 (34.6) |
| Xception + SSP-ReID| 48.05 | xent 			| 77.1 (57.5) | 29.9 (29.4) | 29.6 (30.0) | 63.9 (42.8) |
| Xception + SSP-ReID + RR| 48.05 | xent 		| 79.3 (70.5) | 79.3 (70.5) | 34.7 (39.2) | 79.3 (70.5) |



## How to prepare data

Create a directory to store reid datasets under this repo via
```bash
cd deep-person-reid/
mkdir data/
```

If you wanna store datasets in another directory, you need to specify `--root path_to_your/data` when running the training code. Please follow the instructions below to prepare each dataset. After that, you can simply do `-d the_dataset` when running the training code. 

Please do not call image dataset when running video reid scripts, otherwise error would occur, and vice versa.

**Market1501** [7]:
1. Download dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html.
2. Extract dataset and rename to `market1501`. The data structure would look like:
```
market1501/
    bounding_box_train/
    bounding_box_test/
		query/
    ...
```
3. Use `-d market1501` when running the training code.

**CUHK03** [13]:
1. Create a folder named `cuhk03-np/` under `data/`.
2. Download dataset to `data/cuhk03-np/` from https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP (we use the version that is in [Market-1501 format](https://goo.gl/bT5f54)). Extract the zip and put the needed directories inside `data/cuhk03-np/`. Finally, the data structure would look like
```
cuhk03-np/
		bounding_box_train/
		bounding_box_test/
		query/
    ...
```
4. Use `-d cuhk03-np` when running the training code. We use new split (767/700)[14], previous protocol is supported but has not been tested. In addition, we support both `labeled` and `detected` modes. The default mode loads `detected` images. Specify `--cuhk03-labeled` if you want to train and test on `labeled` images.


**DukeMTMC-reID** [16, 17]:
1. Create a directory under `data/` called `dukemtmc-reid`.
2. Download dataset `DukeMTMC-reID.zip` from https://github.com/layumi/DukeMTMC-reID_evaluation#download-dataset, extract the zip file and put the needed directories inside `data/dukemtmc-reid`. Finally, the datas structure would look like
```
dukemtmc-reid/
		bounding_box_train/
		bounding_box_test/
		query/ 
```
3. Use `-d dukemtmcreid` when running the training code.


## Acknowledgment
This repo use https://github.com/KaiyangZhou/deep-person-reid as base code, thanks to KaiyangZhou for his great work :).

