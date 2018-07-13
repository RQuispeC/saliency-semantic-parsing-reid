TODO:
* add bibtex cite and link to paper
* review documentation before changing repo to public

# Enhanced Person Re-Identification Based on Saliency and Semantic Parsing with Deep Neural Network Models

This repo implements our paper [*Enhanced Person Re-Identification Based on Saliency and Semantic Parsing with Deep Neural Network Models*]() and it has been released for academic pourposes only, please cite if it is useful in your research

```
BIBTEX FORMAT CITE
```

This repo explains dependencies, data preparation, training and testing for every module of our network:

![proposed framework](framework.png)

## Dependecies
The implementation has been tested under a Linux enviroment with:

* Python 2
* PyTorch (V0.4.0)
* Numpy (V1.14.3)
* OpenCV (V3.4.0)

Other library versions and OS may need small changes.

## Prepare data

Create a directory to store reid datasets under this repo via

```bash
cd deep-person-reid/
mkdir data/
```

Please follow the instructions below to prepare each dataset. After that, you can simply do `-d the_dataset` when running the training code. 

**Market1501** [8]:
1. Download dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html.
2. Extract dataset and rename to `market1501`. 
3. Download the precomputed saliency and semantic parsing maps from [here](https://drive.google.com/file/d/1lKxZOsdh2F3jOHeebO3JExQLyqSbecKi/view?usp=sharing), extract the zip under the directory `data/market1501`. Finally,  the data structure would look like:

```
market1501/
	bounding_box_train/
	bounding_box_test/
	query/
	salience/
		bounding_box_train/
		bounding_box_test/
		query/
	parsing/
		bounding_box_train/
		bounding_box_test/
		query/
    ...
```

4. Use `-d market1501` when running the training code.

**CUHK03** [9]:
1. Create a folder named `cuhk03-np/` under `data/`.
2. Download dataset to `data/cuhk03-np/` from https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP (we use the version that is in [Market-1501 format](https://goo.gl/bT5f54)). Extract the zip and put the needed directories inside `data/cuhk03-np/`. 
3. Download the precomputed saliency and semantic parsing maps from [here](https://drive.google.com/file/d/1EJY1HewqTN362JrxCPpUkFCVkNpdprT3/view?usp=sharing), extract the zip under the directory `data/cuhk03-np`. Finally, the data structure would look like
```
cuhk03-np/
	detected/
		bounding_box_train/
		bounding_box_test/
		query/
	labeled/
		bounding_box_train/
		bounding_box_test/
		query/
	salience/
		detected/
			bounding_box_train/
			bounding_box_test/
			query/
		labeled/
			bounding_box_train/
			bounding_box_test/
			query/
	parsing/
		detected/
			bounding_box_train/
			bounding_box_test/
			query/
		labeled/
			bounding_box_train/
			bounding_box_test/
			query/
    ...
```
4. Use `-d cuhk03-np` when running the training code. We use new split (767/700)[14], previous protocol is supported but has not been tested. In addition, we support both `labeled` and `detected` modes. The default mode loads `detected` images. Specify `--cuhk03-labeled` if you want to train and test on `labeled` images.

**DukeMTMC-reID** [10, 11]:
1. Create a directory under `data/` called `dukemtmc-reid`.
2. Download dataset `DukeMTMC-reID.zip` from https://github.com/layumi/DukeMTMC-reID_evaluation#download-dataset, extract the zip file and put the needed directories inside `data/dukemtmc-reid`. 
3. Download the precomputed saliency and semantic parsing maps from [here](https://drive.google.com/file/d/14fk9lOQzcJaJBCTsLmXaAcwHkh6LLjwu/view?usp=sharing), extract the zip under the directory `data/dukemtmc-reid`. Finally, the datas structure would look like
```
dukemtmc-reid/
	bounding_box_train/
	bounding_box_test/
	query/
	salience/
		bounding_box_train/
		bounding_box_test/
		query/
	parsing/
		bounding_box_train/
		bounding_box_test/
		query/
```
4. Use `-d dukemtmcreid` when running the training code.

In case you want to generate saliency and semantic parsing maps for other datasets, we used the methods proposed by [1] and [2].

## Train

For training we have two scripts: `train_img_model_xent.py` for training with cross entropy and `train_img_model_xent_htri.py` for training with cross entropy and triplet loss.

We use the parameter `-a` to specify the architecture, we support:

* ResNet[3]: Use `-a resnet50`
* ResNet-M[4]: Use `-a resnet50m`
* DenseNet[5]: use `-a densenet121`
* Inception-V4[6]: Use `-a inceptionv4`
* Xception[7]: Use `-a xception`

To train the network using backbones with salience use the sufix `-salience` in the network name (e.g. `-a resnet-salience`)

To train the network using backbones with semantic parsing use the sufix `-parsing` in the network name (e.g. `-a resnet-parsing`)

Note that the complete framework needs to train its `-salience` and `-parsing` subnets independently. For instance, to train ResNet50 combined with salience maps over Market1501 dataset run:

```bash

python train_imgreid_xent_htri.py -d market1501 -a resnet50-salience --optim adam --lr 0.0003 --max-epoch 60 --stepsize 20 --train-batch 32 --test-batch 100 --save-dir log/resnet50-xent-market1501 --gpu-devices 0

```

To use multiple GPUs, you can set `--gpu-devices 0,1,2,3`.

For full list of arguments use `python train_blah_blah.py -h`

Note: we make available all our [trained models](https://drive.google.com/drive/folders/1BoDNtlqhtUFIAKychTou-qesL_lHshLy?usp=sharing)

## Test

We have two different ways to test our network, one for individual subnets and the other for the complete framework. For instance to train an individual network: let's suppose that you already trained a ResNet50 with saliency maps for Market1501 and have its results in `saved-models/resnet-salience.pth.tar`, then to evaluate it run:

```bash
python train_imgreid_xent_htri.py -d market1501 -a resnet50-salience --resume saved-models/resnet-salience.pth.tar --evaluate --save-dir log/test-resnet-salience-market1501
```

In orther to test the complete framework you need to first have to train its saliency and semantic parsing subnets, let's suppose that you already train ResNet with Market1501 and have their results in `saved-models/resnet-salience.pth.tar` and `saved-models/resnet-parsing.pth.tar` then run:

``` bash
python test_two_nets_img.py -a1 resnet50-salience -a2 resnet50-parsing --resume1 saved-models/resnet-salience.pth.tar --resume2 saved-models/resnet-parsing.pth.tar --save-dir log/test-resnet-salience-parsing-market1501
```

It is important that `-a1` specifies a architecture that uses salience and `-a2` an architecure that uses semantic parsing, otherwise, code will give an error.
## Results
All models are initialized with imagenet pretrained weights.
Results are presented using the format of `Rank1 (mAP)` in the following table

CUHK03 uses the [new protocol](https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP#State-of-the-art) (767/700)

| Model | # param (M) | Loss | Market1501 | CUHK03 - Detected | CUHK03 - Labeled| DukeMTMC-reID |
| :---  |     :---:   | :---:|  :---: 	  |      :---:	      | 	  :---: 	| 	:---:       |
| ResNet50 | 25.04 | xent + htri 				| 88.1 (72.9) | 55.6 (52.9) | 58.8 (56.7) | 77.7 (62.1) |
| ResNet50 + S-ReID| 25.81 | xent + htri 		| 87.6 (73.0) | 56.0 (53.4) | 55.9 (54.4) | 78.9 (63.1) |
| ResNet50 + SP-ReID| 28.89 | xent + htri 		| 87.8 (72.4) | 56.2 (53.5) | 57.3 (55.5) | 78.0 (62.7) |
| ResNet50 + SSP-ReID| 54.7 | xent + htri 		| 89.3 (75.9) | 59.4 (57.1) | 60.6 (58.9) | 80.1 (66.1) |
| ResNet50 + SSP-ReID + RR| 54.7 | xent + htri 	| 91.5 (88.2) | 67.6 (71.1) | 68.4 (72.4) | 84.8 (81.4) |
| ResNet-M | 30.01 | xent + htri 				| 91.2 (77.5) | 58.7 (56.3) | 61.1 (58.9) | 78.8 (63.5) |
| ResNet-M + S-ReID| 30.78 | xent + htri 		| 91.2 (77.6) | 59.4 (56.7) | 62.1 (59.7) | 80.6 (65.2) |
| ResNet-M + SP-ReID| 33.85 | xent + htri 		| 90.9 (76.6) | 59.9 (57.3) | 61.4 (59.7) | 79.6 (64.9) |
| ResNet-M + SSP-ReID| 64.63 | xent + htri 	| 92.5 (80.1) | 63.1 (60.5) | 65.6 (63.3) | 81.8 (68.6) |
| ResNet-M + SSP-ReID + RR| 64.63 | xent + htri| 93.7 (90.8) | 72.4 (75.0) | 74.6 (77.5) | 86.4 (83.7) |
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


## Acknowledgment
This repo use https://github.com/KaiyangZhou/deep-person-reid as base code, thanks to KaiyangZhou for his great work :).

## References
[1] [Deepsaliency: Multi-task deep neural network model for salient object detection, IEEE Transactions on Image Processing](https://arxiv.org/pdf/1510.05484.pdf) [[code](http://www.zhaoliming.net/research/deepsaliency)] <br />
[2] [Look Into Person: Self-Supervised Structure-Sensitive Learning and a New Benchmark for Human Parsing, CVPR 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Gong_Look_Into_Person_CVPR_2017_paper.pdf) [[code](https://github.com/Engineering-Course/LIP_SSL)] <br />
[3] [He et al. Deep Residual Learning for Image Recognition. CVPR 2016.](https://arxiv.org/abs/1512.03385)<br />
[4] [Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching. arXiv:1711.08106.](https://arxiv.org/abs/1711.08106) <br />
[5] [Huang et al. Densely Connected Convolutional Networks. CVPR 2017.](https://arxiv.org/abs/1608.06993) <br />
[6] [Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.](https://arxiv.org/abs/1512.00567) <br />
[7] [Chollet. Xception: Deep Learning with Depthwise Separable Convolutions. CVPR 2017.](https://arxiv.org/abs/1610.02357) <br />
[8] [Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) <br />
[9] [Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf) <br />
[10] [Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.](https://arxiv.org/abs/1609.01775) <br />
[11] [Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.](https://arxiv.org/abs/1701.07717) <br />
