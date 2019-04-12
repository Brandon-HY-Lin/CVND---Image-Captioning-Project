[dataset]: https://raw.githubusercontent.com/Brandon-HY-Lin/CVND---Image-Captioning-Project/985e1b624ffa27007b01ebec9f544701046d06a6/images/coco-examples.jpg "CoCo 2014"

[image_caption_paper]: https://arxiv.org/pdf/1411.4555.pdf "Paper- Show and Tell: A Neural Image Caption Generator"

# Abstract
This work adopts methods in [Vinyals's work][image_caption_paper]. The architecture combines CNN and RNN to train [CoCo 2014 dataset](http://cocodataset.org/#download). Due to computation power, only 3 epochs are executed and achieves a training loss of 1.8. The BLEU-4 score of validation dataset is 0.517.


# Introduction
Before diving into the algorithm, let's look more closly on the dateset, the Microsoft Common Objects in COntext (MS COCO) dataset. It is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.

![Dataset of MS COCO][dataset]


The figure and its captions as shown below are a sample of MS COCO.



A living room is decorated with floral print.
A living room filled with chairs, a fireplace, and television. 
A living area has heavily patterned walls, furniture, and rug.
A living room with chairs, a piano, and a fireplace.
A living room with floral patterned walls and chairs.





The first step is __feature extraction__. The output of 1st step could be [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) or [spectrogram](https://en.wikipedia.org/wiki/Spectrogram). The second step is __acoustic model__ is to fullfill phonetic representation which could be CNN or RNN. The 3rd step is __decoder__ which is to fullfill language model. It could be fully-connected layer or RNN. In this work, [deep sppech 2][deep_speech_2] is implemented and its result is compared with other methods.

In this project, the input is spectrogram with size of 161. The output size is 29 which includes 26 characters, space character (" "), an apostrophe ('), and an empty "character" used to pad training examples within batches containing uneven lengths.


# Deep Speech 2
The flow chart of [deep speech 2][deep_speech_2] is shown below. The input is spectrogram and then fed into 2-layer 2d-CNN. The result is passed into 1-layer RNN and 1 fully-connected layer. Each output of layer has batch-norm except final output.


![Flow chart of deep speech 2][flow_deep_speech_2] 

*Flow chart of __deep speech 2__*


# Implementation

* Architecture:
The architecture is (1-layer Conv2D + 2-layer Bidirectional GRU + TimeDistributed Dense). To reducing overfitting, batch-normalization and dropout layers are added between each layer. Futhurmore, the dropout feature in GRU cell is also enabled.

* Steps:
    * Input: The spectrogram with shape (None, TimeSeqLength, 161, 1).
    * Conv2D: Served as encoder to extract features.
            Stride=2, Out-channel=8, padding=same
            output shape = (None, TimeSeqLength/2, 81, 8)
    * BatchNorm: Speed-up traininig at first few epochs.
    * Dropout: Reduce strong overfitting caused by Conv2D.
    * Lambda: reshape the 4-dim data to 3-dim data in order to fit into GRU layer.
            output shape = (None, TimeSeqLength/2, 81 * 8)
    * Bidirectional GRU with dropout (layer-0) : served as decoder.
            output shape = (None, TimeSeqLength/2, 512)
    * BatchNorm: Speed-up training
    * Dropout: Reduce overfitting of 2-layer GRU
    * Bidirectional GRU with dropout (layer-1) : served as decoder.
            output shape = (None, TimeSeqLength/2, 512)
    * BatchNorm: Speed-up training
    * Dropout: Reduce overfitting of 2-layer GRU
    * TimeDistributed Dense: convert time series to characters.
            output shape = (None, TimeSeqLength/2, 29)


# Results

The model name of __deep speech 2__ is _model_cnn2d_dropout_, and the validation loss is 93.7 which is better than other combinations of CNN, RNN, and FC. The detail results are shown below.


| Model               	| Architecture                                	| Training loss @ Epoch-20 	| Validation Loss @ Epoch-20 	| Lowest Validation Loss 	| Epoch of lowest Valid Loss 	|
|---------------------	|---------------------------------------------	|--------------------------	|----------------------------	|------------------------	|----------------------------	|
| model_0             	| RNN                                         	| 778.4                    	| 752.1                      	| 752.2                  	| 20                         	|
| model_1             	| RNN + Dense                                 	| 119.1                    	| 137.3                      	| 137.3                  	| 18                         	|
| model_2             	| CNN-1D + RNN + Dense                        	| 73.4                     	| 149.8                      	| 134.5                  	| 8                          	|
| model_3             	| 3-layer RNN + Dense                         	| 93.1                     	| 134.5                      	| 132.7                  	| 16                         	|
| model_4             	| Bidirectional RNN + Dense                   	| 102.1                    	| 135.7                      	| 134.3                  	| 17                         	|
| model_cnn2d         	| CNN-2D + RNN + Dense                        	| 76.7                     	| 140.7                      	| 128.3                  	| 9                          	|
| model_deep_cnn2d    	| 2-layer CNN-2D + RNN + Dense                	| 66.5                     	| 191.9                      	| 151.1                  	| 6                          	|
| model_cnn2d_dropout 	| CNN2D + 2-layer Bidirectional GRU + Dropout 	| 75.7                     	| 93.7                       	| 93.7                   	| 20                         	|


# Conclusion
In this work, the similar architecture in [deep sppech 2][deep_speech_2] is implemented and achieve validation loss of 93.7 which is much better compared to other combinations of RNN and CNN.


# Future Works
Visualize attention by implementing [Xu's work](https://arxiv.org/pdf/1502.03044.pdf).


# Appendix
#### Hyper-Parameters

* model_cnn2d_dropout
	* CNN
		* #layer: 1
	    * #input width: 161
	    * #input channel: 1
	    * #output channel: 8
	    * kernel size: 11
	    * stride: 2
	    * padding: same
	    * dropout: 0.2
	* Bidirectional GRU
		* #layer: 2
	    * unit of neurons: 256
	    * dropout: 0.2
	* Fully-connected layer
		* layer: 1
		* input size: 256
		* output size: 29