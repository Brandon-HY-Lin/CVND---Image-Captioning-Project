[dataset]: https://raw.githubusercontent.com/Brandon-HY-Lin/CVND---Image-Captioning-Project/985e1b624ffa27007b01ebec9f544701046d06a6/images/coco-examples.jpg "CoCo 2014"

[dataset_sample_image]: https://github.com/Brandon-HY-Lin/CVND---Image-Captioning-Project/blob/master/images/dataset_sample_1.png "dateset_sample_image"

[image_caption_paper]: https://arxiv.org/pdf/1411.4555.pdf "Paper- Show and Tell: A Neural Image Caption Generator"

[diagram_cnn_rnn_model]: https://raw.githubusercontent.com/Brandon-HY-Lin/CVND---Image-Captioning-Project/985e1b624ffa27007b01ebec9f544701046d06a6/images/encoder-decoder.png "Diagram of CNN-RNN model"

[diagram_cnn_encoder]: https://raw.githubusercontent.com/Brandon-HY-Lin/CVND---Image-Captioning-Project/985e1b624ffa27007b01ebec9f544701046d06a6/images/encoder.png "Diagram of encoder"

[diagram_rnn_decoder]: https://raw.githubusercontent.com/Brandon-HY-Lin/CVND---Image-Captioning-Project/985e1b624ffa27007b01ebec9f544701046d06a6/images/decoder.png "Diagram of decoder"

# Project: Image Captioning

# Abstract
This work adopts methods in [Vinyals's work][image_caption_paper], which utilizes CNN-RNN model to train [CoCo 2014 dataset](http://cocodataset.org/#download). Due to computation power, only 3 epochs are executed and achieves a training loss of 1.93. The BLEU-4 score of validation dataset is 0.517.


# Introduction
Before diving into the algorithm, let's look more closly on the dateset, the Microsoft Common Objects in COntext (MS COCO) dataset. It is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.

![Dataset of MS COCO][dataset]


The figure and its captions as shown below are a sample of MS COCO.

* Sample Image

![Sample image of MS COCO dataset][dataset_sample_image]

* Captions
    * A living room is decorated with floral print.
    * A living room filled with chairs, a fireplace, and television. 
    * A living area has heavily patterned walls, furniture, and rug.
    * A living room with chairs, a piano, and a fireplace.
    * A living room with floral patterned walls and chairs.


# CNN-RNN Model
The CNN-RNN model is a encoder-decoder architecture. First, let's talk about CNN part which is served as encoder and extractes features. The diagram of encoder is shown below.

![Diagram of encoder][diagram_cnn_encoder]

_Diagram of CNN part (encoder)_


The RNN part is served as decoder, which is a language model and generates word-level tokens. 

![Diagram of decoder][diagram_rnn_decoder]

_Diagram of RNN part (decoder)_


After introducing both CNN and RNN part, let's put them together. The diagram of CNN-RNN model is shown below.

![Diagram of CNN-RNN model][diagram_cnn_rnn_model]

_Diagram of CNN RNN model_


# Implementation
I use a pretrained ResNet50 and remove FC layer in the last layer. The reason why I choose pretrained ResNet50 is that the size of training dataset is 82,783, which is small. Beside RestNet50, other parameters are trainable. Then the ouput of this CNN is feed into the RNN. In the paper Vinyals et al., 2015, authors found that feeding the encoded features into every time step makes it overfit easily. As a result, I only feed image as first time step. The dimensions of embeddeding and hidden are both 512, which is adopted from the Vinyals' work. However, I do not add dropout layer because I only train few epochs and it's most likely underfitting than overfitting.

To fit image into pretrained ResNet50, 2 tranformers are used to resize the images to (255x255) and then random crop resized images into (224x224). The random cropping generates more images and MIGHT increase the accuracy which is a plus. The code snippet is shown below:
```
transforms.Compose([ 
                    transforms.Resize(256),                          # smaller edge of image resized to 256
                    transforms.RandomCrop(224),                      # get 224x224 crop from random location
                    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
                    transforms.ToTensor(),                           # convert the PIL Image to a tensor,
                    ...,
                    ])
```

After getting image from the previous 2 transformer, the cropped images are normalized. as shown below.
```
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                        (0.229, 0.224, 0.225))
```
The magic number in the Noralization() is the sampled mean and standard deviation in the ImageNet. Note that, the ResNet adopted the batch normalization in order to alleviate internal covariance shift. Since the pretrained model has adapted to the pixel intensity in ImageNet, it's better to normalize CoCo-images first.


# Results
The number of test images is 40,504. The training time for 3 epochs is 8 hours. The training loss is 1.93 and BLEU-4 of testing dataset is 0.517.


# Conclusion
In this work, CNN-RNN model is implemented and it achieves BLEU-4 score of 0.517.


# Future Works
Visualize attention by implementing [Xu's work](https://arxiv.org/pdf/1502.03044.pdf).


# Appendix
#### Hyper-Parameters

* Encoder
	* CNN
		* pretrained ResNet50 provided by pytorch
            * input size: (224, 224, 3)
        * Linear:
            * output size: 256
* Decoder
    * Embedding:
        * input size: 9955    (vocabulary size)
        * output size: 512
	* LSTM
		* #layer: 1
        * input size: 512
	    * hidden size: 512
	* Fully-connected layer
		* layer: 1
		* input size: 512
		* output size: 9955   (vocabulary size)