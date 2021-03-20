## CRAFT: Character-Region Awareness For Text detection
Official Pytorch implementation of CRAFT text detector | [Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) | [Supplementary](https://youtu.be/HI8MzpY8KMI)

**[Youngmin Baek](mailto:youngmin.baek@navercorp.com), Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee.**

Clova AI Research, NAVER Corp.

**Packaged by [Vinh Quang Tran](mailto:starcraft6723@hotmail.com)**

**Based on [Ashish Jha's implementation](https://github.com/arj7192/CRAFT-pytorch). Modified for use with Windows and CUDA support.**

### Sample Results

### Overview
PyTorch implementation for CRAFT text detector that effectively detect text area by exploring each character region and affinity between characters. The bounding box of texts are obtained by simply finding minimum bounding rectangles on binary map after thresholding character region and affinity scores. 

<img width="1000" alt="teaser" src="./figures/craft_example.gif">

## Updates
**13 Jun, 2019**: Initial update

**20 Jul, 2019**: Added post-processing for polygon result

**28 Sep, 2019**: Added the trained model on IC15 and the link refiner

**25 Jan, 2020**: Put it together as a PyPI package

**20 Mar, 2021**: Modified for use with Windows and CUDA support.

Changes from [Ashish Jha's implementation](https://github.com/arj7192/CRAFT-pytorch):
* CRAFT will now run on CUDA GPU if available.
* Changed model downloading directory to working directory for compatibility with Windows.
* Removed required packages' version from `requirements.txt`.

## Getting started

### Use it straight from PyPI
#### Installation
```
git clone https://github.com/vinhtq115/CRAFT-pytorch
cd CRAFT-pytorch
pip install -r requirements.txt
pip install -e .
```
#### Usage
```
import craft
import cv2

img = cv2.imread('/path/to/image/file')

# run the detector
bboxes, polys, heatmap = craft.detect_text(img)

# view the image with bounding boxes
img_boxed = craft.show_bounding_boxes(img, bboxes)
cv2.imshow('fig', img_boxed)

# view detection heatmap
cv2.imshow('fig', heatmap)
```

### Use from source - install dependencies
#### Requirements
- PyTorch
- torchvision
- opencv-python
- check requirements.txt
```
pip install -r requirements.txt
```

### Training
The code for training is not included in this repository, and we cannot release the full training code for IP reason.


### Arguments for detect_text
* `--text_threshold`: text confidence threshold
* `--low_text`: text low-bound score
* `--link_threshold`: link confidence threshold
* `--canvas_size`: max image size for inference
* `--mag_ratio`: image magnification ratio
* `--refine`: use link refiner for sentence-level dataset
* `--refiner_model`: pretrained refiner model


## Links
- WebDemo : https://demo.ocr.clova.ai/
- Repo of recognition : https://github.com/clovaai/deep-text-recognition-benchmark

## Citation
```
@inproceedings{baek2019character,
  title={Character Region Awareness for Text Detection},
  author={Baek, Youngmin and Lee, Bado and Han, Dongyoon and Yun, Sangdoo and Lee, Hwalsuk},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9365--9374},
  year={2019}
}
```

## License
```
Copyright (c) 2019-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
