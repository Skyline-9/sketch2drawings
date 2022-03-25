# Sketch2Drawings

Using Conditional Generative Adversial Networks (cGANs), Sketch2drawings performs paired image-to-image translation on sketches and drawings. This deep learning mapping allows the project to turn a black and white sketch into a colorized drawing.

![](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
![Magic](https://img.shields.io/badge/Made%20with-Magic-orange?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyBpZD0iQ2FwYV8xIiBlbmFibGUtYmFja2dyb3VuZD0ibmV3IDAgMCA1MTIgNTEyIiBoZWlnaHQ9IjUxMiIgdmlld0JveD0iMCAwIDUxMiA1MTIiIHdpZHRoPSI1MTIiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGc+PHBhdGggZD0ibTM5NS44MiAxODIuNjE2LTE4OC43MiAxODguNzItMTIuOTEgMS43Mi05LjM1IDIwLjU0LTM0LjMxIDM0LjMxLTExLjAxLS43My0xMS4yNSAyMi45OS01Ni40OCA1Ni40OGMtMi45MyAyLjkzLTYuNzcgNC4zOS0xMC42MSA0LjM5cy03LjY4LTEuNDYtMTAuNjEtNC4zOWwtMjIuNjItMjIuNjJoLS4wMWwtMjIuNjItMjIuNjNjLTUuODYtNS44Ni01Ljg2LTE1LjM2IDAtMjEuMjJsNzcuNjMtNzcuNjMgMTYuNi03LjAzIDUuNjYtMTUuMjMgMzQuMzEtMzQuMzEgMTQuODQtNC45MiA3LjQyLTE3LjM0IDE2Ny41Ny0xNjcuNTcgMzMuMjQgMzMuMjR6IiBmaWxsPSIjZjY2Ii8+PHBhdGggZD0ibTM5NS44MiAxMTYuMTQ2djY2LjQ3bC0xODguNzIgMTg4LjcyLTEyLjkxIDEuNzItOS4zNSAyMC41NC0zNC4zMSAzNC4zMS0xMS4wMS0uNzMtMTEuMjUgMjIuOTktNTYuNDggNTYuNDhjLTIuOTMgMi45My02Ljc3IDQuMzktMTAuNjEgNC4zOXMtNy42OC0xLjQ2LTEwLjYxLTQuMzlsLTIyLjYyLTIyLjYyIDMzNC42NC0zMzQuNjR6IiBmaWxsPSIjZTYyZTZiIi8+PHBhdGggZD0ibTUwNi42MSAyMDkuMDA2LTY5LjE0LTY5LjEzIDQzLjA1LTg4LjM4YzIuOC01Ljc1IDEuNjUtMTIuNjUtMi44OC0xNy4xNy00LjUyLTQuNTMtMTEuNDItNS42OC0xNy4xNy0yLjg4bC04OC4zOCA0My4wNS02OS4xMy02OS4xNGMtNC4zNS00LjM1LTEwLjkyLTUuNi0xNi41Ni0zLjE2LTUuNjUgMi40NS05LjIzIDguMDktOS4wNCAxNC4yNGwyLjg2IDkwLjQ1LTg1LjM3IDU3LjgzYy00LjkxIDMuMzItNy40IDkuMjItNi4zNiAxNS4wNCAxLjA0IDUuODMgNS40IDEwLjUxIDExLjE1IDExLjk0bDk2LjYyIDI0LjAxIDI0LjAxIDk2LjYyYzEuNDMgNS43NSA2LjExIDEwLjExIDExLjk0IDExLjE1Ljg3LjE2IDEuNzUuMjMgMi42Mi4yMyA0LjkyIDAgOS42LTIuNDIgMTIuNDItNi41OWw1Ny44My04NS4zNyA5MC40NSAyLjg2YzYuMTQuMTkgMTEuNzktMy4zOSAxNC4yNC05LjA0IDIuNDQtNS42NCAxLjE5LTEyLjIxLTMuMTYtMTYuNTZ6IiBmaWxsPSIjZmFiZTJjIi8+PHBhdGggZD0ibTI5Ni4yNiAyMTUuNzA2IDI0LjAxIDk2LjYyYzEuNDMgNS43NSA2LjExIDEwLjExIDExLjk0IDExLjE1Ljg3LjE2IDEuNzUuMjMgMi42Mi4yMyA0LjkyIDAgOS42LTIuNDIgMTIuNDItNi41OWw1Ny44My04NS4zNyA5MC40NSAyLjg2YzYuMTQuMTkgMTEuNzktMy4zOSAxNC4yNC05LjA0IDIuNDQtNS42NCAxLjE5LTEyLjIxLTMuMTYtMTYuNTZsLTY5LjE0LTY5LjEzIDQzLjA1LTg4LjM4YzIuOC01Ljc1IDEuNjUtMTIuNjUtMi44OC0xNy4xN3oiIGZpbGw9IiNmZDkwMjUiLz48cGF0aCBkPSJtNDY1IDQxNi45NjZjLTI1LjkyIDAtNDcgMjEuMDgtNDcgNDdzMjEuMDggNDcgNDcgNDcgNDctMjEuMDggNDctNDctMjEuMDgtNDctNDctNDd6IiBmaWxsPSIjZmFiZTJjIi8+PHBhdGggZD0ibTEwNCAyOC45NjZoLTEzdi0xM2MwLTguMjg0LTYuNzE2LTE1LTE1LTE1cy0xNSA2LjcxNi0xNSAxNXYxM2gtMTNjLTguMjg0IDAtMTUgNi43MTYtMTUgMTVzNi43MTYgMTUgMTUgMTVoMTN2MTNjMCA4LjI4NCA2LjcxNiAxNSAxNSAxNXMxNS02LjcxNiAxNS0xNXYtMTNoMTNjOC4yODQgMCAxNS02LjcxNiAxNS0xNXMtNi43MTYtMTUtMTUtMTV6IiBmaWxsPSIjZmVkODQzIi8+PHBhdGggZD0ibTIwNy4xIDM3MS4zMzYtMjIuMjYgMjIuMjYtNDUuMzItODcuNjIgMjIuMjYtMjIuMjZ6IiBmaWxsPSIjZmVkODQzIi8+PHBhdGggZD0ibTE4NC44NCAzOTMuNTk2IDIyLjI2LTIyLjI2LTIyLjY2LTQzLjgxLTIyLjI2NSAyMi4yNjV6IiBmaWxsPSIjZmFiZTJjIi8+PHBhdGggZD0ibTE1MC41MyA0MjcuOTA2LTIyLjI2IDIyLjI2LTQ1LjMyLTg3LjYyIDIyLjI2LTIyLjI2eiIgZmlsbD0iI2ZlZDg0MyIvPjxwYXRoIGQ9Im0xMjguMjcgNDUwLjE2NiAyMi4yNi0yMi4yNi0yMi42NTUtNDMuODE1LTIyLjI2IDIyLjI2eiIgZmlsbD0iI2ZhYmUyYyIvPjxjaXJjbGUgY3g9IjE1IiBjeT0iMTE5Ljk2OSIgZmlsbD0iIzVlZDhkMyIgcj0iMTUiLz48Y2lyY2xlIGN4PSIxMjgiIGN5PSIxOTkuOTY5IiBmaWxsPSIjZDU5OWVkIiByPSIxNSIvPjxjaXJjbGUgY3g9IjE5MiIgY3k9IjYzLjk2NCIgZmlsbD0iI2Y2NiIgcj0iMTUiLz48Y2lyY2xlIGN4PSIzMjgiIGN5PSI0MTUuOTY3IiBmaWxsPSIjMzFiZWJlIiByPSIxNSIvPjxjaXJjbGUgY3g9IjQ0MCIgY3k9IjMyNy45NjciIGZpbGw9IiNhZDc3ZTMiIHI9IjE0Ljk5OSIvPjwvZz48L3N2Zz4=)

<details>
      <summary>Table of Contents</summary>
      
- [Sketch2Drawings](#sketch2drawings)
  - [Samples](#samples)
  - [How to train Sketch2Drawings??](#how-to-train-sketch2drawings)
      - [Big Overview Steps](#big-overview-steps)
    - [After Downloading Data](#after-downloading-data)
    - [Install dependencies](#install-dependencies)
    - [Resizing Operation](#resizing-operation)
    - [Detect Edges Using Canny](#detect-edges-using-canny)
    - [Combine Operation](#combine-operation)
    - [Split Operation](#split-operation)
    - [Training](#training)
    - [Testing](#testing)
    - [Export Mode](#export-mode)
    - [Port model to Tensorflow.js](#port-model-to-tensorflowjs)
  - [Testing With Different Dataset](#testing-with-different-dataset)
</details>

---

## Samples

![](sample-img/old_model/example7.png)
![](sample-img/old_model/example11.png)
![](sample-img/new_model/example1.png)
![](sample-img/new_model/example2.png)
![](sample-img/new_model/example5.png)
_Left is input, center is output, and right is target_

## How to train Sketch2Drawings??

#### Big Overview Steps

1. Download data from [this kaggle dataset](https://www.kaggle.com/shanmukh05/anime-names-and-image-generation)
2. Prepare/Preprocess the data
3. Train the model
4. Test the model
5. Export the model??

<hr>

### After Downloading Data

Create a folder called original and edges under images so that the directory looks like this

```
.
├── README.md
├── docker
│        └── Dockerfile
├── images
│        └── original
│        └── edges
│        └── resized
│        └── combined
```

Move the images (only, no folders) to images/original

We will

1. Resize all images into 256 x 256
2. Detect edges to get the "sketch"
3. Combine input images and target images
4. Split combined images into train and val set

### Install dependencies

Install all the dependencies

```bash
pip3 install -r requirements.txt
```

Warning: Python version must be at max 3.6! I spent too much time trying to do with Python 3.8 `D:`

If you have conda installed, you can also try this

1. Create virtual environment named sketch2drawings

```bash
conda create -n "sketch2drawings" python=3.6.0
```

2. Activate conda environment

```bash
conda activate sketch2drawings
```

3. Install OpenCV and Tensorflow v1.4.1 (since numpy is already installed)

```bash
conda install opencv-python
pip install tensorflow==1.4.1
```

### Resizing Operation

After putting all the images into the original folder, run

```bash
python preprocessing/process.py --input_dir images/original --operation resize --output_dir images/resized
```

This operation took me about 25 minutes to run. When finished, we should see a folder called resize with 256 x 256 images

### Detect Edges Using Canny

We used Canny to detect edges. Navigate to the folder with process in it and then run process.py. This cd is important because it's used to figure out where the image folder is located.

```bash
cd preprocessing
python edge_detection.py
```

### Combine Operation

Navigate back to the root directory with `cd ..`. Now run the combine operation with

```bash
python preprocessing/process.py --input_dir images/resized --b_dir images/edges --operation combine --output_dir images/combined
```

This operation took me about 30 minutes to run. The script will skip over files that already exist, so you can pause the operation and resume later.

### Split Operation

Generate train/validation splits

```bash
python preprocessing/split.py --dir images/combined
```

### Training

Hopefully you have a GPU because if you train on CPU you will definitely be waiting for a bit.

```bash
python pix2pix.py --mode train --output_dir s2d_train --max_epochs 200 --input_dir images/combined/train --which_direction BtoA --ngf 32 --ndf 32
```

Maybe try changing `--ngf 32` and `--ndf32` to 64 to see how well it does, but it takes more computation

If you have Docker installed, you can use the Docker image to train without having to install the correct version of Tensorflow

```bash
# Train the model with docker
python dockrun.py python pix2pix.py \
      --mode train \
      --output_dir s2d_train \
      --max_epochs 200 \
      --input_dir images/combined/train \
      --which_direction BtoA
```

### Testing

```bash
python pix2pix.py --mode test --output_dir s2d_test --input_dir images/combined/val --checkpoint s2d_train
```

Results shouldbe in a new folder called `s2d_test`

### Export Mode

```bash
python pix2pix.py --mode export --output_dir export/ --checkpoint s2d_train/ --which_direction BtoA
```

After running this command, you should see a folder called `export` with `checkpoint` and various files inside it.

### Port model to Tensorflow.js

```bash
python3 export_checkpoint.py --checkpoint export --output_file s2d.pict
```

This will create a file called `s2d.pict`.

## Other Notes
I also tested this with the anime-sketch-colorization-pair dataset and it works well for some sketches (Canny tends to not give very clear edges). However, for some reason, it doesn't work as well when I port it to Tensorflow.js, which I'm not sure why
