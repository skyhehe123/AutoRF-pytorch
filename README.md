# AutoRF (unofficial)
This is unofficial implementation of "AutoRF: Learning 3D Object Radiance Fields from Single View Observations", which performs implicit neural reconstruction, manipulation and scene composition for 3D object. In this repo, we use KITTI dataset.

<img src="output/000008.png" alt="drawing" width="500"/>
<img src="output/scene.gif" alt="drawing" width="500"/>
<img src="output/scene_rotate.gif" alt="drawing" width="500"/>


<details>
  <summary> Dependencies (click to expand) </summary>
  
  ## Dependencies
  - pytorch==1.10.1
  - matplotlib
  - numpy
  - imageio
</details>

## Quick Start

Download KITTI data and here we only use image data
```plain
└── DATA_DIR
       ├── training   <-- training data
       |   ├── image_2
       |   ├── label_2
       |   ├── calib
```     
Run the preprocess scripts, which produce instance mask using pretrained PointRend model.      
```
python scripts/preproc.py
```
After this, you will have a certain directory which contains the image, mask and 3D anotation of each instance.
```plain
└── DATA_DIR
       ├── training
       |   ├── nerf
           |   ├── 0000008_01_patch.png
           |   ├── 0000008_01_mask.png
           |   ├── 0000008_01_label.png
```

Run the following sciprts to train a nerf model

```
python src/train.py
```

After training for serveral iterations (enough is ok), you can find the checkpoint file in the ``output'' folder, and then you can perform scene rendering by running

```
python src/train.py --demo
```


## Notice ###

You can adjust the manipulaion function (in kitti.py) by your self, here I only provide the camera pushing/pulling and instance rotation.


