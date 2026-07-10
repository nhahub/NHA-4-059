# Data

## Dataset: ImageNet Truck Subset

The actual dataset used by the pipeline (`CH_Detection_Pipeline.ipynb`) is
stored as pickled examples on Google Drive, loaded directly by Section 0 of
the notebook — this repo does not vendor the raw images.

## Classes and Split

8 classes: `garbage_truck`, `moving_van`, `pickup`, `trailer_truck`,
`police_van`, `fire_engine`, `tow_truck`, `minivan` (see
`src/models/clip_model.py::TRUCK_CLASSES` for the ImageNet class-index
mapping).

- Train: 10,133 images
- Test: 400 images

## Source

Downloaded from HuggingFace: ILSVRC/imagenet-1k, pre-extracted into
`truck_samples_train_FULL.pkl` / `truck_samples_test_FULL.pkl` on Drive.

## Note

An earlier data-collection scheme (1,600 images / 200 per class, including
a `beer_truck` class not used anywhere else in this project) was explored
but never became the dataset the pipeline actually runs on — it's been
removed from this repo to avoid confusion with the numbers above.
