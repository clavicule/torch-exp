# torch-exp

### Experimentations with Torch.
To get help, type:
```
th run.lua -help
```

#### Current options are:
* **-images**     *[required]* EITHER path to directory in which images to load are organized by sub-directories OR existing file containing data previously dumped []
* **-nn**         *[required if training or testing NN]* path to the file neural network file []
* **-width**      *[ignored if loading data file]* width in pixel to which images are resized to [32]
* **-height**     *[ignored if loading data file]* height in pixel to which images are resized to [32]
* **-dump**       *[optional]* path to the file in which to dump the ready-to-go images (easier for reloading data) []
* **-test**       *[optional]* if on, neural net uses the data to test, otherwise neural network is trained and dumped to the <nn> file [false]
* **-use_cuda**   *[optional]* set to true to use the GPU to train and test the neural network [false]


#### Examples on running torch-exp:
##### From Terminal
Process images from folder and dump to file
```
th run.lua -images /home/clavicule/train_data -width 32 -height 32 -dump /home/clavicule/my_training_set.dat
```

Process images from folder and dump to file, then train NN on images
```
th run.lua -images /home/clavicule/train_data -width 32 -height 32 -dump /home/clavicule/my_training_set.dat -nn /home/clavicule/my_net.nn
```

Load a previously dumped image set and train NN on it
```
th run.lua -images /home/clavicule/my_training_set.dat -nn /home/clavicule/my_net.nn
```

Process images from folder, dump it and test it on NN
```
th run.lua -images /home/clavicule/test_data -width 32 -height 32 -nn /home/clavicule/my_net.nn -test -dump /home/clavicule/my_testing_set.dat
```

Load a previously dumped image set and test it on NN
```
th run.lua -images /home/clavicule/my_testing_set.dat -nn /home/clavicule/my_net.nn -test
```

##### From iTorch
Load a previously dumped image set and test it on NN
```
dofile( 'main.lua' )
opt = {}
opt.images = '/home/clavicule/my_testing_set.dat'
opt.nn = '/home/clavicule/my_net.nn'
opt.test = ''
opt.use_cuda = true
main()
```

#### Other functions
Have a look at the helpers lua file
##### In `data_io.lua`

###### `load_images_and_labels( folder_path, width, height )`
* [input: folder_path] expected path should have subfolder
 *   each subfolder represents a category
 *   the name of the subfolder is used as the label for that category
 *   the subfolder is expected to contain only images
 *   grayscale images are ignored (for now)
* [input: width] width to which the images will be resized to
* [input: height] height to which the images will be resize to
* [output: image_set] the format is:
 *   image_set.data: 4D tensor no_img x 3-channels x w x h
 *   image_set.label: 1D tensor containing IDs as integers
* [output: classes] an array of string mapping the categories and the table.label IDs

###### `process_pascal_voc_dataset( path_to_images, path_to_annotations, output_path )`
  call this function to restructure the Pascal VOC dataset folders into expected structure by load_images_and_labels() function.
* [input: path_to_images] path to VOC folder containing the jpg images
* [input: path_to_annotations] path to txt files containing list of images for train, val and both
* [input: output_path] output folder where data will be copied to following the new folder structure
* Note: this function copies images, the original data is not removed nor its original folder structure

###### `crop_pascal_voc_dataset( path_to_xml, path_to_images, output_path )`
  call this function to crop the pascal VOC images by their anootation bounding box
* [input: path_to_xml] path to VOC folder containing the xml annotation files
* [input: path_to_images] path to VOC folder containing the jpg images
* [input: output_path] output folder where data will be copied to following the new folder structure
* Note: this function copies images before cropping, the original data is not removed

###### `split_dataset( path_to_images, output_path, split_ratio )`
  split a dataset in 2 parts: train and test
* [input: path_to_images] path to folder containing the jpg images organized in labelled sub-folders
* [input: output_path] output folder where data will be copied to following the new folder structure (train/{label} and test/{label})
* [input: split_ratio] % of images to copy in the train dataset. The rest is copied to test dataset
