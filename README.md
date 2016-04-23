# torch-exp

### Experimentations with Torch.
To get help, type:
```
th main.lua -help
```

#### Current options are:
* **-images**     *[required]* EITHER path to directory in which images to load are organized by sub-directories OR existing file containing data previously dumped []
* **-nn**         *[required if training or testing NN]* path to the file neural network file []
* **-width**      *[ignored if loading data file]* width in pixel to which images are resized to [32]
* **-height**     *[ignored if loading data file]* height in pixel to which images are resized to [32]
* **-dump**       *[optional]* path to the file in which to dump the ready-to-go images (easier for reloading data) []
* **-test**       *[optional]* if on, neural net uses the data to test, otherwise neural network is trained and dumped to the <nn> file [false]
* **-use_cuda**   *[optional]* set to true to use the GPU to train and test the neural network [false]


#### Examples:
Process images from folder and dump to file
```
th main.lua -images /home/clavicule/train_data -width 32 -height 32 -dump /home/clavicule/my_training_set.dat
```

Process images from folder and dump to file, then train NN on images
```
th main.lua -images /home/clavicule/train_data -width 32 -height 32 -dump /home/clavicule/my_training_set.dat -nn /home/clavicule/my_net.nn
```

Load a previously dumped image set and train NN on it
```
th main.lua -images /home/clavicule/my_training_set.dat -nn /home/clavicule/my_net.nn
```

Process images from folder, dump it and test it on NN
```
th main.lua -images /home/clavicule/test_data -width 32 -height 32 -nn /home/clavicule/my_net.nn -test -dump /home/clavicule/my_testing_set.dat
```

Load a previously dumped image set and test it on NN
```
th main.lua -images /home/clavicule/my_testing_set.dat -nn /home/clavicule/my_net.nn -test
```
