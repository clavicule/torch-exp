# torch-exp

### Experimentations with Torch.
To get help, type:
```
th main.lua -help
```

#### Currently options are:
* **-images**   *[required]* EITHER path to directory in which images to load are organized by sub-directories OR existing file containing data previously dumped []
* **-nn**       *[required]* path to the file neural network file []
* **-width**    *[ignored if loading data file]* width in pixel to which images are resized to [32]
* **-height**   *[ignored if loading data file]* height in pixel to which images are resized to [32]
* **-dump**     *[optional]* path to the file in which to dump the ready-to-go images (easier for reloading data) []
* **-test**     *[optional]* if on, neural net uses the data to test, otherwise neural network is trained and dumped to the <nn> file [false]
* **-use_cuda** *[optional]* set to true to use the GPU to train and test the neural network [false]

#### For instance:
```
th main.lua -images /home/clavicule/data -width 199 -height 199 -dump /home/clavicule/mydataset.dat
```
or
````
th main.lua -images /home/clavicule/mydataset.dat
```
