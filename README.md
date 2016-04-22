# torch-exp

Experimentations with Torch. To get help, type:
`th main.lua -help`

Currently options are:
  -images [required] EITHER path to directory in which images to load are organized by sub-directories OR existing file containing data previously dumped []
  -width  [ignored if loading data file] width in pixel to which images are resized to [231]
  -height [ignored if loading data file] height in pixel to which images are resized to [231]
  -dump   [optional] path to file in which to dump the loaded images (easier for reloading data) []

For instance:
`th main.lua -images /home/clavicule/data -width 199 -height 199 -dump /home/clavicule/mydataset.dat`
or
`th main.lua -images /home/clavicule/mydataset.dat`
