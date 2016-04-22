require 'torch';
require 'paths';

local options = paths.dofile('args.lua')
opt = options.parse(arg)

paths.dofile( 'data_io.lua' )

trainset, classes = load_images_and_labels( opt.image_path, opt.width, opt.height )

print( 'size of the image dataset: ')
print( trainset.data:size() )

print( 'classes are:' )
print( classes )
