require 'torch';
require 'paths';

local options = paths.dofile('args.lua')
opt = options.parse(arg)

if paths.dirp( opt.images ) then
    paths.dofile( 'data_io.lua' )
    trainset, classes = load_images_and_labels( opt.images, opt.width, opt.height )

elseif paths.filep( opt.images ) then
    print( 'loading data file ' .. opt.images )
    loaded_data = torch.load( opt.images )
    trainset = loaded_data.images
    classes = loaded_data.classes

else
    print( 'invalid input data path or file -- aborting' )
    return
end

if opt.dump ~= nil and opt.dump ~= '' then
    print( 'saving data to ' .. opt.dump )
    data_to_save = {
      images = trainset,
      classes = classes
    }
    torch.save( opt.dump, data_to_save )
end

print( 'size of the image dataset: ')
print( trainset.data:size() )

print( 'classes are:' )
print( classes )
