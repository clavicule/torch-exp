require 'torch';
require 'cutorch';
require 'paths';

-- parse arguments
local options = paths.dofile('args.lua')
opt = options.parse(arg)

-- process images and load as tensors
if paths.dirp( opt.images ) then
    paths.dofile( 'data_io.lua' )
    trainset, classes = load_images_and_labels( opt.images, opt.width, opt.height )

-- load directly the tensors from file
elseif paths.filep( opt.images ) then
    print( 'loading data file ' .. opt.images )
    loaded_data = torch.load( opt.images )
    trainset = loaded_data.images
    classes = loaded_data.classes

else
    print( 'invalid input data path or file -- aborting' )
    return
end

-- save tensors to file
if opt.dump ~= nil and opt.dump ~= '' then
    print( 'saving data to ' .. opt.dump )
    data_to_save = {
      images = trainset,
      classes = classes
    }
    torch.save( opt.dump, data_to_save )
end


-- set table index and size function required by Torch NN
setmetatable(trainset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);

function trainset:size()
    return self.data:size(1)
end


print( 'size of the dataset: ')
print( trainset:size() )

print( 'classes are:' )
print( classes )

if opt.nn == nil or opt.nn == '' then
    print( 'nn argument required for training or testing -- aborting')
    return
end

-- load on GPU
if opt.use_cuda then
    print( 'loading dataset on GPU' )
    trainset.data = trainset.data:cuda()
    trainset.label = trainset.label:cuda()
end


paths.dofile( 'cnn.lua' )
if opt.test then
    if not paths.filep( opt.nn ) then
        print( 'no nn file provided -- aborting' )
        return
    end
    -- to do
    print( 'sorry, not implemented yet' )
else
    nn = train( trainset, opt.use_cuda )
    print( 'saving nn to ' .. opt.nn )
    torch.save( opt.nn, nn )
end
