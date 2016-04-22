require 'torch';
require 'paths';
require 'image';
require 'xlua';

-- returns size of string array
function size_of(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

-- [input: folder_path] expected path should have subfolder
--    each subfolder represents a category
--    the name of the subfolder is used as the label for that category
--    the subfolder is expected to contain only images
--    grayscale images are ignored (for now)
-- [input: width] width to which the images will be resized to
-- [input: height] height to which the images will be resize to
-- [output: image_set] the format is:
--   ** image_set.data: 4D tensor no_img x 3-channels x w x h
--   ** image_set.label: 1D tensor containing IDs as integers
-- [output: classes] an array of string mapping the categories and the table.label IDs
-- note: the table indexing function is already setup to be compatible with Torch NN
function load_images_and_labels( folder_path, width, height )
    classes = {}
    local image_paths = {}

    print( 'scanning images' )
    -- for each sub directory
    for dir in paths.iterdirs(folder_path) do
        local label = dir
        local subdir = paths.concat(folder_path,label)

        -- for each file
        for img in paths.iterfiles(subdir) do

            -- create the IDs for each label
            if classes[label] == nil then
                classes[label] = size_of(classes) + 1
            end

            local img_fullpath = paths.concat(subdir, img)

            -- trying to load the image
            -- it may look inneficient to do it twice in this function but it ensures
            -- image can be loaded and that it contains 3 channels
            -- it is important to get the proper tensor size for the returned table
            -- without having to resize it
            local loaded_img = image.load(img_fullpath)
            if loaded_img:size()[1] == 3 then
                image_paths[ img_fullpath ] = label
            end
        end
    end

    print( 'processing images' )
    -- initialize the tensors to be returned
    img_count = size_of(image_paths)
    image_set = {}
    image_set.label = torch.IntTensor(img_count):zero()
    image_set.data = torch.DoubleTensor(img_count, 3, width, height):zero()

    -- loop over the images that loaded successfully
    -- and fill the output tensor
    -- same with labels
    i = 0
    for k, v in pairs(image_paths) do
        i = i + 1
        xlua.progress( i, img_count )
        image_set.label[i] = classes[v]
        local tmp_img = image.load(k)
        image_set.data[i] = image.scale( tmp_img, width, height )
    end

    return image_set, classes
end
