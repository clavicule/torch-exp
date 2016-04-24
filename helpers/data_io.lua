require 'torch';
require 'paths';
require 'image';
require 'xlua';

paths.dofile( 'utils.lua' )

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

-- call this function to restructure the Pascal VOC dataset folders into
-- expected structure by load_images_and_labels() function.
-- [input: path_to_images] path to VOC folder containing the jpg images
-- [input: path_to_annotations] path to txt files containing list of images for train, val and both
-- [input: output_path] output folder where data will be copied to following the new folder structure
-- Note: this function copies images, the original data is not removed nor its original folder structure
function process_pascal_voc_dataset( path_to_images, path_to_annotations, output_path )
    if not paths.dirp( output_path ) then
        print( 'output directory does not exist' )
        return
    end

    -- iterate on all annotations files
    -- Naming expected
    -- *_train.txt *_trainval.txt and *_val.txt
    counter = 0
    for txt in paths.iterfiles( path_to_annotations ) do
        -- print progress: not using xlua because total number of files is not none
        -- it is possible to get that number, however it's not worth it
        counter = counter + 1
        print( '==> (' .. counter .. ') Processing file ' .. txt )

        -- figure out which type of data we are dealing with: val, train or both
        local train_class_name = string.split( txt, "_train" )
        local val_class_name = string.split( txt, "_val" )
        local trainval_class_name = string.split( txt, "_trainval" )
        class_name = nil
        dir_name = nil

        -- start with trainval, other split with train will grab it too
        if #trainval_class_name > 1 then
            class_name = trainval_class_name[1]
            dir_name = "trainval"
        elseif #train_class_name > 1 then
            class_name = train_class_name[1]
            dir_name = "train"
        elseif #val_class_name > 1 then
            class_name = val_class_name[1]
            dir_name = "val"
        end

        if class_name ~= nil and dir_name ~= nil then
            -- check if destination sub-directory structure exists
            -- create it if not
            local output_dir_name = paths.concat( output_path, dir_name )
            local output_dir_class = paths.concat( output_dir_name, class_name )

            if not paths.dirp( output_dir_name ) then
                paths.mkdir( output_dir_name )
            end

            if not paths.dirp( output_dir_class ) then
                paths.mkdir( output_dir_class )
            end

            -- read each line of the annoatation file
            for line in io.lines( paths.concat( path_to_annotations, txt ) ) do

                -- looking for 1's only, so we expect a 2-whitespace separation
                local is_class = string.split( line, "  " )
                local filename = ''
                if #is_class > 1 then
                    filename = paths.concat( path_to_images, is_class[1] .. '.jpg' )
                end

                local file_to_copy = paths.concat( path_to_images, filename )
                if paths.filep( file_to_copy ) then
                    os.execute( 'cp ' .. file_to_copy .. ' ' .. paths.concat( output_path, dir_name, class_name ) )
                end
            end
        end
    end
end
