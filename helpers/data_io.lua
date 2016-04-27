require 'torch';
require 'paths';
require 'image';
require 'xlua';
require 'xml';

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
    local classes = {}
    local image_paths = {}

    print( 'scanning images' )
    -- for each sub directory
    for label in paths.iterdirs(folder_path) do
        local subdir = paths.concat(folder_path,label)

        -- create the IDs for each label
        if classes[label] == nil then
            classes[label] = size_of(classes) + 1
        end

        -- for each image file
        for img in paths.iterfiles(subdir) do
            -- trying to load the image
            -- it may look inneficient to do it twice in this function but it ensures
            -- image can be loaded and that it contains 3 channels
            -- it is important to get the proper tensor size for the returned table
            -- without having to resize it
            local img_fullpath = paths.concat(subdir, img)
            local loaded_img = image.load(img_fullpath)

            if loaded_img:size()[1] == 3 then
                image_paths[ img_fullpath ] = label
            end
        end
    end

    print( 'processing images' )
    -- initialize the tensors to be returned
    local img_count = size_of(image_paths)
    local image_set = {}
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

            paths.mkdir( output_dir_name )
            paths.mkdir( output_dir_class )

            -- read each line of the annotation file
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

-- call this function to crop the pascal VOC images by their anootation bounding box
-- [input: path_to_xml] path to VOC folder containing the xml annotation files
-- [input: path_to_images] path to VOC folder containing the jpg images
-- [input: output_path] output folder where data will be copied to following the new folder structure
-- Note: this function copies images before cropping, the original data is not removed
function crop_pascal_voc_dataset( path_to_xml, path_to_images, output_path )
    if not paths.dirp( output_path ) then
        print( 'output directory does not exist' )
        return
    end

    lubxml = require 'xml';
    lub = require 'lub';

    --  counter is also use to get unique image name and avoid overwritting
    counter = 0
    for xml_file in paths.iterfiles( path_to_xml ) do
        -- print progress: not using xlua because total number of files is not none
        -- it is possible to get that number, however it's not worth it
        counter = counter + 1
        print( '==> (' .. counter .. ') Processing file ' .. xml_file )

        -- read entire xml file as a string (lubxml takes string as input)
        local file = assert( io.open( paths.concat( path_to_xml, xml_file ), 'r' ) )
        local xml_content = file:read( '*all' )
        file:close()

        -- get the filename of the image we are dealing with
        local xml_table = lubxml.load( xml_content )
        local filename = lubxml.find( xml_table, 'filename' )[1]
        local image_filename =  paths.concat( path_to_images, filename )

        if paths.filep( image_filename ) then

            -- load the image once only
            local image_to_load = image.load( image_filename )

            -- recursive search to find all nodes of type object
            local object_list = {}
            lub.search( xml_table, function( node )
                if node.xml == 'object' then
                    table.insert( object_list, node )
                end
            end)

            -- get the bounding box and class name for each object
            for k, object in pairs( object_list ) do
                local class = lubxml.find( object, 'name' )
                local bbox = lubxml.find( object, 'bndbox' )
                local xmax = lubxml.find( bbox, 'xmax' )
                local xmin = lubxml.find( bbox, 'xmin' )
                local ymax = lubxml.find( bbox, 'ymax' )
                local ymin = lubxml.find( bbox, 'ymin' )

                local output_dir = paths.concat( output_path, class[1] )
                paths.mkdir( output_dir )

                local x1 = tonumber( xmin[1] )
                local y1 = tonumber( ymin[1] )
                local x2 = tonumber( xmax[1] )
                local y2 = tonumber( ymax[1] )

                -- crop the image by the bounding box and save it in the folder of the proper class
                local cropped_image = image.crop( image_to_load, x1, y1, x2, y2 )
                image.save( paths.concat( output_dir, counter .. '.jpg' ), cropped_image )
            end
        end
    end
end

-- split a dataset in 2 parts: train and test
-- [input: path_to_images] path to folder containing the jpg images organized in labelled sub-folders
-- [input: output_path] output folder where data will be copied to following the new folder structure (train/{label} and test/{label})
-- [input: split_ratio] % of images to copy in the train dataset. The rest is copied to test dataset
function split_dataset( path_to_images, output_path, split_ratio )
    if not paths.dirp( output_path ) then
        print( 'output directory does not exist' )
        return
    end

    if split_ratio < 0  or split_ratio > 1 then
        print( 'split ratio must be in range [0,1]')
        return
    end

    local train_path = paths.concat( output_path, 'train' )
    local test_path = paths.concat( output_path, 'test' )

    paths.mkdir( train_path )
    paths.mkdir( test_path )

    -- for each sub directory
    for label in paths.iterdirs(path_to_images) do
        local input_subdir = paths.concat(path_to_images, label)
        local train_sub_dir = paths.concat(train_path, label)
        local test_sub_dir = paths.concat(test_path, label)

        paths.mkdir( train_sub_dir )
        paths.mkdir( test_sub_dir )

        -- get the total count of file
        local count = 0
        for img in paths.iterfiles( input_subdir ) do
            count = count + 1
        end

        local train_size = math.floor(split_ratio * count)
        local total_count = count
        count = 0

        print( 'working on ' .. label )

        -- copy files to test or train based on split ratio
        for img in paths.iterfiles( input_subdir ) do
            count = count + 1
            xlua.progress( count, total_count )

            if count > train_size then
                os.execute( 'cp ' .. paths.concat( input_subdir, img ) .. ' ' .. test_sub_dir )
            else
                os.execute( 'cp ' .. paths.concat( input_subdir, img ) .. ' ' .. train_sub_dir )
            end
        end
    end
end
