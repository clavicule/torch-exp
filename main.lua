require 'torch';
require 'paths';
require 'xlua';

-- use main.lua from iTorch
-- see README instructions

function main()
    paths.dofile( 'helpers/utils.lua' )

    -- process images and load as tensors
    if paths.dirp( opt.images ) then
        paths.dofile( 'helpers/data_io.lua' )
        dataset, classes = load_images_and_labels( opt.images, opt.width, opt.height )

    -- load directly the tensors from file
    elseif paths.filep( opt.images ) then
        print( 'loading data file ' .. opt.images )
        local loaded_data = torch.load( opt.images )
        dataset = loaded_data.images
        classes = loaded_data.classes

    else
        print( 'invalid input data path or file -- aborting' )
        return
    end

    -- save tensors to file
    if opt.dump ~= nil and opt.dump ~= '' then
        print( 'saving data to ' .. opt.dump )
        local data_to_save = {
            images = dataset,
            classes = classes
        }
        torch.save( opt.dump, data_to_save )
    end

    -- set table index and size function required by Torch NN
    setmetatable(dataset,
        {__index = function(t, i)
                        return {t.data[i], t.label[i]}
                    end}
    );

    function dataset:size()
        return self.data:size(1)
    end

    function dataset:nChannels()
        return self.data:size(2)
    end

    function dataset:w()
        return self.data:size(3)
    end

    function dataset:h()
        return self.data:size(4)
    end

    print( 'size of the dataset: ' .. dataset:size() )

    print( 'classes are:' )
    print( classes )

    if opt.nn == nil or opt.nn == '' then
        print( 'no NN provided, cannot train or test -- finished')
        return
    end

    no_classes = size_of(classes)

    -- load on GPU
    if opt.use_cuda then
        print( 'loading dataset on GPU' )
        require 'cutorch';
        require 'cunn';
        require 'cudnn';

        dataset.data = dataset.data:cuda()
        dataset.label = dataset.label:cuda()
    end


    paths.dofile( 'nn/cnn.lua' )
    if opt.test then
        if not paths.filep( opt.nn ) then
            print( 'no nn file provided -- aborting' )
            return
        end

        print( 'loading nn ' .. opt.nn )
        local loaded_data  = torch.load( opt.nn )
        net = loaded_data.nn
        mean = loaded_data.mean
        stdv = loaded_data.stdv

        if opt.use_cuda then
            cudnn.fastest = true
            cudnn.benchmark = true
            -- cudnn.verbose = true
            cudnn.convert(net, cudnn)
            dataset.data = dataset.data:cuda()
            dataset.label = dataset.label:cuda()
        end

        -- normalize dataset using the stats with which NN was trained on
        dataset.data = normalize_data( dataset.data, mean, stdv )

        classes_found = torch.DoubleTensor( no_classes ):zero()
        classes_total = torch.DoubleTensor( no_classes ):zero()
        for i = 1,dataset:size() do
            xlua.progress(i, dataset:size())

            local groundtruth = dataset.label[i]
            classes_total[ groundtruth ] = classes_total[ groundtruth ] + 1

            local prediction = net:forward( dataset.data[i] )
            prediction = prediction:exp()

            -- sort from most likely to less likely
            local proba, id_for_class = torch.sort( prediction, 1, true )

            -- get most probable class
            if groundtruth == id_for_class[1] then
                classes_found[ groundtruth ] = classes_found[ groundtruth ] + 1
            end
        end

        -- print test results
        for k, v in pairs(classes) do
            print( k .. ': ' .. 100 * classes_found[v] / classes_total[v] .. ' %' )
        end

    else
        -- get dataset statistics
        -- and normalize dataset
        mean, stdv = get_data_stats( dataset.data )
        dataset.data = normalize_data( dataset.data, mean, stdv )

        net = train( dataset, no_classes, opt.use_cuda )
        print( 'saving nn to ' .. opt.nn )
        local nn_to_save = {
            nn = net,
            mean = mean,
            stdv = stdv
        }
        torch.save( opt.nn, nn_to_save )
    end

    return net
end
