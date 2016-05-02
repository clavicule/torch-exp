require 'torch';
require 'nn';


-- train the image set on the CNN
-- [input: dataset] the training dataset. Must comply with Torch NN classes
--    dataset must have 3 functions implemented:
--    nChannels(): number of color channels
--    w(): width of the images
--    h(): height of the images
-- the CNN parameters will be adjusted based on it
-- [input: use_cuda] if on, the training dataset must already be on GPU (not done because it would restric the dataset data structure)
-- [output: net] return the neural network
function train( dataset, no_classes, use_cuda )

    -- get number of color channels, width and height of the images
    c = dataset:nChannels()
    w = dataset:w()
    h = dataset:h()

    -- calculate the output dimensions of the tensors for each layer
    f1 = 5
    f2 = 2
    s2 = 2
    w2 = w - f1 + 1
    h2 = h - f1 + 1
    w3 = math.floor((w2 - f2) / s2 + 1)
    h3 = math.floor((h2 - f2) / s2 + 1)
    w4 = w3 - f1 + 1
    h4 = h3 - f1 + 1
    w5 = math.floor((w4 - f2) / s2 + 1)
    h5 = math.floor((h4 - f2) / s2 + 1)

    -- simple CNN architecture
    net = nn.Sequential()
    net:add(nn.SpatialConvolution( c, 6, f1, f1 ))
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling( f2, f2, s2, s2 ))
    net:add(nn.SpatialConvolution(6, 16, f1, f1))
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling( f2, f2, s2, s2 ))
    net:add(nn.View(16 * w5 * h5))
    net:add(nn.Linear(16 * w5 * h5, 120))
    net:add(nn.ReLU())
    net:add(nn.Linear(120, 84))
    net:add(nn.ReLU())
    net:add(nn.Linear(84, no_classes))
    net:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()

    -- load data on GPU if cuda option is on
    if use_cuda then
        require 'cunn';
        require 'cutorch';
        require 'cudnn';

        -- cudnn faster than nn with cuda
        cudnn.fastest = true
        cudnn.benchmark = true
        -- cudnn.verbose = true
        cudnn.convert(net, cudnn)
        net:cuda()
        criterion:cuda()
    end

    sgd = nn.StochasticGradient(net, criterion)
    sgd.learningRate = 0.01
    sgd.maxIteration = 10 -- # of epochs

    sgd:train(dataset)

    return net
end
