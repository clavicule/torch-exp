require 'torch';
require 'nn';


-- NN parameters to be changed: right now, it accepts input image size of 32x32
-- train the image set on the CNN
-- [input: dataset] the training dataset. Must comply with Torch NN classes
-- [input: use_cuda] if on, the training dataset must already be on GPU (not done because it would restric the dataset data structure)
-- [output: net] return the neural network
function train( dataset, use_cuda )

    -- from Torch getting started tutorial
    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 6, 5, 5))     -- 3 input image channel, 6 output channels, 5x5 convolution kernel
    net:add(nn.ReLU())                             -- non-linearity
    net:add(nn.SpatialMaxPooling(2,2,2,2))         -- A max-pooling operation that looks at 2x2 windows and finds the max.
    net:add(nn.SpatialConvolution(6, 16, 5, 5))
    net:add(nn.ReLU())                             -- non-linearity
    net:add(nn.SpatialMaxPooling(2,2,2,2))
    net:add(nn.View(16*5*5))                       -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
    net:add(nn.Linear(16*5*5, 120))                -- fully connected layer (matrix multiplication between input and weights)
    net:add(nn.ReLU())                             -- non-linearity
    net:add(nn.Linear(120, 84))
    net:add(nn.ReLU())                             -- non-linearity
    net:add(nn.Linear(84, 5))                      -- 5 is the number of outputs of the network (no classes)
    net:add(nn.LogSoftMax())                       -- converts the output to a log-probability. Useful for classification problems

    criterion = nn.ClassNLLCriterion()

    -- load data on GPU if cuda option is on
    if use_cuda then
        require 'cunn';
        require 'cutorch';
        require 'cudnn';

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
