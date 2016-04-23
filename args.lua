local M = { }

function M.parse( arg )
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('experimenting Torch')
    cmd:text()
    cmd:text('Options:')

    cmd:option('-images', '',      '[required] EITHER path to directory in which images to load are organized by sub-directories OR existing file containing data previously dumped')
    cmd:option('-nn', '',          '[required] path to the file neural network file')
    cmd:option('-width', 32,       '[ignored if loading data file] width in pixel to which images are resized to')
    cmd:option('-height', 32,      '[ignored if loading data file] height in pixel to which images are resized to')
    cmd:option('-dump', '',        '[optional] path to the file in which to dump the ready-to-go images (easier for reloading data)')
    cmd:option('-test', false,     '[optional] if on, neural net uses the data to test, otherwise neural network is trained and dumped to the <nn> file')
    cmd:option('-use_cuda', false, '[optional] set to true to use the GPU to train and test the neural network')

    cmd:text()

    return cmd:parse(arg or {})
end

return M
