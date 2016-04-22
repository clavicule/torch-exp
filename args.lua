local M = { }

function M.parse( arg )
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('experimenting Torch')
    cmd:text()
    cmd:text('Options:')

    cmd:option('-images', '', '[required] EITHER path to directory in which images to load are organized by sub-directories OR existing file containing data previously dumped')
    cmd:option('-width', 231, '[ignored if loading data file] width in pixel to which images are resized to')
    cmd:option('-height', 231, '[ignored if loading data file] height in pixel to which images are resized to')
    cmd:option('-dump', '', '[optional] path to file in which to dump the loaded images (easier for reloading data)' )
    cmd:text()

    return cmd:parse(arg or {})
end

return M
