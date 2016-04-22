local M = { }

function M.parse( arg )
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('experimenting Torch')
    cmd:text()
    cmd:text('Options:')

    cmd:option('-image_path', '', 'root directory in which images to load are organized by sub-directories')
    cmd:option('-width', 231, 'width to which images are resized to')
    cmd:option('-height', 231, 'height to which images are resized to')
    cmd:text()

    return cmd:parse(arg or {})
end

return M
