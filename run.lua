require 'paths';

-- parse arguments
local options = paths.dofile('args.lua')
opt = options.parse(arg)

dofile( 'main.lua' )
main( opt )
