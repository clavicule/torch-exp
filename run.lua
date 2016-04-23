require 'paths';

-- use run.lua from Terminal
-- see README instructions

-- parse arguments
local options = paths.dofile('args.lua')
opt = options.parse(arg)

-- call main with arguments
dofile( 'main.lua' )
main()
