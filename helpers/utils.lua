require 'torch';

-- returns size of string array
function size_of(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

-- returns the mean and std dev of each channel (3x1 tensor for each)
function get_data_stats( data )
    mean = {} -- store the mean, to normalize the test set in the future
    stdv  = {} -- store the standard-deviation for the future
    for i = 1,3 do -- over each image channel
        mean[i] = data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
        stdv[i] = data[{ {}, {i}, {}, {}  }]:std() -- std estimation

        print( 'mean = ' .. mean[i] .. ' | std_dev = ' .. stdv[i] )
    end

    return mean, stdv
end

-- normalize the dataset with the given mean and std dev
function normalize_data( data, mean, stdv )
    for i = 1,3 do -- over each image channel
        data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
        data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end

    return data
end
