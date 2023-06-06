%% Helper function that returns the stair height of each gt ascent trial given the string encoding the incline
function [stair_height] = return_stair_height_descent_gt(trialString)

switch trialString
    case 'in4'
        stair_height = 102/1000;
    
    case 'in5'
        stair_height = 127/1000;
        
    case 'in6'
        stair_height = 152/1000;

    case 'in7'
        stair_height = 178/1000;

end


stair_height = -stair_height;

end