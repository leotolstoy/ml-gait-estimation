%% Helper function that returns the stair height of each gt ascent trial given the string encoding the incline
function [stair_height] = return_stair_height_ascent_gt(trialString)


switch trialString
        case 'i4'
            stair_height = 102/1000;
        
        case 'i5'
            stair_height = 127/1000;
            
        case 'i6'
            stair_height = 152/1000;

        case 'i7'
            stair_height = 178/1000;

end



end