%% Helper function
function [speed] = return_speed(trialString)

speedString = trialString(2:end);

switch speedString
            
    case '0x8'
        speed = 0.8;

    case '1'
        speed = 1;

    case '1x2'
        speed = 1.2;

end


end