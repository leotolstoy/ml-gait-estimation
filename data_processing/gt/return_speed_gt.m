%% Helper function that returns the speed of each gt trial given the string encoding the speed
function [speed] = return_speed_gt(trialString)

speedString = trialString(2:end);

left_of_x = speedString(1);
if length(speedString) > 1
    right_of_x = speedString(3:end);
    
    divisor = 100;
    if length(right_of_x) == 1
        divisor = 10;
    end
    
    speed = str2double(left_of_x) + str2double(right_of_x)/divisor;

else
    speed = str2double(left_of_x);
end