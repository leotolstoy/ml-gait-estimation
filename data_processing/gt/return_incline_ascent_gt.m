%% Helper function that returns the incline of each gt ascent trial given the string encoding the incline
function [incline] = return_incline_ascent_gt(trialString)

inclineString = trialString(2:end);

x_idx = strfind(inclineString,'x');

if x_idx > 0
    left_of_x = inclineString(1:x_idx-1);

    right_of_x = inclineString(x_idx+1:end);
    divisor = 10^length(right_of_x);
    incline = str2double(left_of_x) + str2double(right_of_x)/divisor;

else
    left_of_x = inclineString;
    incline = str2double(left_of_x);
end


end