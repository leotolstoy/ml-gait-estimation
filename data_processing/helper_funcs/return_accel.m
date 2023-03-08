%% Helper function
function [accel] = return_accel(trialString)

accelString = trialString(2:end);

switch accelString
            
    case '0x2'
        accel = 0.2;

    case '0x5'
        accel = 0.5;

end

if strcmp(trialString(1),'d')
    accel = -accel;
end


end