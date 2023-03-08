%% Helper function
function [incline, stair_height] = return_incline_stair_height_streaming(trialString)

d_idx = strfind(trialString,'in');
i_idx = strfind(trialString,'i');

isIncline = ~isempty(i_idx) & isempty(d_idx);
isDecline = ~isempty(d_idx);

if isIncline
    inclineString = trialString(2:end);

    switch inclineString
        case '20'
            incline = 0;
            stair_height = 97/1000;
        
        case '25'
            incline = 0;
            stair_height = 120/1000;
            
        case '30'
            incline = 0;
            stair_height = 146/1000;

        case '35'
            incline = 0;
            stair_height = 162/1000;

    end




elseif isDecline
    inclineString = trialString(3:end);

    switch inclineString
        case '20'
            incline = 0;
            stair_height = -97/1000;
        
        case '25'
            incline = 0;
            stair_height = -120/1000;
            
        case '30'
            incline = 0;
            stair_height = -146/1000;

        case '35'
            incline = 0;
            stair_height = -162/1000;

    end

end


end