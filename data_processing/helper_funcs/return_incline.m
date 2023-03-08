%% Helper function
function [incline] = return_incline(trialString)

d_idx = strfind(trialString,'in');
i_idx = strfind(trialString,'i');

isIncline = ~isempty(i_idx) & isempty(d_idx);
isDecline = ~isempty(d_idx);

if isIncline
    inclineString = trialString(2:end);

    switch inclineString
        case '10'
            incline = 10;
        
        case '5'
            incline = 5;
            
        case '0'
            incline = 0;

    end




elseif isDecline
    inclineString = trialString(3:end);

    switch inclineString
        case '10'
            incline = -10;
        
        case '5'
            incline = -5;
            
        case '0'
            incline = 0;

    end

end


end