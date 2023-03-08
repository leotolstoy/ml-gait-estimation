function foot_angle_data_sagittal = processFootAngles(subj_string,incline_string,foot_angle_data_sagittal)
    switch subj_string
            
        case 'AB01'
            footAngleZero = 91.1257; %subject 1
        case 'AB02'
            footAngleZero = 90.577; %subject 1
        case 'AB03'
            footAngleZero = 89.7283; %subject 1
        case 'AB04'
            footAngleZero = 88.3549; %subject 1
        case 'AB05'
            footAngleZero = 93.0399; %subject 1
        case 'AB06'
            footAngleZero = 89.4403-1.97656; %subject 1
        case 'AB07'
            footAngleZero = 89.4938; %subject 4
        case 'AB08'
            footAngleZero = 89.1704; %subject 1
        case 'AB09'
            footAngleZero = 89.6244; %subject 1
        case 'AB10'
            footAngleZero = 90.1239; %subject 1

    end



    %apply transform with special handling for this one trial in
    %AB04
    if strcmp(subj_string,'AB04') & strcmp(incline_string,'i0')
        foot_angle_data_sagittal = foot_angle_data_sagittal - footAngleZero;
    else
        foot_angle_data_sagittal = -foot_angle_data_sagittal - footAngleZero;
    end

    %compensate for some foot angles being 180 offset
    avg_foot_angle_stride = mean(foot_angle_data_sagittal,1);
    for ii = 1:length(avg_foot_angle_stride)
        if avg_foot_angle_stride(ii) > 130
            foot_angle_data_sagittal(:,ii) = foot_angle_data_sagittal(:,ii) - 180;

        elseif avg_foot_angle_stride(ii) < -130
            foot_angle_data_sagittal(:,ii) = foot_angle_data_sagittal(:,ii) + 180;
        end
    end



end