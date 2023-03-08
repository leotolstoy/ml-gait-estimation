function ankle_angle_data_sagittal = processAnkleAngles(ankle_angle_data_sagittal)
    

    %compensate for some foot angles being 180 offset
    avg_foot_angle_stride = mean(ankle_angle_data_sagittal,1);
    for ii = 1:length(avg_foot_angle_stride)
        if avg_foot_angle_stride(ii) > 130
            ankle_angle_data_sagittal(:,ii) = ankle_angle_data_sagittal(:,ii) - 180;

        elseif avg_foot_angle_stride(ii) < -130
            ankle_angle_data_sagittal(:,ii) = ankle_angle_data_sagittal(:,ii) + 180;
        end
    end



end