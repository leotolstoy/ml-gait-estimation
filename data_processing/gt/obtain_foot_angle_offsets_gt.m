%% This script generates a vector of the foot angle offset (the bias from zero when the foot angle is in flat foot contact) for each subject

close all;clc;clearvars -except GT_Dataset_Normalized %removes all variables except for gaitcyle and continuous from the matlab workspace. Useful because loading these variables can take several minutes.

sub_ids = string(fieldnames(GT_Dataset_Normalized));
sub_ids(sub_ids == 'AB100') = [];


%% loop through subjects

foot_offsets = zeros(length(sub_ids),1);

for i =1:length(sub_ids)

    % handle treadmill
    sub_ids(i)
    sub_treadmill_data = GT_Dataset_Normalized.(sub_ids(i)).treadmill;

    foot_angles_flat_contact_sub = [];

    %loop through treadmill speeds

    treadmill_speed_strings = fieldnames(sub_treadmill_data);
    for j = 1:length(treadmill_speed_strings)
        
        %obtain speed for trial
        speed_string = treadmill_speed_strings{j};
        speed = return_speed_gt(speed_string);

        jointAngles = sub_treadmill_data.(speed_string).jointAngles;
        times = sub_treadmill_data.(speed_string).time;

        %extract angles in the sagittal plane and convert to degrees
        pelvis_angle = jointAngles.pelvis_tilt * 180/pi;
        hip_angle = jointAngles.hip_flexion * 180/pi;
        knee_angle = jointAngles.knee_angle * 180/pi;
        ankle_angle = jointAngles.ankle_angle * 180/pi;

        thigh_angle = pelvis_angle + hip_angle;
        shank_angle = thigh_angle - knee_angle;
        foot_angle = shank_angle + ankle_angle;
        
        %calculate phases by dividing the time vector by the total duration
        phases = times./times(:,end);


        idxs_in_stance = find(phases >=0.2 & phases < 0.3);

        foot_angles_flat_contact_trial = foot_angle(idxs_in_stance);
        foot_angles_flat_contact_sub = [foot_angles_flat_contact_sub;foot_angles_flat_contact_trial];

        %plot for sanity
%         figure(100)
%         subplot(2,1,1)
%         plot(times', hip_angle')
% 
%         subplot(2,1,2)
%         plot(times', thigh_angle')
% 
%         figure(101)
%         subplot(2,1,1)
%         plot(phases', shank_angle')
%         subplot(2,1,2)
%         plot(phases', foot_angle')


    end
    
    foot_offsets(i) = mean(mean(foot_angles_flat_contact_sub));

end




