close all;clc;clearvars -except GT_Dataset_Normalized %removes all variables except for gaitcyle and continuous from the matlab workspace. Useful because loading these variables can take several minutes.

sub_ids = string(fieldnames(GT_Dataset_Normalized));
sub_ids(sub_ids == 'AB100') = [];
% sub_ids = ["AB12"]



figure(100)
subplot(2,1,1)
hold on
subplot(2,1,2)
hold on
figure(101)
subplot(2,1,1)
hold on
subplot(2,1,2)
hold on

stride_count_master = 0;
data_matrix_master = [];
stride_count = 0;

DO_TREADMILL = true;
DO_INCLINE_ASCENT = true;
DO_INCLINE_DESCENT = true;
DO_STAIRS_ASCENT = true;
DO_STAIRS_DESCENT = true;
%% loop through subjects for treadmill

if DO_TREADMILL
    for i =1:length(sub_ids)
        sub_id = sub_ids(i)
        % handle treadmill
        
        sub_treadmill_data = GT_Dataset_Normalized.(sub_ids(i)).treadmill;
    
        % get foot angle offset
        foot_angle_offset = return_foot_offset_gt(sub_ids(i));
    
        %loop through treadmill speeds
    
        treadmill_speed_strings = fieldnames(sub_treadmill_data);
    %     treadmill_speed_strings = {'s1'}
        for j = 1:length(treadmill_speed_strings)
            
            %obtain speed for trial
            speed_string = treadmill_speed_strings{j};
            speed = return_speed_gt(speed_string);
    
            
            kinematics = return_kinematics_gt( sub_treadmill_data.(speed_string), foot_angle_offset);
            times = sub_treadmill_data.(speed_string).time;
    
            
            knee_angle = kinematics.knee_angle;
            ankle_angle = kinematics.ankle_angle;
            foot_angle = kinematics.foot_angle;
            shank_angle = kinematics.shank_angle;
            phases = kinematics.phases;
            shank_angle_vel = kinematics.shank_angle_vel;
            foot_angle_vel = kinematics.foot_angle_vel;
    
            heel_acc_forward = kinematics.heel_Accel_X;
            heel_acc_upward = kinematics.heel_Accel_Z;

            heel_pos_forward = kinematics.heel_Pos_X;
            heel_pos_upward = kinematics.heel_Pos_Z;
            
    
            %plot for sanity
%             figure(100)
%             subplot(2,1,1)
%             plot(times', knee_angle')
%     
%             subplot(2,1,2)
%             plot(times', ankle_angle')
%     
%             figure(101)
%             subplot(2,1,1)
%             plot(phases', shank_angle')
%             subplot(2,1,2)
%             plot(phases', foot_angle')
%     
%             figure(102)
%             subplot(2,1,1)
%             plot(phases', shank_angle_vel')
%             subplot(2,1,2)
%             plot(phases', foot_angle_vel')
%     
            figure(103)
            subplot(2,1,1)
            plot(phases', heel_acc_forward')
            subplot(2,1,2)
            plot(phases', heel_acc_upward')

            figure(104)
            subplot(2,1,1)
            plot(phases', heel_pos_forward')
            subplot(2,1,2)
            plot(phases', heel_pos_upward')
    
            [N_strides, N_samples] = size(phases);

            %calculate any remaining foot angle bias by taking the average of the
            %foot angle in early stance (the overall average is
            %taken to ensure that these strides are continuous)
            remaining_foot_biases = zeros(N_strides,1);
            for nn = 1:N_strides
                idxs_in_stance = find(phases(nn,:) >=0.15 & phases(nn,:) < 0.25);

                remaining_foot_biases(nn) = mean(foot_angle(nn,idxs_in_stance)) ;
            end

            foot_angle = foot_angle - remaining_foot_biases;
            shank_angle = shank_angle - remaining_foot_biases;

    
            %reshape all the N_strides x N_samples matrixes to column vectors
            phases = reshape(phases',[],1);
            foot_angle = reshape(foot_angle',[],1);
            shank_angle = reshape(shank_angle',[],1);
            foot_angle_vel = reshape(foot_angle_vel',[],1);
            shank_angle_vel = reshape(shank_angle_vel',[],1);
            heel_acc_forward = reshape(heel_acc_forward',[],1);
            heel_acc_upward = reshape(heel_acc_upward',[],1);
            
            %generate task vectors
            speed_vec = speed * ones(size(phases));
            incline_vec = zeros(size(phases));

            %generate subject id vector
            subj_id_vec = str2double(sub_id{1}(3:end))*ones(size(phases));
        
            %create is_moving vector that's all ones
            is_moving_vec = ones(size(phases));
        
            %set is_stairs vector to all zeros
            is_stairs_vec = zeros(size(phases));
    
            %based on how the datafiles were set up, phase is exactly zero at HS
            HSDetected = phases == 0;
            HSDetected = reshape(HSDetected',[],1);
            
            %update stride count
            stride_count_vec = (stride_count + (1:N_strides));
            stride_count_vec = repelem(stride_count_vec, N_samples)';
            stride_count = stride_count + N_strides;
    
            %calculate dts
    
            dts = diff(times,1,2);
            dts = [dts(:,1),dts];
            dts = reshape(dts',[],1);
        
        
            data_matrix = [foot_angle,...
                            foot_angle_vel,...
                            shank_angle,...
                            shank_angle_vel,...
                            heel_acc_forward,...
                            heel_acc_upward,...
                            phases,...
                            speed_vec,...
                            incline_vec,...
                            is_stairs_vec,...
                            is_moving_vec,...
                            stride_count_vec,...
                            HSDetected,...
                            subj_id_vec,...
                            dts];
        
            %plot for sanity
            phases = data_matrix(:,7);
            dts = data_matrix(:,15);
            foot_angles = data_matrix(:,1);
            shank_angles = data_matrix(:,3);
            is_stairs = data_matrix(:,10);

            figure(901)
            subplot(2,1,1)
            plot(foot_angles)
            hold on
            
            plot(phases*10)

            subplot(2,1,2)
            plot(shank_angles)
            hold on
            plot(phases*10)

            data_matrix_master = [data_matrix_master;data_matrix];
    
    
        end
    
    end
end

if DO_INCLINE_ASCENT
    for i = 1:length(sub_ids)
        sub_id = sub_ids(i)
        sub_ramp_data = GT_Dataset_Normalized.(sub_ids(i)).ramp;
    
        % get foot angle offset
        foot_angle_offset = return_foot_offset_gt(sub_ids(i));
    
        % extract the different incline ascents
        incline_ascent_strings = fieldnames(sub_ramp_data.s1_ascent);
%         incline_ascent_strings = {'i18'}


        %loop through incline ascents
        for j = 1:length(incline_ascent_strings)

            incline_ascent_string = incline_ascent_strings{j};
            
            

            %obtain incline for trial
            incline = return_incline_ascent_gt(incline_ascent_string);
            data_matrix_incline = [];
            step_idxs_incline = [];
            
            % order the step strings such that it's all the leading leg
            % steps first, then the contralateral
%             ascent_ordered_step_strings = {'w2s_ascent',...
%                 's2_ascent',...
%                 's4_ascent',...
%                 's2w_ascent',...
%                 'a2_ascent',...
%                 's1_ascent',...
%                 's3_ascent',...
%                 's5_ascent',...
%                 'a1_ascent'};

            %only extract the steps that are on the ramp, which don't have
            %this issue where the phases from the other leg are out of sync
            ascent_ordered_step_strings = {
                's1_ascent',...
                's3_ascent',...
                's5_ascent',...
                'a1_ascent'};
            
            %filter out missing fields
            ascent_ordered_step_strings = intersect(ascent_ordered_step_strings,...
                fieldnames(sub_ramp_data),'stable');

            ascent_step_strings_marker = {};

            for k = 1:length(ascent_ordered_step_strings)
                ascent_ordered_step_string = ascent_ordered_step_strings{k};

                %obtain actual inclines present
                incline_ascent_strings_act = fieldnames(sub_ramp_data.(ascent_ordered_step_string));

                if any(strcmp(incline_ascent_strings_act, incline_ascent_string))
            
                    trial_data = sub_ramp_data.(ascent_ordered_step_string).(incline_ascent_string);
                    avg_speeds = trial_data.avgSpeed; %there'll be one speed per stride
    
                    %extract kinematics
                    kinematics = return_kinematics_gt( trial_data, foot_angle_offset);
                    times = trial_data.time;
                    
                    knee_angle = kinematics.knee_angle;
                    ankle_angle = kinematics.ankle_angle;
                    foot_angle = kinematics.foot_angle;
                    shank_angle = kinematics.shank_angle;
                    phases = kinematics.phases;
                    shank_angle_vel = kinematics.shank_angle_vel;
                    foot_angle_vel = kinematics.foot_angle_vel;
            
                    heel_acc_forward = kinematics.heel_Accel_X;
                    heel_acc_upward = kinematics.heel_Accel_Z;

                    heel_pos_forward = kinematics.heel_Pos_X;
                    heel_pos_upward = kinematics.heel_Pos_Z;
                    
            
                    %plot for sanity
            
%                     figure(101)
%                     subplot(2,1,1)
%                     plot(phases', shank_angle')
%                     subplot(2,1,2)
%                     plot(phases', foot_angle')
%             
    %                 figure(102)
    %                 subplot(2,1,1)
    %                 plot(phases', shank_angle_vel')
    %                 subplot(2,1,2)
    %                 plot(phases', foot_angle_vel')
    %         
                    figure(103)
                    subplot(2,1,1)
                    plot(phases', heel_acc_forward')
                    subplot(2,1,2)
                    plot(phases', heel_acc_upward')

                    figure(104)
                    subplot(2,1,1)
                    plot(phases', heel_pos_forward')
                    subplot(2,1,2)
                    plot(phases', heel_pos_upward')
            
                    [N_strides, N_samples] = size(phases);
    
                    %we're gonna create a matrix of indexes that range from 1
                    %to the number of strides. Later, we'll use this matrix to
                    %effectively sort the ordered ascent steps such that we get
                    %a continuous vector of locomotion following the ordered
                    %ascent (e.g. w2s, then s1, etc). 
    
                    step_idxs = repmat((1:N_strides)',1,N_samples);
    
                    %generate incline vector
                    %by default, use the nominal incline they're walking in
                    incline_vec = incline * ones(size(phases));
                    
                    %handle transition inclines
                    if strcmp(ascent_ordered_step_string,'w2s_ascent')
                        incline_vec = linspace(0,incline,N_samples);
                        incline_vec = repmat(incline_vec,N_strides,1);
    
                    elseif strcmp(ascent_ordered_step_string,'s2w_ascent')
                        incline_vec = linspace(incline,0,N_samples);
                        incline_vec = repmat(incline_vec,N_strides,1);
    
                    elseif strcmp(ascent_ordered_step_string,'a1_ascent') || ...
                        strcmp(ascent_ordered_step_string,'a2_ascent')
                        incline_vec = 0 * ones(size(phases));
                    end

                    %calculate any remaining foot angle bias by taking the average of the
                    %foot angle in early stance (the overall average is
                    %taken to ensure that these strides are continuous)
                    remaining_foot_biases = zeros(N_strides,1);
                    for nn = 1:N_strides
                        idxs_in_stance = find(phases(nn,:) >=0.15 & phases(nn,:) < 0.25);

                        remaining_foot_biases(nn) = mean(foot_angle(nn,idxs_in_stance)) ;
                    end

                    foot_angle = foot_angle - remaining_foot_biases;
                    shank_angle = shank_angle - remaining_foot_biases;
                
                    %enforce foot being equal to incline
                    foot_angle = foot_angle + incline_vec;
                    shank_angle = shank_angle + incline_vec;
                    
%                     %plot for sanity that this offsetting worked
%                     figure(101)
%                     subplot(2,1,1)
%                     plot(phases', shank_angle')
%                     subplot(2,1,2)
%                     plot(phases', foot_angle')

            
                    %reshape all the N_strides x N_samples matrixes to column vectors
                    phases = reshape(phases',[],1);
                    incline_vec = reshape(incline_vec',[],1);
                    foot_angle = reshape(foot_angle',[],1);
                    shank_angle = reshape(shank_angle',[],1);
                    foot_angle_vel = reshape(foot_angle_vel',[],1);
                    shank_angle_vel = reshape(shank_angle_vel',[],1);
                    heel_acc_forward = reshape(heel_acc_forward',[],1);
                    heel_acc_upward = reshape(heel_acc_upward',[],1);
                    
                    
    
                    % generate speed vector               
                    speed_vec = repelem(avg_speeds, N_samples);
    
            
                    %generate subject id vector
                    subj_id_vec = str2double(sub_id{1}(3:end))*ones(size(phases));
                
                    %create is_moving vector that's all ones
                    is_moving_vec = ones(size(phases));
                
                    %set is_stairs vector to all zeros
                    is_stairs_vec = zeros(size(phases));
            
                    %based on how the datafiles were set up, phase is exactly zero at HS
                    HSDetected = phases == 0;
                    HSDetected = reshape(HSDetected',[],1);
                    
                    %update stride count
                    stride_count_vec = (stride_count + (1:N_strides));
                    stride_count_vec = repelem(stride_count_vec, N_samples)';
                    stride_count = stride_count + N_strides;
            
                    %calculate dts
            
                    dts = diff(times,1,2);
                    dts = [dts(:,1),dts];
                    dts = reshape(dts',[],1);
    
    
                    %reshape sort indexes
                    step_idxs = reshape(step_idxs',[],1);
    
                    %update marker of what step we're on
                    ascent_step_strings_marker = [ascent_step_strings_marker;repmat({ascent_ordered_step_strings{k}},N_samples*N_strides,1)];
                
                
                    data_matrix = [foot_angle,...
                                    foot_angle_vel,...
                                    shank_angle,...
                                    shank_angle_vel,...
                                    heel_acc_forward,...
                                    heel_acc_upward,...
                                    phases,...
                                    speed_vec,...
                                    incline_vec,...
                                    is_stairs_vec,...
                                    is_moving_vec,...
                                    stride_count_vec,...
                                    HSDetected,...
                                    subj_id_vec,...
                                    dts];
                
                    data_matrix_incline = [data_matrix_incline;data_matrix];
                    step_idxs_incline = [step_idxs_incline;step_idxs];

                end

            end

            % sort all the steps in the incline by their actual ascent
            % order
            [sorted, sort_idxs] = sort(step_idxs_incline);

            data_matrix_incline = data_matrix_incline(sort_idxs,:);

            %undo the sort on the stride_count_vec
            data_matrix_incline(:,12) = sort(data_matrix_incline(:,12));

            ascent_step_strings_marker = ascent_step_strings_marker(sort_idxs);
            
            %plot for sanity
            phases = data_matrix_incline(:,7);
            dts = data_matrix_incline(:,15);
            foot_angles = data_matrix_incline(:,1);
            inclines = data_matrix_incline(:,9);

            figure(1001)
            plot(foot_angles)
            hold on
            plot(phases*10)
            plot(inclines)

    
            data_matrix_master = [data_matrix_master;data_matrix_incline];

        end
    
    end
end

if DO_INCLINE_DESCENT
    for i = 1:length(sub_ids)
        sub_id = sub_ids(i)       
        sub_ramp_data = GT_Dataset_Normalized.(sub_ids(i)).ramp;
    
        % get foot angle offset
        foot_angle_offset = return_foot_offset_gt(sub_ids(i));
    
        % extract the different decline ascents
        incline_descent_strings = fieldnames(sub_ramp_data.s1_descent);

        %loop through incline ascents
        for j = 1:length(incline_descent_strings)

            incline_descent_string = incline_descent_strings{j};
            %obtain incline for trial
            incline = return_incline_descent_gt(incline_descent_string);
            data_matrix_incline = [];
            step_idxs_incline = [];
            
            % order the step strings such that it's all the leading leg
            % steps first, then the contralateral
%             descent_ordered_step_strings = {'w2s_descent',...
%                 's2_descent',...
%                 's4_descent',...
%                 's2w_descent',...
%                 'a2_descent',...
%                 's1_descent',...
%                 's3_descent',...
%                 's5_descent',...
%                 'a1_descent'};
            
            %only extract the steps that are on the ramp, which don't have
            %this issue where the phases from the other leg are out of sync
            descent_ordered_step_strings = {
                's1_descent',...
                's3_descent',...
                's5_descent',...
                'a1_descent'};
            
            %filter out missing fields
            descent_ordered_step_strings = intersect(descent_ordered_step_strings,...
                fieldnames(sub_ramp_data),'stable');

            descent_step_strings_marker = {};

            for k = 1:length(descent_ordered_step_strings)
                descent_ordered_step_string = descent_ordered_step_strings{k};

                %obtain actual declines present
                incline_descent_strings_act = fieldnames(sub_ramp_data.(descent_ordered_step_string));

                if any(strcmp(incline_descent_strings_act, incline_descent_string))
            
                    trial_data = sub_ramp_data.(descent_ordered_step_string).(incline_descent_string);
                    avg_speeds = trial_data.avgSpeed; %there'll be one speed per stride
    
                    %extract kinematics
                    kinematics = return_kinematics_gt( trial_data, foot_angle_offset);
                    times = trial_data.time;
                    
                    knee_angle = kinematics.knee_angle;
                    ankle_angle = kinematics.ankle_angle;
                    foot_angle = kinematics.foot_angle;
                    shank_angle = kinematics.shank_angle;
                    phases = kinematics.phases;
                    shank_angle_vel = kinematics.shank_angle_vel;
                    foot_angle_vel = kinematics.foot_angle_vel;
            
                    heel_acc_forward = kinematics.heel_Accel_X;
                    heel_acc_upward = kinematics.heel_Accel_Z;

                    heel_pos_forward = kinematics.heel_Pos_X;
                    heel_pos_upward = kinematics.heel_Pos_Z;
                    
            
                    %plot for sanity
            
%                     figure(101)
%                     subplot(2,1,1)
%                     plot(phases', shank_angle')
%                     subplot(2,1,2)
%                     plot(phases', foot_angle')
            
%                     figure(102)
%                     subplot(2,1,1)
%                     plot(phases', shank_angle_vel')
%                     subplot(2,1,2)
%                     plot(phases', foot_angle_vel')
%             
                    figure(103)
                    subplot(2,1,1)
                    plot(phases', heel_acc_forward')
                    subplot(2,1,2)
                    plot(phases', heel_acc_upward')

                    figure(104)
                    subplot(2,1,1)
                    plot(phases', heel_pos_forward')
                    subplot(2,1,2)
                    plot(phases', heel_pos_upward')
            
                    [N_strides, N_samples] = size(phases);
    
                    %we're gonna create a matrix of indexes that range from 1
                    %to the number of strides. Later, we'll use this matrix to
                    %effectively sort the ordered ascent steps such that we get
                    %a continuous vector of locomotion following the ordered
                    %ascent (e.g. w2s, then s1, etc). 
    
                    step_idxs = repmat((1:N_strides)',1,N_samples);
    
                    %generate incline vector
                    incline_vec = incline * ones(size(phases));
                    if strcmp(descent_ordered_step_string,'w2s_descent')
                        incline_vec = linspace(0,incline,N_samples);
                        incline_vec = repmat(incline_vec,N_strides,1);
    
                    elseif strcmp(descent_ordered_step_string,'s2w_descent')
                        incline_vec = linspace(incline,0,N_samples);
                        incline_vec = repmat(incline_vec,N_strides,1);
    
                    elseif strcmp(descent_ordered_step_string,'a1_descent') || ...
                        strcmp(descent_ordered_step_string,'a2_descent')
                        incline_vec = 0 * ones(size(phases));
                    end

                    %calculate any remaining foot angle bias by taking the average of the
                    %foot angle in early stance (the overall average is
                    %taken to ensure that these strides are continuous)
                    remaining_foot_biases = zeros(N_strides,1);
                    for nn = 1:N_strides
                        idxs_in_stance = find(phases(nn,:) >=0.15 & phases(nn,:) < 0.25);

                        remaining_foot_biases(nn) = mean(foot_angle(nn,idxs_in_stance)) ;
                    end

                    foot_angle = foot_angle - remaining_foot_biases;
                    shank_angle = shank_angle - remaining_foot_biases;
                
                    %enforce foot being equal to incline
                    foot_angle = foot_angle + incline_vec;
                    shank_angle = shank_angle + incline_vec;
    
            
                    %reshape all the N_strides x N_samples matrixes to column vectors
                    phases = reshape(phases',[],1);
                    incline_vec = reshape(incline_vec',[],1);
                    foot_angle = reshape(foot_angle',[],1);
                    shank_angle = reshape(shank_angle',[],1);
                    foot_angle_vel = reshape(foot_angle_vel',[],1);
                    shank_angle_vel = reshape(shank_angle_vel',[],1);
                    heel_acc_forward = reshape(heel_acc_forward',[],1);
                    heel_acc_upward = reshape(heel_acc_upward',[],1);
                    
                    
    
                    % generate speed vector               
                    speed_vec = repelem(avg_speeds, N_samples);
    
            
                    %generate subject id vector
                    subj_id_vec = str2double(sub_id{1}(3:end))*ones(size(phases));
                
                    %create is_moving vector that's all ones
                    is_moving_vec = ones(size(phases));
                
                    %set is_stairs vector to all zeros
                    is_stairs_vec = zeros(size(phases));
            
                    %based on how the datafiles were set up, phase is exactly zero at HS
                    HSDetected = phases == 0;
                    HSDetected = reshape(HSDetected',[],1);
                    
                    %update stride count
                    stride_count_vec = (stride_count + (1:N_strides));
                    stride_count_vec = repelem(stride_count_vec, N_samples)';
                    stride_count = stride_count + N_strides;
            
                    %calculate dts
            
                    dts = diff(times,1,2);
                    dts = [dts(:,1),dts];
                    dts = reshape(dts',[],1);
    
    
                    %reshape sort indexes
                    step_idxs = reshape(step_idxs',[],1);
    
                    %update marker of what step we're on
                    descent_step_strings_marker = [descent_step_strings_marker;repmat({descent_ordered_step_strings{k}},N_samples*N_strides,1)];
                
                
                    data_matrix = [foot_angle,...
                                    foot_angle_vel,...
                                    shank_angle,...
                                    shank_angle_vel,...
                                    heel_acc_forward,...
                                    heel_acc_upward,...
                                    phases,...
                                    speed_vec,...
                                    incline_vec,...
                                    is_stairs_vec,...
                                    is_moving_vec,...
                                    stride_count_vec,...
                                    HSDetected,...
                                    subj_id_vec,...
                                    dts];
                
                    data_matrix_incline = [data_matrix_incline;data_matrix];
                    step_idxs_incline = [step_idxs_incline;step_idxs];

                end

            end

            % sort all the steps in the incline by their actual ascent
            % order
            [sorted, sort_idxs] = sort(step_idxs_incline);

            data_matrix_incline = data_matrix_incline(sort_idxs,:);

            %undo the sort on the stride_count_vec
            data_matrix_incline(:,12) = sort(data_matrix_incline(:,12));

            descent_step_strings_marker = descent_step_strings_marker(sort_idxs);
            
            %plot for sanity
            phases = data_matrix_incline(:,7);
            dts = data_matrix_incline(:,15);
            foot_angles = data_matrix_incline(:,1);
            inclines = data_matrix_incline(:,9);

            figure(1001)

            plot(foot_angles)
            hold on
            plot(phases*10)
            plot(inclines)
            data_matrix_master = [data_matrix_master;data_matrix_incline];

        end
    
    end
end

if DO_STAIRS_ASCENT
    for i = 1:length(sub_ids)
        sub_id = sub_ids(i)     
        sub_stairs_data = GT_Dataset_Normalized.(sub_ids(i)).stair;
    
        % get foot angle offset
        foot_angle_offset = return_foot_offset_gt(sub_ids(i));
    
        % extract the different incline ascents
        stairs_ascent_strings = fieldnames(sub_stairs_data.s1_ascent);
%         incline_ascent_strings = {'i18'}

        %loop through stairs ascents
        for j = 1:length(stairs_ascent_strings)

            stairs_ascent_string = stairs_ascent_strings{j};
            %logic for stair height
            stair_height = return_stair_height_ascent_gt(stairs_ascent_string);

            data_matrix_stairs = [];
            step_idxs_stairs = [];
            
            % order the step strings such that it's all the leading leg
            % steps first, then the contralateral
            ascent_ordered_step_strings = {'w2s_ascent',...
                's2_ascent',...
                's4_ascent',...
                's2w_ascent',...
                'a2_ascent',...
                's1_ascent',...
                's3_ascent',...
                's5_ascent',...
                'a1_ascent'};

            %only extract the steps that are on the ramp, which don't have
            %this issue where the phases from the other leg are out of sync
%             ascent_ordered_step_strings = {
%                 's1_ascent',...
%                 's3_ascent',...
%                 's5_ascent',...
%                 'a1_ascent'};
            
            %filter out missing fields
            ascent_ordered_step_strings = intersect(ascent_ordered_step_strings,...
                fieldnames(sub_stairs_data),'stable');

            ascent_step_strings_marker = {};

            for k = 1:length(ascent_ordered_step_strings)
                ascent_ordered_step_string = ascent_ordered_step_strings{k};

                %obtain actual inclines present
                stairs_ascent_strings_act = fieldnames(sub_stairs_data.(ascent_ordered_step_string));

                if any(strcmp(stairs_ascent_strings_act, stairs_ascent_string))
            
                    trial_data = sub_stairs_data.(ascent_ordered_step_string).(stairs_ascent_string);
                    avg_speeds = trial_data.avgSpeed; %there'll be one speed per stride
    
                    %extract kinematics
                    kinematics = return_kinematics_gt( trial_data, foot_angle_offset);
                    times = trial_data.time;
                    
                    knee_angle = kinematics.knee_angle;
                    ankle_angle = kinematics.ankle_angle;
                    foot_angle = kinematics.foot_angle;
                    shank_angle = kinematics.shank_angle;
                    phases = kinematics.phases;
                    shank_angle_vel = kinematics.shank_angle_vel;
                    foot_angle_vel = kinematics.foot_angle_vel;
            
                    heel_acc_forward = kinematics.heel_Accel_X;
                    heel_acc_upward = kinematics.heel_Accel_Z;

                    heel_pos_forward = kinematics.heel_Pos_X;
                    heel_pos_upward = kinematics.heel_Pos_Z;
                    
            
                    %plot for sanity
            
                    figure(101)
                    subplot(2,1,1)
                    plot(phases', shank_angle')
                    subplot(2,1,2)
                    plot(phases', foot_angle')
%             
                    figure(103)
                    subplot(2,1,1)
                    plot(phases', heel_acc_forward')
                    subplot(2,1,2)
                    plot(phases', heel_acc_upward')

                    figure(104)
                    subplot(2,1,1)
                    plot(phases', heel_pos_forward')
                    subplot(2,1,2)
                    plot(phases', heel_pos_upward')
            
                    [N_strides, N_samples] = size(phases);
    
                    %we're gonna create a matrix of indexes that range from 1
                    %to the number of strides. Later, we'll use this matrix to
                    %effectively sort the ordered ascent steps such that we get
                    %a continuous vector of locomotion following the ordered
                    %ascent (e.g. w2s, then s1, etc). 
    
                    step_idxs = repmat((1:N_strides)',1,N_samples);
    
                    %generate stairs vector
                    is_stairs_vec = stair_height*ones(size(phases));


                    if strcmp(ascent_ordered_step_string,'w2s_ascent')
                        is_stairs_vec = linspace(0,stair_height,N_samples);
                        is_stairs_vec = repmat(is_stairs_vec,N_strides,1);
    
                    elseif strcmp(ascent_ordered_step_string,'s2w_ascent')
                        is_stairs_vec = linspace(stair_height,0,N_samples);
                        is_stairs_vec = repmat(is_stairs_vec,N_strides,1);
    
                    elseif strcmp(ascent_ordered_step_string,'a1_ascent') || ...
                        strcmp(ascent_ordered_step_string,'a2_ascent')
                        is_stairs_vec = 0 * ones(size(phases));
                    end
                    
                    %calculate any remaining foot angle bias by taking the average of the
                    %foot angle in early stance (the overall average is
                    %taken to ensure that these strides are continuous)
%                     remaining_foot_biases = zeros(N_strides,1);
%                     for nn = 1:N_strides
%                         idxs_in_stance = find(phases(nn,:) >=0.15 & phases(nn,:) < 0.25);
% 
%                         remaining_foot_biases(nn) = mean(foot_angle(nn,idxs_in_stance)) ;
%                     end
% 
%                     foot_angle = foot_angle - remaining_foot_biases;
%                     shank_angle = shank_angle - remaining_foot_biases;
            
                    %reshape all the N_strides x N_samples matrixes to column vectors
                    phases = reshape(phases',[],1);
                    is_stairs_vec = reshape(is_stairs_vec',[],1);
                    foot_angle = reshape(foot_angle',[],1);
                    shank_angle = reshape(shank_angle',[],1);
                    foot_angle_vel = reshape(foot_angle_vel',[],1);
                    shank_angle_vel = reshape(shank_angle_vel',[],1);
                    heel_acc_forward = reshape(heel_acc_forward',[],1);
                    heel_acc_upward = reshape(heel_acc_upward',[],1);
                    
                    
    
                    % generate speed vector               
                    speed_vec = repelem(avg_speeds, N_samples);
    
            
                    %generate subject id vector
                    subj_id_vec = str2double(sub_id{1}(3:end))*ones(size(phases));
                
                    %create is_moving vector that's all ones
                    is_moving_vec = ones(size(phases));
                
                    %set incline vector to all zeros
                    incline_vec = zeros(size(phases));
            
                    %based on how the datafiles were set up, phase is exactly zero at HS
                    HSDetected = phases == 0;
                    HSDetected = reshape(HSDetected',[],1);
                    
                    %update stride count
                    stride_count_vec = (stride_count + (1:N_strides));
                    stride_count_vec = repelem(stride_count_vec, N_samples)';
                    stride_count = stride_count + N_strides;
            
                    %calculate dts
            
                    dts = diff(times,1,2);
                    dts = [dts(:,1),dts];
                    dts = reshape(dts',[],1);
    
    
                    %reshape sort indexes
                    step_idxs = reshape(step_idxs',[],1);
    
                    %update marker of what step we're on
                    ascent_step_strings_marker = [ascent_step_strings_marker;repmat({ascent_ordered_step_strings{k}},N_samples*N_strides,1)];
                
                
                    data_matrix = [foot_angle,...
                                    foot_angle_vel,...
                                    shank_angle,...
                                    shank_angle_vel,...
                                    heel_acc_forward,...
                                    heel_acc_upward,...
                                    phases,...
                                    speed_vec,...
                                    incline_vec,...
                                    is_stairs_vec,...
                                    is_moving_vec,...
                                    stride_count_vec,...
                                    HSDetected,...
                                    subj_id_vec,...
                                    dts];
                
                    data_matrix_stairs = [data_matrix_stairs;data_matrix];
                    step_idxs_stairs = [step_idxs_stairs;step_idxs];
                end

            end

            % sort all the steps in the stairs trial by their actual ascent
            % order
            [sorted, sort_idxs] = sort(step_idxs_stairs);

            data_matrix_stairs = data_matrix_stairs(sort_idxs,:);

            %undo the sort on the stride_count_vec
            data_matrix_stairs(:,12) = sort(data_matrix_stairs(:,12));

            ascent_step_strings_marker = ascent_step_strings_marker(sort_idxs);
            
            %plot for sanity
            phases = data_matrix_stairs(:,7);
            dts = data_matrix_stairs(:,15);
            foot_angles = data_matrix_stairs(:,1);
            shank_angles = data_matrix_stairs(:,3);
            is_stairs = data_matrix_stairs(:,10);

            figure(2001)
            subplot(2,1,1)
            plot(foot_angles)
            hold on
            
            plot(is_stairs)
            plot(phases*10)

            subplot(2,1,2)
            plot(shank_angles)
            hold on
            plot(is_stairs)
            plot(phases*10)

            
    
            data_matrix_master = [data_matrix_master;data_matrix_stairs];

        end
    
    end
end

if DO_STAIRS_DESCENT
    for i = 1:length(sub_ids)
        sub_id = sub_ids(i)        
        sub_stairs_data = GT_Dataset_Normalized.(sub_ids(i)).stair;
    
        % get foot angle offset
        foot_angle_offset = return_foot_offset_gt(sub_ids(i));
    
        % extract the different incline ascents
        stairs_descent_strings = fieldnames(sub_stairs_data.s1_descent);

        %loop through stairs ascents
        for j = 1:length(stairs_descent_strings)

            stairs_descent_string = stairs_descent_strings{j};

            stair_height = return_stair_height_descent_gt(stairs_descent_string);


            data_matrix_stairs = [];
            step_idxs_stairs = [];
            
            % order the step strings such that it's all the leading leg
            % steps first, then the contralateral
            descent_ordered_step_strings = {'w2s_descent',...
                's2_descent',...
                's4_descent',...
                's2w_descent',...
                'a2_descent',...
                's1_descent',...
                's3_descent',...
                's5_descent',...
                'a1_descent'};

             %only extract the steps that are on the ramp, which don't have
            %this issue where the phases from the other leg are out of sync
%             descent_ordered_step_strings = {
%                 's1_descent',...
%                 's3_descent',...
%                 's5_descent',...
%                 'a1_descent'};
            
            
            %filter out missing fields
            descent_ordered_step_strings = intersect(descent_ordered_step_strings,...
                fieldnames(sub_stairs_data),'stable');

            descent_step_strings_marker = {};

            for k = 1:length(descent_ordered_step_strings)
                descent_ordered_step_string = descent_ordered_step_strings{k};

                %obtain actual stairs present
                stairs_descent_strings_act = fieldnames(sub_stairs_data.(descent_ordered_step_string));

                if any(strcmp(stairs_descent_strings_act, stairs_descent_string))
            
                    trial_data = sub_stairs_data.(descent_ordered_step_string).(stairs_descent_string);
                    avg_speeds = trial_data.avgSpeed; %there'll be one speed per stride
    
                    %extract kinematics
                    kinematics = return_kinematics_gt( trial_data, foot_angle_offset);
                    times = trial_data.time;
                    
                    knee_angle = kinematics.knee_angle;
                    ankle_angle = kinematics.ankle_angle;
                    foot_angle = kinematics.foot_angle;
                    shank_angle = kinematics.shank_angle;
                    phases = kinematics.phases;
                    shank_angle_vel = kinematics.shank_angle_vel;
                    foot_angle_vel = kinematics.foot_angle_vel;
            
                    heel_acc_forward = kinematics.heel_Accel_X;
                    heel_acc_upward = kinematics.heel_Accel_Z;
                    
            
                    %plot for sanity
            
%                     figure(101)
%                     subplot(2,1,1)
%                     plot(phases', shank_angle')
%                     subplot(2,1,2)
%                     plot(phases', foot_angle')
%             
%                     figure(103)
%                     subplot(2,1,1)
%                     plot(phases', heel_acc_forward')
%                     subplot(2,1,2)
%                     plot(phases', heel_acc_upward')
%             
                    [N_strides, N_samples] = size(phases);
    
                    %we're gonna create a matrix of indexes that range from 1
                    %to the number of strides. Later, we'll use this matrix to
                    %effectively sort the ordered ascent steps such that we get
                    %a continuous vector of locomotion following the ordered
                    %ascent (e.g. w2s, then s1, etc). 
    
                    step_idxs = repmat((1:N_strides)',1,N_samples);
    
                    %generate stairs vector
                    is_stairs_vec = stair_height * ones(size(phases));

                    if strcmp(descent_ordered_step_string,'w2s_descent')
                        is_stairs_vec = linspace(0,stair_height,N_samples);
                        is_stairs_vec = repmat(is_stairs_vec,N_strides,1);
    
                    elseif strcmp(descent_ordered_step_string,'s2w_descent')
                        is_stairs_vec = linspace(stair_height,0,N_samples);
                        is_stairs_vec = repmat(is_stairs_vec,N_strides,1);
    
                    elseif strcmp(descent_ordered_step_string,'a1_descent') || ...
                        strcmp(descent_ordered_step_string,'a2_descent')
                        is_stairs_vec = 0 * ones(size(phases));
                    end

                    %calculate any remaining foot angle bias by taking the average of the
                    %foot angle in early stance (the overall average is
                    %taken to ensure that these strides are continuous)
%                     remaining_foot_biases = zeros(N_strides,1);
%                     for nn = 1:N_strides
%                         idxs_in_stance = find(phases(nn,:) >=0.15 & phases(nn,:) < 0.25);
% 
%                         remaining_foot_biases(nn) = mean(foot_angle(nn,idxs_in_stance)) ;
%                     end
% 
%                     foot_angle = foot_angle - remaining_foot_biases;
%                     shank_angle = shank_angle - remaining_foot_biases;
    
            
                    %reshape all the N_strides x N_samples matrixes to column vectors
                    phases = reshape(phases',[],1);
                    is_stairs_vec = reshape(is_stairs_vec',[],1);
                    foot_angle = reshape(foot_angle',[],1);
                    shank_angle = reshape(shank_angle',[],1);
                    foot_angle_vel = reshape(foot_angle_vel',[],1);
                    shank_angle_vel = reshape(shank_angle_vel',[],1);
                    heel_acc_forward = reshape(heel_acc_forward',[],1);
                    heel_acc_upward = reshape(heel_acc_upward',[],1);
                    
                    
    
                    % generate speed vector               
                    speed_vec = repelem(avg_speeds, N_samples);
    
            
                    %generate subject id vector
                    subj_id_vec = str2double(sub_id{1}(3:end))*ones(size(phases));
                
                    %create is_moving vector that's all ones
                    is_moving_vec = ones(size(phases));
                
                    %set incline vector to all zeros
                    incline_vec = zeros(size(phases));
            
                    %based on how the datafiles were set up, phase is exactly zero at HS
                    HSDetected = phases == 0;
                    HSDetected = reshape(HSDetected',[],1);
                    
                    %update stride count
                    stride_count_vec = (stride_count + (1:N_strides));
                    stride_count_vec = repelem(stride_count_vec, N_samples)';
                    stride_count = stride_count + N_strides;
            
                    %calculate dts
            
                    dts = diff(times,1,2);
                    dts = [dts(:,1),dts];
                    dts = reshape(dts',[],1);
    
    
                    %reshape sort indexes
                    step_idxs = reshape(step_idxs',[],1);
    
                    %update marker of what step we're on
                    descent_step_strings_marker = [descent_step_strings_marker;repmat({descent_ordered_step_strings{k}},N_samples*N_strides,1)];
                
                
                    data_matrix = [foot_angle,...
                                    foot_angle_vel,...
                                    shank_angle,...
                                    shank_angle_vel,...
                                    heel_acc_forward,...
                                    heel_acc_upward,...
                                    phases,...
                                    speed_vec,...
                                    incline_vec,...
                                    is_stairs_vec,...
                                    is_moving_vec,...
                                    stride_count_vec,...
                                    HSDetected,...
                                    subj_id_vec,...
                                    dts];
                
                    data_matrix_stairs = [data_matrix_stairs;data_matrix];
                    step_idxs_stairs = [step_idxs_stairs;step_idxs];
                end

            end

            % sort all the steps in the incline by their actual ascent
            % order
            [sorted, sort_idxs] = sort(step_idxs_stairs);

            data_matrix_stairs = data_matrix_stairs(sort_idxs,:);

            %undo the sort on the stride_count_vec
            data_matrix_stairs(:,12) = sort(data_matrix_stairs(:,12));

            descent_step_strings_marker = descent_step_strings_marker(sort_idxs);
            
            %plot for sanity
            phases = data_matrix_stairs(:,7);
            dts = data_matrix_stairs(:,15);
            foot_angles = data_matrix_stairs(:,1);
            shank_angles = data_matrix_stairs(:,3);
            is_stairs = data_matrix_stairs(:,10);

            figure(3001)
            subplot(2,1,1)
            plot(foot_angles)
            hold on
            
            plot(is_stairs)
            plot(phases*10)

            subplot(2,1,2)
            plot(shank_angles)
            hold on
            plot(is_stairs)
            plot(phases*10)
    
            data_matrix_master = [data_matrix_master;data_matrix_stairs];

        end
    
    end
end
%process total time
DT_IDX = 15;
dts = data_matrix_master(:,DT_IDX);
time = cumsum(dts);
data_matrix_master = [data_matrix_master,time];

%% plot foot angle vs phase
phase = data_matrix_master(:,7);
foot_angle = data_matrix_master(:,1);
shank_angle = data_matrix_master(:,3);
heel_acc_forward = data_matrix_master(:,5);

is_stairs = data_matrix_master(:,10);


figure(10)
subplot(3,1,1)
hold on
plot(phase, foot_angle,'o')

subplot(3,1,2)
hold on
plot(phase, shank_angle,'o')
subplot(3,1,3)
hold on

subplot(3,1,3)
hold on
plot(phase(10:4000), heel_acc_forward(10:4000),'o')

figure(11)
hold on
plot(is_stairs,'o')

%% export data matrix to table
data_table_master = array2table(data_matrix_master);
data_table_master.Properties.VariableNames = {...
    'foot_angle',...
    'foot_vel_angle',...
    'shank_angle',...
    'shank_vel_angle',...
    'heel_acc_forward',...
    'heel_acc_upward',...
    'phase',...
    'speed',...
    'incline',...
    'stair_height',...
    'is_moving',...
    'stride_count',...
    'HSDetected',...
    'subj_id',...
    'dt',...
    'time'};

table_filename = 'gt_ordered_stairs_label.csv';
writetable(data_table_master,table_filename)
