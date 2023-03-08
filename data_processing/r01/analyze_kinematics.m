function [processed_struct] = analyze_kinematics(foot_angle_data_unproc,shank_angle_data_unproc, ...
    heel_pos_forward_unproc, heel_pos_upward_unproc,...
    HS_frames,...
    treadmill_speed, treadmill_incline, stride_count_start, leg_side_num, SAMPLE_RATE_VICON)

%only look at data starting from when the treadmill speed isn't nan
        nonnan_speed_idx = find(~isnan(treadmill_speed));
        first_idx = nonnan_speed_idx(1);
        last_idx = nonnan_speed_idx(end);

        %remove HSs outside the range of treadmill speeds not being nan
        HS_frames(HS_frames<first_idx) = [];
        HS_frames(HS_frames>last_idx) = [];

%         only look at events starting at the first HS and ending at the
        %last HS
        HS_range_idxs = HS_frames(1):HS_frames(end);
        foot_angle_data = foot_angle_data_unproc(HS_range_idxs);
        shank_angle_data = shank_angle_data_unproc(HS_range_idxs);
        treadmill_speed = treadmill_speed(HS_range_idxs);

        heel_pos_forward = heel_pos_forward_unproc(HS_range_idxs);
        heel_pos_upward = heel_pos_upward_unproc(HS_range_idxs);

        %generate foot velocity
        time_vec = ((0:length(foot_angle_data)-1)/SAMPLE_RATE_VICON)';
        diff_time_vec = diff(time_vec);
    
        foot_vel_angle_data = diff(foot_angle_data)./diff_time_vec;
        foot_vel_angle_data = [foot_vel_angle_data(1);foot_vel_angle_data];

        shank_vel_angle_data = diff(shank_angle_data)./diff_time_vec;
        shank_vel_angle_data = [shank_vel_angle_data(1);shank_vel_angle_data];

        %generate heel accelerations
        %heel forward
        heel_vel_forward = diff(heel_pos_forward)./diff_time_vec;
        heel_vel_forward = [heel_vel_forward(1);heel_vel_forward];

        heel_acc_forward = diff(heel_vel_forward)./diff_time_vec;
        heel_acc_forward = [heel_acc_forward(1);heel_acc_forward];

        %heel upward
        heel_vel_upward = diff(heel_pos_upward)./diff_time_vec;
        heel_vel_upward = [heel_vel_upward(1);heel_vel_upward];

        heel_acc_upward = diff(heel_vel_upward)./diff_time_vec;
        heel_acc_upward = [heel_acc_upward(1);heel_acc_upward];

        %saturate the heel accelerations to account for spikes
        heel_acc_forward(heel_acc_forward>100) = 100;
        heel_acc_forward(heel_acc_forward<-100) = -100;
        heel_acc_upward(heel_acc_upward>100) = 100;
        heel_acc_upward(heel_acc_upward<-100) = -100;

        %set up subject speed which can be zero if they're stationary
        speed = treadmill_speed;

        % update HS idxs
        HS_frames = HS_frames-HS_range_idxs(1)+1;

        HSDetected = zeros(size(foot_angle_data));
        HSDetected(HS_frames) = 1;

        %generate vector that encodes if the subject is moving
        is_moving = ones(size(foot_angle_data));

        %generate phases
        phase = zeros(size(foot_angle_data));


        %generate ramps
        incline = treadmill_incline*ones(size(foot_angle_data));
        
%         is_moving((abs(foot_vel_angle_data)) < 5) = 0;

        stride_count_vec = zeros(size(foot_angle_data));
        stride_count = stride_count_start;
%         iterate through strides
        for ii = 2:length(HS_frames)
            stride_start_idx = HS_frames(ii-1);
            stride_end_idx = HS_frames(ii);
            stride_idxs = stride_start_idx:stride_end_idx-1;

            if ii == length(HS_frames)
                stride_idxs = stride_start_idx:stride_end_idx;
            end

            %update foot angles if they're wrapped around
            avg_foot_angle_stride = mean(foot_angle_data(stride_idxs));
            if avg_foot_angle_stride > 130
                foot_angle_data(stride_idxs) = foot_angle_data(stride_idxs) - 180;
    
            elseif avg_foot_angle_stride < -130
                foot_angle_data(stride_idxs) = foot_angle_data(stride_idxs) - 180;
            end

            %update shank angles if they're wrapped around
            avg_shank_angle_stride = mean(shank_angle_data(stride_idxs));
            if avg_shank_angle_stride > 130
                shank_angle_data(stride_idxs) = shank_angle_data(stride_idxs) - 180;
    
            elseif avg_shank_angle_stride < -130
                shank_angle_data(stride_idxs) = shank_angle_data(stride_idxs) - 180;
            end



            N_samples = stride_end_idx - stride_start_idx + 1;
        
            phase_stride = linspace(0,1,N_samples);

            %if the subject obviously stopped (if the stride duration was
            %too long, set speed to zero and phase to zero
            if (stride_end_idx - stride_start_idx)/SAMPLE_RATE_VICON > 3
                speed(stride_idxs) = 0;
                is_moving(stride_idxs) = 0;
                phase_stride = zeros(1,N_samples);
            end

            if ii == length(HS_frames)
                phase(stride_idxs) = phase_stride(1:end);
            else
                phase(stride_idxs) = phase_stride(1:end-1);
            end
            %update total stride count
            stride_count = stride_count + 1;
            stride_count_vec(stride_idxs) = stride_count;

        end

        %adjust indexes so that the last frame isn't a heelstrike
        %this will help when concatenating the strides together across
        %trials

    processed_struct.foot_angle_data = foot_angle_data(1:end);
    processed_struct.foot_vel_angle_data = foot_vel_angle_data(1:end);
    processed_struct.shank_angle_data = shank_angle_data(1:end);
    processed_struct.shank_vel_angle_data = shank_vel_angle_data(1:end);
    
    processed_struct.heel_pos_forward = heel_pos_forward(1:end);
    processed_struct.heel_vel_forward = heel_vel_forward(1:end);
    processed_struct.heel_acc_forward = heel_acc_forward(1:end);

    processed_struct.heel_pos_upward = heel_pos_upward(1:end);
    processed_struct.heel_vel_upward = heel_vel_upward(1:end);
    processed_struct.heel_acc_upward = heel_acc_upward(1:end);

    processed_struct.time_vec = time_vec(1:end);
    processed_struct.phase = phase(1:end);
    processed_struct.speed = speed(1:end)';
    processed_struct.incline = incline(1:end);
    processed_struct.is_moving = is_moving(1:end);
    processed_struct.stride_count = stride_count;
    processed_struct.stride_count_vec = stride_count_vec(1:end);
    processed_struct.leg_side_num_vec = leg_side_num*ones(size(foot_angle_data(1:end)));
    processed_struct.HSDetected = HSDetected(1:end);
end