function kinematics = return_kinematics_gt(data_struct, foot_angle_offset)


    jointAngles = data_struct.jointAngles;
    jointVelocities = data_struct.jointVelocities;
    IMU = data_struct.IMU;
    times = data_struct.time;

    %extract angles in the sagittal plane and convert to degrees
    pelvis_angle = jointAngles.pelvis_tilt * 180/pi;
    hip_angle = jointAngles.hip_flexion * 180/pi;
    knee_angle = jointAngles.knee_angle * 180/pi;
    ankle_angle = jointAngles.ankle_angle * 180/pi;

    thigh_angle = pelvis_angle + hip_angle;
    shank_angle = thigh_angle - knee_angle;
    foot_angle = shank_angle + ankle_angle;

    % apply foot angle offset
    foot_angle = foot_angle - foot_angle_offset;
    shank_angle = shank_angle - foot_angle_offset;

    %calculate phases by dividing the time vector by the total duration
    phases = times./times(:,end);

    %calculate the idxs in early stance
    idxs_in_stance = find(phases >=0.15 & phases < 0.25);

    
    
    [N_strides, N_samples] = size(phases);

    kinematics.knee_angle = knee_angle;
    kinematics.ankle_angle = ankle_angle;
    kinematics.foot_angle = foot_angle;
    kinematics.shank_angle = shank_angle;
    kinematics.phases = phases;

    %extract joint velocities
    %extract angle vels in the sagittal plane and convert to degrees
    pelvis_angle_vel = jointVelocities.pelvis_tilt * 180/pi;
    hip_angle_vel = jointVelocities.hip_flexion * 180/pi;
    knee_angle_vel = jointVelocities.knee_angle * 180/pi;
    ankle_angle_vel = jointVelocities.ankle_angle * 180/pi;

    thigh_angle_vel = pelvis_angle_vel + hip_angle_vel;
    shank_angle_vel = thigh_angle_vel - knee_angle_vel;
    foot_angle_vel = shank_angle_vel + ankle_angle_vel;

    kinematics.shank_angle_vel = shank_angle_vel;
    kinematics.foot_angle_vel = foot_angle_vel;

    %extract foot IMU
    %this is in some stupid IMU frame, gonna have to get creative to beat
    %it into the heel accel
    foot_Accel_X_IMU = IMU.foot_Accel_X * 9.81; %close foot forward
    foot_Accel_Y_IMU = IMU.foot_Accel_Y * 9.81; %close to foot side
    foot_Accel_Z_IMU = IMU.foot_Accel_Z * 9.81; %close to foot upward

   

    %first, find the rotation about Y that rotates the IMU into Z upwards

    %obtain the acc measurements during foot flat
    foot_Accel_X_flat_contact = foot_Accel_X_IMU(idxs_in_stance);
    foot_Accel_Z_flat_contact = foot_Accel_Z_IMU(idxs_in_stance);
    
    %obtain the angle in the sagittal plane of the gravity vector
    theta_imu = -1 * atan(mean(foot_Accel_X_flat_contact)/mean(foot_Accel_Z_flat_contact));

    R_imu_to_flat_foot = [[cos(theta_imu), 0, sin(theta_imu)];[0, 1, 0];[-sin(theta_imu), 0 , cos(theta_imu)]];

    %apply rotation
    foot_Accel_X_aligned = zeros(N_strides, N_samples);
    foot_Accel_Y_aligned = zeros(N_strides, N_samples);
    foot_Accel_Z_aligned = zeros(N_strides, N_samples);
    for i = 1:N_strides
        accel_IMU_i = [foot_Accel_X_IMU(i,:);foot_Accel_Y_IMU(i,:);foot_Accel_Z_IMU(i,:);];
        accel_aligned_i = R_imu_to_flat_foot * accel_IMU_i;
        foot_Accel_X_aligned(i,:) = accel_aligned_i(1,:);
        foot_Accel_Y_aligned(i,:) = accel_aligned_i(2,:);
        foot_Accel_Z_aligned(i,:) = accel_aligned_i(3,:);


    end

    %remove gravity from Z acceleration
    foot_Accel_Z_aligned = foot_Accel_Z_aligned - 9.81;
    
    %now we need the distance in the inertial frame from the IMU location
    %to the back of the heel
    %define the distance in terms of the local foot frame first
    %this is based off an estimate using my own foot
    r_imu_to_heel_footFrame = [-5.8/100;0;-1.5/100]; %meters

    % we will also need the rotational acceleration of the foot
    %which we can obtain by numerically differentiating the rotational
    %velocity of the foot
    
    foot_angle_rot_acc = diff(foot_angle_vel,1,2)./(diff(times,1,2));
    foot_angle_rot_acc = [foot_angle_rot_acc(:,1),foot_angle_rot_acc];

    % now we can construct the accelerations of the heel usign the
    % kinematic formula for 2 pts on a rigid body
    
    heel_Accel_X = zeros(N_strides, N_samples);
    heel_Accel_Y = zeros(N_strides, N_samples);
    heel_Accel_Z = zeros(N_strides, N_samples);

    for j = 1:N_samples
        for i = 1:N_strides

            % define the rotation matrix that goes from foot to world
            foot_angle_j = -foot_angle(i,j) * pi/180;
            R_foot_to_world = [[cos(foot_angle_j), 0, sin(foot_angle_j)];...
                [0, 1, 0];...
                [-sin(foot_angle_j), 0 , cos(foot_angle_j)]];

            %aggregate the linear acceleration of the foot IMU into a 3 vector
            acc_foot_j = [foot_Accel_X_aligned(i,j);foot_Accel_Y_aligned(i,j);foot_Accel_Z_aligned(i,j)];

            %calculate the distance vector between imu and heel in global
            %frame
            r_imu_to_heel_globalFrame = R_foot_to_world * r_imu_to_heel_footFrame;
            
            %aggregate rotational acceleration (assuming only acc in the
            %sagittal plane
            %we need to negate the velocities since the foot angles follow
            %a negative convention relative to thee RHR implied by the IMU
            %frame
            rot_acc_foot_j = [0;-foot_angle_rot_acc(i,j)*pi/180;0];

            alpha_cross_r = cross(rot_acc_foot_j, r_imu_to_heel_globalFrame);

            %aggregate rotational vector
            omega_foot_j = [0;-foot_angle_vel(i,j)*pi/180;0];
                
            omega_cross_omega_cross_r = cross(omega_foot_j, cross(omega_foot_j, r_imu_to_heel_globalFrame));
            acc_heel_j = acc_foot_j + alpha_cross_r + omega_cross_omega_cross_r;
            
            %store variables
            heel_Accel_X(i,j) = acc_heel_j(1);
            heel_Accel_Y(i,j) = acc_heel_j(2);
            heel_Accel_Z(i,j) = acc_heel_j(3);

        end

    end

    %filter the accelerations to remove noise
    avg_dt = mean(mean(diff(times,1,2)));
    fpass = 5;
    heel_Accel_X = lowpass(heel_Accel_X',fpass,1/avg_dt);
    heel_Accel_Y = lowpass(heel_Accel_Y',fpass,1/avg_dt);
    heel_Accel_Z = lowpass(heel_Accel_Z',fpass,1/avg_dt);

    
    heel_Accel_X = heel_Accel_X';
    heel_Accel_Y = heel_Accel_Y';
    heel_Accel_Z = heel_Accel_Z';


    %numerically integrate heel positions
    heel_Pos_X = zeros(N_strides, N_samples);
    heel_Pos_Y = zeros(N_strides, N_samples);
    heel_Pos_Z = zeros(N_strides, N_samples);

%     for i = 1:N_strides
%         heel_Vel_X(i,:) = cumtrapz(times(i,:)-times(i,1),heel_Accel_X(i,:));
%         heel_Vel_Y(i,:) = cumtrapz(times(i,:)-times(i,1),heel_Accel_Y(i,:));
%         heel_Vel_Z(i,:) = cumtrapz(times(i,:)-times(i,1),heel_Accel_Z(i,:));
% 
%         %filter the vels to remove drift
%         fpass = 0.1;
%         heel_Vel_X = highpass(heel_Vel_X',fpass,1/avg_dt);
%         heel_Vel_Y = highpass(heel_Vel_Y',fpass,1/avg_dt);
%         heel_Vel_Z = highpass(heel_Vel_Z',fpass,1/avg_dt);
% 
%         heel_Vel_X = heel_Vel_X';
%         heel_Vel_Y = heel_Vel_Y';
%         heel_Vel_Z = heel_Vel_Z';
%     
% 
%         heel_Pos_X(i,:) = cumtrapz(times(i,:)-times(i,1),heel_Vel_X(i,:));
%         heel_Pos_Y(i,:) = cumtrapz(times(i,:)-times(i,1),heel_Vel_Y(i,:));
%         heel_Pos_Z(i,:) = cumtrapz(times(i,:)-times(i,1),heel_Vel_Z(i,:));
% 
%     end
    
    %filter the positions to remove drift
%     fpass = 0.01;
%     heel_Pos_X = highpass(heel_Pos_X',fpass,1/avg_dt);
%     heel_Pos_Y = highpass(heel_Pos_Y',fpass,1/avg_dt);
%     heel_Pos_Z = highpass(heel_Pos_Z',fpass,1/avg_dt);
    
%     heel_Pos_X = heel_Pos_X';
%     heel_Pos_Y = heel_Pos_Y';
%     heel_Pos_Z = heel_Pos_Z';

    kinematics.heel_Accel_X = heel_Accel_X;
    kinematics.heel_Accel_Y = heel_Accel_Y;
    kinematics.heel_Accel_Z = heel_Accel_Z;
    
    kinematics.heel_Pos_X = heel_Pos_X;
    kinematics.heel_Pos_Y = heel_Pos_Y;
    kinematics.heel_Pos_Z = heel_Pos_Z;


end








