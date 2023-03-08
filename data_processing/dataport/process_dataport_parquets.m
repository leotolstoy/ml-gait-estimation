close all;clc;clearvars -except Normalized Streaming %removes all variables except for gaitcyle and continuous from the matlab workspace. Useful because loading these variables can take several minutes.
%load the InclineExperiment data from the folder location you specify
% load('Z:\your_file_location_here\InclineExperiment.mat') 
%This file is large and may take several minutes to load. We recommend not
%clearing the variable, and only loading it as many times as necessary.

%% Example: removing strides with outliers from kinematic data, and taking the mean

sub={'AB01','AB02','AB03','AB04','AB05','AB06','AB07','AB08','AB09','AB10'};
% sub={'AB01'};


figure(1)
subplot(4,1,1)
hold on
subplot(4,1,2)
hold on
subplot(4,1,3)
hold on
subplot(4,1,4)
hold on

SAMPLE_RATE_VICON_NOMINAL = 100;
LEFT_LEG_NUM = 1;
RIGHT_LEG_NUM = 2;

figure(2)
subplot(2,1,1)
hold on
subplot(2,1,2)
hold on

figure(3)
subplot(2,1,1)
hold on
subplot(2,1,2)
hold on

%four measurements
dim_input = 4;

% the four traditional gait states
% speed
% stairs
% stop
dim_output = 4 + 1 + 1 + 1;


stride_count_master = 0;
data_matrix_master = [];
stride_count = 0;
for i = 1:length(sub) %loop through all subjects in 'sub'
    sub{i}
    switch sub{i}
            
        case 'AB01'
            footAngleZero = 85.46; %subject 1

        case 'AB02'
            footAngleZero = 85.46 + 4.506 + 1.62; %subject 2

        case 'AB03'
            footAngleZero = 85.46 + 5.564; %subject 3
        case 'AB04'
            footAngleZero = 85.46 + 7.681; %subject 4
        case 'AB05'
            footAngleZero = 85.46 + 5.405; %subject 5
        case 'AB06'
            footAngleZero = 85.46 + 4.089; %subject 6
        case 'AB07'
            footAngleZero = 85.46 + 1.523; %subject 7
        case 'AB08'
            footAngleZero = 85.46 + 3.305; %subject 8
        case 'AB09'
            footAngleZero = 85.46 + 4.396; %subject 9
        case 'AB10'
            footAngleZero = 85.46 + 6.555; %subject 10

    end
    
    %set up filename, which will have an AB number
    filename = sprintf('dataport_flattened_partial_%s.parquet',sub{i});
    
    %read parquet table
    data_subj_table = parquetread(filename);
    
    %extract relevant kinematics
    foot_angles_sagittal = -(data_subj_table.jointangles_foot_x + footAngleZero);
    shank_angles_sagittal = -(data_subj_table.jointangles_shank_x + footAngleZero);
    incline = data_subj_table.ramp;
    speed = data_subj_table.speed;
    phase = data_subj_table.phase;
    
    %obtain angle velocities
    foot_angle_vel_sagittal = -(data_subj_table.jointangles_foot_dot_x);
    shank_angle_vel_sagittal = -data_subj_table.jointangles_shank_dot_x;

    
    
    %based on how the parquets were set up, phase is exactly zero at HS
    HSDetected = phase == 0;

    %obtain heel accelerations
    %extract heel positions
    heel_pos_forward = data_subj_table.markers_heel_y/1000;
    heel_pos_upward = data_subj_table.markers_heel_z/1000;

    %multiply forward heel position by -1 when incline is negative to
    %account for world frame changing
    heel_pos_forward(incline<0) = -heel_pos_forward(incline<0);

    

    figure(99)
    hold on
    plot(heel_pos_forward)
    plot(heel_pos_upward)
    plot(HSDetected,'k')
    plot(incline/10,'r','LineWidth',2)
%     plot(speed,'b','LineWidth',2)

    
    %% Verify correct extraction
    figure(1)
    subplot(4,1,1)
    plot(phase,'b','LineWidth',2)
    plot(HSDetected,'k','LineWidth',2)

    subplot(4,1,2)
    plot(foot_angles_sagittal,'b','LineWidth',2)
    plot(HSDetected,'k','LineWidth',2)
    subplot(4,1,3)
    plot(shank_angles_sagittal,'b','LineWidth',2)
    plot(HSDetected,'k','LineWidth',2)

    subplot(4,1,4)
    plot(incline,'b','LineWidth',2)
    plot(speed,'r','LineWidth',2)
    plot(HSDetected,'k','LineWidth',2)

    figure(2)
    subplot(2,1,1)
    plot(phase(1:300),foot_angles_sagittal(1:300),'o')
    subplot(2,1,2)
    plot(phase(1:300),foot_angle_vel_sagittal(1:300),'o')

    figure(3)
    subplot(2,1,1)
    plot(phase(1:300),shank_angles_sagittal(1:300),'o')
    subplot(2,1,2)
    plot(phase(1:300),shank_angle_vel_sagittal(1:300),'o')
    
    %% Loop through the HSs to reconstruct time steps and update stride count

    HS_idxs = find(HSDetected);

    %set up dts vector
    dts = zeros(size(phase));
    stride_count_vec = zeros(size(phase));

    for jj = 1:length(HS_idxs)

        %construct stride indices
        if jj == length(HS_idxs)
            idxs = HS_idxs(jj):length(phase);
        else
            idxs = HS_idxs(jj):HS_idxs(jj+1)-1;
        end
        
        %extract phase rate
        phase_rate = data_subj_table.phase_dot(idxs(1));

        %obtain stride duration
        stride_duration = 1/phase_rate;

        %artificially populate dt vector
        dts_stride = stride_duration/(length(idxs));
        dts(idxs) = dts_stride;

        stride_count = stride_count + 1;
        stride_count_vec(idxs) = stride_count;

    end

    %obtain derivatives
    %heel forward
    heel_vel_forward = diff(heel_pos_forward)./dts(2:end);
    %median filter to get rid of large derivative spike due to the positions being
    %discontinuous
    heel_vel_forward = medfilt1(heel_vel_forward);
    heel_vel_forward = [heel_vel_forward(1);heel_vel_forward];

    heel_acc_forward = lowpass(diff(heel_vel_forward)./dts(2:end),20,1./mean(dts));
    heel_acc_forward = [heel_acc_forward(1);heel_acc_forward];

    %heel upward
    heel_vel_upward = diff(heel_pos_upward)./dts(2:end);
    heel_vel_upward = medfilt1(heel_vel_upward);
    heel_vel_upward = [heel_vel_upward(1);heel_vel_upward];

    heel_acc_upward = lowpass(diff(heel_vel_upward)./dts(2:end),20,1./mean(dts));
    heel_acc_upward = [heel_acc_upward(1);heel_acc_upward];

    
    
    figure(100)
    hold on
    plot(heel_acc_forward)
    plot(heel_acc_upward)
    plot(HSDetected*10,'k')
%     plot(speed,'b','LineWidth',2)
    

    %generate subject id vector
    subj_id_vec = i*ones(size(phase));

    %create is_moving vector that's all ones
    is_moving = ones(size(phase));

    %set is_stairs vector to all zeros
    is_stairs = zeros(size(phase));


    data_matrix = [foot_angles_sagittal,...
                    foot_angle_vel_sagittal,...
                    shank_angles_sagittal,...
                    shank_angle_vel_sagittal,...
                    heel_acc_forward,...
                    heel_acc_upward,...
                    phase,...
                    speed,...
                    incline,...
                    is_stairs,...
                    is_moving,...
                    stride_count_vec,...
                    HSDetected,...
                    subj_id_vec,...
                    dts];

    data_matrix_master = [data_matrix_master;data_matrix];


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

figure(10)
subplot(3,1,1)
hold on
plot(phase(10:5000), foot_angle(10:5000),'o')

subplot(3,1,2)
hold on
plot(phase(10:5000), shank_angle(10:5000),'o')
subplot(3,1,3)
hold on

subplot(3,1,3)
hold on
plot(phase(10:5000), heel_acc_forward(10:5000),'o')

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

table_filename = 'dataport_ordered_stairs_label.csv';
writetable(data_table_master,table_filename)


