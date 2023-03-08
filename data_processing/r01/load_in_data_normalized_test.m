close all;clc;clearvars -except Normalized Streaming %removes all variables except for gaitcyle and continuous from the matlab workspace. Useful because loading these variables can take several minutes.
%load the InclineExperiment data from the folder location you specify
% load('Z:\your_file_location_here\InclineExperiment.mat') 
%This file is large and may take several minutes to load. We recommend not
%clearing the variable, and only loading it as many times as necessary.

%% Example: removing strides with outliers from kinematic data, and taking the mean

sub={'AB01','AB02','AB03','AB04','AB05','AB06','AB07','AB08','AB09','AB10'};

sub={'AB07'};

joint={'foot'};

phase_vec = linspace(0,1,150)';
N_phase = length(phase_vec);

figure(1)
subplot(4,1,1)
hold on
subplot(4,1,2)
hold on
subplot(4,1,3)
hold on
subplot(4,1,4)
hold on

SAMPLE_RATE_VICON = 100;

%assume 0.2 seconds at 100 Hz
LENGTH_TIME_WINDOW = 20;

%four measurements
dim_input = 4;

% the four traditional gait states
% speed
% stairs
% stop
dim_output = 4 + 1 + 1 + 1;


figure(3)
subplot(4,1,1)
hold on
subplot(4,1,2)
hold on
subplot(4,1,3)
hold on
subplot(4,1,4)
hold on


for i = 1:length(sub) %loop through all subjects in 'sub'

    %initialize matices
    samples_master = [];
    
    foot_angle_data_y_master = [];
    foot_angle_vel_data_y_master = [];
    shank_angle_data_y_master = [];
    shank_angle_vel_data_y_master = [];
    
    phase_data_master = [];
    phase_rate_data_master = [];
    stride_length_data_master = [];
    incline_data_master = [];
    
    speed_data_master = [];
    is_stairs_data_master = [];
    is_moving_data_master = [];
    samples_master = [];
    
    speed_fieldnames = fieldnames(Normalized.(sub{i}).Walk);
    %remove accel for now
    speed_fieldnames = setdiff(speed_fieldnames,{'a0x2','a0x5','d0x2','d0x5'})

    for j = 1:length(speed_fieldnames) %loop through all trials in 'trial'
        speed_fieldnames{j}
        [treadmill_speed] = return_speed(speed_fieldnames{j});
        incline_fieldnames = fieldnames(Normalized.(sub{i}).Walk.(speed_fieldnames{j}));
        
        
        for k = 1:length(incline_fieldnames)
            incline_fieldnames{k}
            [treadmill_incline] = return_incline(incline_fieldnames{k});
            
            foot_angle_data = Normalized.(sub{i}).Walk.(speed_fieldnames{j}).(incline_fieldnames{k}).jointAngles.FootProgressAngles;
            ankle_angle_data = Normalized.(sub{i}).Walk.(speed_fieldnames{j}).(incline_fieldnames{k}).jointAngles.AnkleAngles;
            
            %extract just the sagittal plane stuff
            foot_angle_data = squeeze(foot_angle_data(:,1,:));
            ankle_angle_data = squeeze(ankle_angle_data(:,1,:));
            foot_angle_data = processFootAngles(sub{i},incline_fieldnames{k},foot_angle_data);
            
            shank_angle_data = foot_angle_data - ankle_angle_data;

            
            [dim1,N_strides] = size(foot_angle_data);


            events = Normalized.(sub{i}).Walk.(speed_fieldnames{j}).(incline_fieldnames{k}).events.StrideDetails;
            time_stride_durations = events(:,3)/SAMPLE_RATE_VICON;
            
            %concatenate all the strides together to simulate continuous
            %walking
            %extract the data just in the sagittal plane
            foot_angle_data_y = foot_angle_data; %rows = phase, cols = strides
            shank_angle_data_y = shank_angle_data;
            
            %initialize matrices to hold velocity data
            foot_angle_vel_data_y = zeros(N_phase,N_strides);
            shank_angle_vel_data_y = zeros(N_phase,N_strides);

            %initialize matrices to hold gait state data
            phase_data = zeros(N_phase,N_strides);
            phase_rate_data = zeros(N_phase,N_strides);
            stride_length_data = zeros(N_phase,N_strides);
            incline_data = zeros(N_phase,N_strides);

            speed_data = zeros(N_phase,N_strides);
            is_stairs_data = zeros(N_phase,N_strides);
            is_moving_data = zeros(N_phase,N_strides);

            for stride_idx = 1:N_strides
                foot_angle_data_stride_y = foot_angle_data(:,stride_idx);
                shank_angle_data_stride_y = shank_angle_data(:,stride_idx);
                
                %compute time vector
                time_stride_duration = time_stride_durations(N_strides);
                time_vec = linspace(0,time_stride_duration,N_phase)';

                %compute gait state elements
                phase_rate = 1/time_stride_duration;
                stride_length = time_stride_duration * treadmill_speed;
                incline = treadmill_incline;

                is_stairs = 0;
                is_moving = 1;
                
                %compute derivatives
                diff_time_vec = (1/150)*ones(N_phase-1,1);
    
                foot_vel_angle_data_stride_y = diff(foot_angle_data_stride_y)./diff_time_vec;
                foot_vel_angle_data_stride_y = [foot_vel_angle_data_stride_y(1);foot_vel_angle_data_stride_y];

                shank_vel_angle_data_stride_y = diff(shank_angle_data_stride_y)./diff_time_vec;
                shank_vel_angle_data_stride_y = [shank_vel_angle_data_stride_y(1);shank_vel_angle_data_stride_y];
                
                %store data
                foot_angle_vel_data_y(:,stride_idx) = foot_vel_angle_data_stride_y;
                shank_angle_vel_data_y(:,stride_idx) = shank_vel_angle_data_stride_y;

                phase_data(:,stride_idx) = phase_vec;
                phase_rate_data(:,stride_idx) = phase_rate*ones(size(phase_vec));
                stride_length_data(:,stride_idx) = stride_length*ones(size(phase_vec));
                incline_data(:,stride_idx) = incline*ones(size(phase_vec));

                speed_data(:,stride_idx) = treadmill_speed*ones(size(phase_vec));
                is_stairs_data(:,stride_idx) = is_stairs*ones(size(phase_vec));
                is_moving_data(:,stride_idx) = is_moving*ones(size(phase_vec));

%                 figure(1)
%                 subplot(4,1,1)
%                 plot(phase_vec,foot_angle_data_stride_y,'b')
% 
%                 subplot(4,1,2)
%                 plot(phase_vec,foot_vel_angle_data_stride_y,'b')
% 
%                 subplot(4,1,3)
%                 plot(phase_vec,shank_angle_data_stride_y,'r')
% 
%                 subplot(4,1,4)
%                 plot(phase_vec,shank_vel_angle_data_stride_y,'r')

                

            end

%             pause

            %flatten the measurements and gait states to column vectors to
            %simulate continuous motion

            foot_angle_data_y = reshape(foot_angle_data_y,[],1);
            [N_trial_data,~] = size(foot_angle_data_y);
            foot_angle_vel_data_y = reshape(foot_angle_vel_data_y,[],1);
            shank_angle_data_y = reshape(shank_angle_data_y,[],1);
            shank_angle_vel_data_y = reshape(shank_angle_vel_data_y,[],1);

            phase_data = reshape(phase_data,[],1);
            phase_rate_data = reshape(phase_rate_data,[],1);
            stride_length_data = reshape(stride_length_data,[],1);
            incline_data = reshape(incline_data,[],1);

            speed_data = reshape(speed_data,[],1);
            is_stairs_data = reshape(is_stairs_data,[],1);
            is_moving_data = reshape(is_moving_data,[],1);

            %concatenate to cross_trial matrices
            
    
%             figure(2)
%             subplot(4,1,1)
%             plot(foot_angle_data_y,'b')
% 
%             subplot(4,1,2)
%             plot(phase_data,'b')
% 
%             subplot(4,1,3)
%             plot(incline_data,'r')
% 
%             subplot(4,1,4)
%             plot(speed_data,'r')

            foot_angle_data_y_master = [foot_angle_data_y_master;foot_angle_data_y];
            foot_angle_vel_data_y_master = [foot_angle_vel_data_y_master;foot_angle_vel_data_y];
            shank_angle_data_y_master = [shank_angle_data_y_master;shank_angle_data_y];
            shank_angle_vel_data_y_master = [shank_angle_vel_data_y_master;shank_angle_vel_data_y];
            
            phase_data_master = [phase_data_master;phase_data];
            phase_rate_data_master = [phase_rate_data_master;phase_rate_data];
            stride_length_data_master = [stride_length_data_master;stride_length_data];
            incline_data_master = [incline_data_master;incline_data];
            
            speed_data_master = [speed_data_master;speed_data];
            is_stairs_data_master = [is_stairs_data_master;is_stairs_data];
            is_moving_data_master = [is_moving_data_master;is_moving_data];



            %window the data
            N_samples = N_trial_data - LENGTH_TIME_WINDOW;

            samples = zeros(N_samples, (dim_input+dim_output)*LENGTH_TIME_WINDOW);

            for sample_idx = LENGTH_TIME_WINDOW:N_trial_data

                window_idxs = sample_idx-LENGTH_TIME_WINDOW+1:sample_idx;
%                 length(window_idxs)
%                 LENGTH_TIME_WINDOW %should be same


                %construct frame matrix, where every column is a
                %measurement/gait state and every row corresponds to a time step
                frame_matrix = [foot_angle_data_y(window_idxs),...
                    foot_angle_vel_data_y(window_idxs),...
                    shank_angle_data_y(window_idxs),...
                    shank_angle_vel_data_y(window_idxs),...
                    phase_data(window_idxs),...
                    phase_rate_data(window_idxs),...
                    stride_length_data(window_idxs),...
                    incline_data(window_idxs),...
                    speed_data(window_idxs),...
                    is_stairs_data(window_idxs),...
                    is_moving_data(window_idxs)];

                %stack all the rows of the windowed data together
                %so in order, it's every measurement+state in a frame
                %sequentially, i.e. [frame1, frame2]
                frame_row = reshape(frame_matrix',1,[]);

                samples(sample_idx-LENGTH_TIME_WINDOW+1,:) = frame_row;


            end

            samples_master = [samples_master;samples];
        end
        
        

    end

    figure(3)
    subplot(4,1,1)
    plot(foot_angle_data_y_master,'b')

    subplot(4,1,2)
    plot(phase_data_master,'b')

    subplot(4,1,3)
    plot(incline_data_master,'r')

    subplot(4,1,4)
    plot(speed_data_master,'r')
end

% The data in Gaitcyle always has 150 points per stride, and they are
% evenly spaced with respect to percent gait.



