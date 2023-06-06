close all;clc;clearvars -except Normalized Streaming %removes all variables except for gaitcyle and continuous from the matlab workspace. Useful because loading these variables can take several minutes.
%load the InclineExperiment data from the folder location you specify
% load('Z:\your_file_location_here\InclineExperiment.mat') 
%This file is large and may take several minutes to load. We recommend not
%clearing the variable, and only loading it as many times as necessary.

%% Example: removing strides with outliers from kinematic data, and taking the mean

sub={'AB01','AB02','AB03','AB04','AB05','AB06','AB07','AB08','AB09','AB10'};

sub={'AB04'};

joint={'foot'};

phase_vec = linspace(0,1,150)';
N_phase = length(phase_vec);

figure(1)
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


%initialize matices
samples_master = [];

for i = 1:length(sub) %loop through all subjects in 'sub'
    

    speed_fieldnames = fieldnames(Normalized.(sub{i}).Walk);
    %remove accel for now
    speed_fieldnames = setdiff(speed_fieldnames,{'a0x2','a0x5','d0x2','d0x5'})
%     speed_fieldnames = {'s1x2'}
    for j = 1:length(speed_fieldnames) %loop through all trials in 'trial'
        speed_fieldnames{j}
        incline_fieldnames = fieldnames(Normalized.(sub{i}).Walk.(speed_fieldnames{j}));
%         incline_fieldnames = {'i0'}
        for k = 1:length(incline_fieldnames)
            incline_fieldnames{k}

            foot_angle_data = Normalized.(sub{i}).Walk.(speed_fieldnames{j}).(incline_fieldnames{k}).jointAngles.FootProgressAngles;
%             foot_angle_data = Normalized.(sub{i}).Walk.(speed_fieldnames{j}).('i10').jointAngles.FootProgressAngles;
            
            %extract just the sagittal plane stuff
            foot_angle_data = squeeze(foot_angle_data(:,1,:));

            foot_angle_data = processFootAngles(sub{i},incline_fieldnames{k},foot_angle_data)


    
            [dim1,N_strides] = size(foot_angle_data);
            
            for stride_idx = 1:N_strides
                foot_angle_data_stride_y = foot_angle_data(:,stride_idx);
                
                figure(1)
                plot(phase_vec,foot_angle_data_stride_y,'b')
    
                
    
            end
%             pause

        end
        
        

    end
end

% The data in Gaitcyle always has 150 points per stride, and they are
% evenly spaced with respect to percent gait.



