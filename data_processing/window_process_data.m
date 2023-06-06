close all;clc;clearvars -except Normalized Streaming %removes all variables except for gaitcyle and continuous from the matlab workspace. Useful because loading these variables can take several minutes.

%assume 0.5 seconds at 100 Hz
LENGTH_TIME_WINDOW = 50;
% table_filename = 'r01_ordered_corrupt_time.csv';
% table_filename = 'r01_randomized_corrupt_time.csv';
% table_filename = 'dataport_ordered_corrupt_time.csv';
table_filename = 'dataport_randomized_corrupt_time.csv';

data = readtable(table_filename);

%remove non-live-estimated var indices
% STRIDE_COUNT_IDX = 9;
% HSDETECTED_IDX = 10;
% SUBJ_IDX = 11;
% TIME_IDX = 13;
% IDXS_TO_REMOVE = [STRIDE_COUNT_IDX,HSDETECTED_IDX, SUBJ_IDX, TIME_IDX];
% data_processed(:,IDXS_TO_REMOVE) = [];

%window the data
%only window the measurements
INPUT_VARS = ["foot_angle","foot_vel_angle","shank_angle","shank_vel_angle","dt"];
input_data = table2array(data(:,INPUT_VARS));


[N_trial_data,N_inputs] = size(input_data);
N_samples = N_trial_data - LENGTH_TIME_WINDOW;

data_windowed = zeros(N_samples, (N_inputs)*LENGTH_TIME_WINDOW);

for sample_idx = LENGTH_TIME_WINDOW:N_trial_data

    window_idxs = sample_idx-LENGTH_TIME_WINDOW+1:sample_idx;
%                 length(window_idxs)
%                 LENGTH_TIME_WINDOW %should be same


    %construct frame matrix, where every column is a
    %measurement/gait state and every row corresponds to a time step
    frame_matrix = input_data(window_idxs,:);

    %stack all the rows of the windowed data together
    %so in order, it's every measurement+state in a frame
    %sequentially, i.e. [frame1, frame2]
    frame_row = reshape(frame_matrix',1,[]);

    data_windowed(sample_idx-LENGTH_TIME_WINDOW+1,:) = frame_row;


end

%% append outputs to windowed data
OUTPUT_VARS = ["phase","speed","incline","is_moving"];
output_data = table2array(data(LENGTH_TIME_WINDOW:end,OUTPUT_VARS));

data_windowed = [data_windowed,output_data];


%% export data
% filename = [table_filename(1:end-4),sprintf('_window_%d',LENGTH_TIME_WINDOW),'.csv'];
% writematrix(data_windowed,filename);

filename = [table_filename(1:end-4),sprintf('_window_%d',LENGTH_TIME_WINDOW),'.parquet'];
data_windowed_T = array2table(data_windowed);
parquetwrite(filename,data_windowed_T)













