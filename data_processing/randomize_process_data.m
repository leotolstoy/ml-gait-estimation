close all;clc;clearvars -except Normalized Streaming %removes all variables except for gaitcyle and continuous from the matlab workspace. Useful because loading these variables can take several minutes.
rng('default')
% table_filename = 'r01_ordered_stairs_label.csv';
% table_filename = 'dataport_ordered_stairs_label.csv';
table_filename = 'gt_ordered_stairs_label.csv';


data_processed = readmatrix(table_filename);
[N_data_total, ~] = size(data_processed)

STRIDE_COUNT_IDX = 12;
DT_IDX = 15;
TIME_IDX = 16;
STAIR_HEIGHT_IDX = 10;


%extract the strides that contain stairs
contains_stairs_idxs = data_processed(:,STAIR_HEIGHT_IDX) ~= 0;
data_stairs = data_processed(contains_stairs_idxs,:);
[N_data_stairs, ~] = size(data_stairs)

%remove the strides that contain stairs from the data matrix
data_processed(contains_stairs_idxs,:) = [];

%extract stride count
stride_counts = data_processed(:,STRIDE_COUNT_IDX);
stride_counts_unique = unique(stride_counts);

%randomize stride counts
stride_counts_unique = stride_counts_unique(randperm(length(stride_counts_unique)));

[N_data, ~] = size(data_processed);

%initialize empty matrix to hold randomized data
data_processed_randomized_strides = zeros(size(data_processed));

%initialize index for placing the stride data
data_idx = 0;

for i = 1:length(stride_counts_unique)
    stride_count = stride_counts_unique(i);
    %find data where the stride count is the stride count we want
    idxs = find(data_processed(:,STRIDE_COUNT_IDX) == stride_count);
    stride_data = data_processed(idxs,:);

    [N, ~] = size(stride_data);
    
    data_processed_randomized_strides(data_idx+1:data_idx+N,:) = stride_data;
    
    %update index
    data_idx = data_idx + N;


end

%append the stairs data to the end
data_processed_randomized_strides = [data_processed_randomized_strides;data_stairs];

%update the time vector so it's not randomized by reintegrating the
%randomized dts
dts = data_processed_randomized_strides(:,DT_IDX);
time = cumsum(dts);
data_processed_randomized_strides(:,TIME_IDX) = time;


%% plot data
phase = data_processed_randomized_strides(:,7);
foot_angle = data_processed_randomized_strides(:,1);
heel_acc_forward = data_processed_randomized_strides(:,5);

is_stairs = data_processed_randomized_strides(:,10);
is_moving = data_processed_randomized_strides(:,11);

shank_angle = data_processed_randomized_strides(:,3);
speed = data_processed_randomized_strides(:,8);

incline = data_processed_randomized_strides(:,9);

%only plot part of the data 
% PLOT_IDXS = 1:200000;
PLOT_IDXS = N_data_total-200000:N_data_total;

figure(10)
subplot(3,1,1)
hold on
plot(time(PLOT_IDXS), phase(PLOT_IDXS))

subplot(3,1,2)
hold on
plot(time(PLOT_IDXS), foot_angle(PLOT_IDXS))

subplot(3,1,3)
hold on
plot(time(PLOT_IDXS), is_moving(PLOT_IDXS))

figure(11)
subplot(4,1,1)
hold on
plot(time(PLOT_IDXS), incline(PLOT_IDXS))

subplot(4,1,2)
hold on
plot(time(PLOT_IDXS), speed(PLOT_IDXS))

subplot(4,1,3)
hold on
plot(time(PLOT_IDXS), heel_acc_forward(PLOT_IDXS))

subplot(4,1,4)
hold on
plot(time(PLOT_IDXS), is_stairs(PLOT_IDXS))

%% export data
data_table_master = array2table(data_processed_randomized_strides);
data_table_master.Properties.VariableNames =  {...
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


filename = strrep(table_filename, 'ordered','randomized')

writetable(data_table_master,filename);


