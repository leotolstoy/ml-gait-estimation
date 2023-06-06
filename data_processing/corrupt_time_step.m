close all;clc;clearvars -except Normalized Streaming %removes all variables except for gaitcyle and continuous from the matlab workspace. Useful because loading these variables can take several minutes.
rng('default')
% table_filename = 'r01_ordered_stairs_label.csv';
table_filename = 'r01_randomized_stairs_label.csv';

% table_filename = 'dataport_ordered_stairs_label.csv';
% table_filename = 'dataport_randomized_stairs_label.csv';

% table_filename = 'gt_ordered_stairs_label.csv';
% table_filename = 'gt_randomized_stairs_label.csv';


data = readmatrix(table_filename);
DT_IDX = 15;
TIME_IDX = 16;
%extract dts
dts = data(:,DT_IDX);
time = data(:,TIME_IDX);

NOMINAL_SAMPLE_RATE = 0.01;

%compute how many samples there should be in the down sample
time_total = time(end);

N_downsamples = round(time_total/NOMINAL_SAMPLE_RATE)



%create corrupting noise
dt_sigma = 0.002;

dts_mean = NOMINAL_SAMPLE_RATE*ones(N_downsamples,1);
noise = dt_sigma * randn(N_downsamples,1);
dts_noise = dts_mean + noise;

%unlikely, but floor the dts to a minimum
dts_noise(dts_noise<1e-3) = 1e-3;

figure(1)
histogram(dts_noise)
xlim([0,0.03])

%generate new cumulative time vector
time_noise = cumsum(dts_noise);

%create new matrix to hold resampled data

[~, cols] = size(data);
data_resample = zeros(N_downsamples, cols);
data_resample(:,DT_IDX) = dts_noise;
data_resample(:,TIME_IDX) = time_noise;


%% artificially resample continuous data at new time vector

%intentionally do not put phase here
CONT_DATA_IDXS = [1,2,3,4,5,6,8,9,10,11];

for i = 1:length(CONT_DATA_IDXS)
    idx = CONT_DATA_IDXS(i);

    data_col = data(:,idx);
    data_col_resample = interp1(time,data_col,time_noise);
    data_resample(:,idx) = data_col_resample;


end

%% handle phase using trig to avoid issues with discontinuity
% otherwise phase won't roll over back to zero 
PHASE_IDX = 7;

phase_col = data(:,PHASE_IDX);

%construct sine and cosine of phase, which is scaled to an angle
%this guarantees continuity
sp = sin(2*pi*(phase_col-0.5));
cp = cos(2*pi*(phase_col-0.5));

%interpolate these continuous angles
sp_resample = interp1(time,sp,time_noise);
cp_resample = interp1(time,cp,time_noise);

%take arctan2 function of these angles
x = atan2(sp_resample,cp_resample);

%convert back to a phase signal
phase_p = ((x)/(2*pi)) + 0.5;

data_resample(:,PHASE_IDX) = phase_p;



%% artificially resample discrete data at new time vector
DISC_DATA_IDXS = [12,13,14];

for i = 1:length(DISC_DATA_IDXS)
    idx = DISC_DATA_IDXS(i);

    data_col = data(:,idx);
    data_col_resample = interp1(time,data_col,time_noise,'nearest');
    data_resample(:,idx) = round(data_col_resample);

end


%strip out any nans
data_resample(any(isnan(data_resample), 2), :) = [];


%% plot data
phase = data(:,7);
foot_angle = data(:,1);
is_stairs = data(:,10);
heel_acc_forward = data(:,5);
shank_angle = data(:,3);
speed = data(:,8);
incline = data(:,9);
stride_count = data(:,12);
subj_id = data(:,14);

phase_resample = data_resample(:,7);
foot_angle_resample = data_resample(:,1);
is_stairs_resample = data_resample(:,10);
heel_acc_forward_resample = data_resample(:,5);
shank_angle_resample = data_resample(:,3);
speed_resample = data_resample(:,8);
incline_resample = data_resample(:,9);
time_noise = data_resample(:,TIME_IDX);
stride_count_resample = data_resample(:,12);
subj_id_resample = data_resample(:,14);

%only plot part of the data 
% PLOT_IDXS = 1:20000;
% N = length(phase_resample);
% PLOT_IDXS = N-20000:N;

time_show_lims = [0,100]
% time_show_lims = [2.92e4,2.923e4]
figure(10)
subplot(5,1,1)
hold on
plot(time, phase)
plot(time_noise, phase_resample,'o')
legend('normal','resample')
ylabel('Phase')
xlim(time_show_lims)

subplot(5,1,2)
hold on
plot(time, foot_angle)
plot(time_noise, foot_angle_resample,'o')
ylabel('Foot angle')

xlim(time_show_lims)

subplot(5,1,3)
hold on
plot(time, is_stairs)
plot(time_noise, is_stairs_resample,'o')
ylabel('stair height')

xlim(time_show_lims)
subplot(5,1,4)
hold on
plot(time, incline)
plot(time_noise, incline_resample,'o')

ylabel('incline')

xlim(time_show_lims)
subplot(5,1,5)
hold on
plot(time, speed)
plot(time_noise, speed_resample,'o')

ylabel('speed')

xlim(time_show_lims)

figure(11)
subplot(3,1,1)
hold on
plot(time, phase)
plot(time_noise, phase_resample,'o')
xlim(time_show_lims)
ylabel('Phase')


subplot(3,1,2)
hold on
plot(time, heel_acc_forward)
plot(time_noise, heel_acc_forward_resample,'o')
xlim(time_show_lims)
ylabel('heel acc')


subplot(3,1,3)
hold on
plot(time, is_stairs)
plot(time_noise, is_stairs_resample,'o')
xlim(time_show_lims)
ylabel('stair height')


figure(12)
subplot(2,1,1)
hold on
plot(time, stride_count)
plot(time_noise, stride_count_resample,'o')
xlim(time_show_lims)
ylabel('Stride count')


subplot(2,1,2)
hold on
plot(time, subj_id)
plot(time_noise, subj_id_resample,'o')
xlim(time_show_lims)
ylabel('subj id')
%% export data
data_table_master = array2table(data_resample);
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


filename = [table_filename(1:end-4),'_corrupt_time.csv'];
writetable(data_table_master,filename);