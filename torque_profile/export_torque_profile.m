close all;clc;clearvars -except GT_Dataset_Normalized %removes all variables except for gaitcyle and continuous from the matlab workspace. Useful because loading these variables can take several minutes.

sub_ids = string(fieldnames(GT_Dataset_Normalized));
sub_ids(sub_ids == 'AB100') = [];
sub_ids = ["AB12"]

avg_subj_mass = 75;


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
stride_count = 0;

A_mat_master = [];
b_ankle_torque_master = [];

numInclineFuncs = 2;
numStepLengthFuncs = 2;
N_FOURIER = 20;
numPhaseFuncs = (length(1:1:N_FOURIER) * 2) + 1;
numFuncs = numInclineFuncs*numStepLengthFuncs*numPhaseFuncs;

DO_TREADMILL = true;
DO_INCLINE_ASCENT = true;
DO_INCLINE_DESCENT = true;
%% loop through subjects for treadmill

if DO_TREADMILL
    for i =1:length(sub_ids)
        sub_id = sub_ids(i)
        % handle treadmill
        
        sub_treadmill_data = GT_Dataset_Normalized.(sub_ids(i)).treadmill;
        
        %loop through treadmill speeds
        treadmill_speed_strings = fieldnames(sub_treadmill_data);
    %     treadmill_speed_strings = {'s1'}
        for j = 1:length(treadmill_speed_strings)
            
            %obtain speed for trial
            speed_string = treadmill_speed_strings{j};
            speed = return_speed_gt(speed_string);
            
            %incline is zero here
            incline = 0;
            
            kinetics = return_kinetics_gt( sub_treadmill_data.(speed_string));
            times = sub_treadmill_data.(speed_string).time;
    
            
            ankle_torque = -kinetics.ankle_torque * avg_subj_mass;
            phases = kinetics.phases;
    
            %plot for sanity
            figure(100)
            subplot(2,1,1)
            plot(phases', ankle_torque')
    
    
            %reshape all the N_strides x N_samples matrixes to column vectors
            phases = reshape(phases',[],1);
            ankle_torque = reshape(ankle_torque',[],1);
            
            %generate task vectors
            speed_vec = speed * ones(size(phases));
            incline_vec = incline * ones(size(phases));
            
            % set up regressor matrices
            [rows,cols] = size(phases);
        
            A_mat = zeros(rows,numFuncs);

            for ii = 1:rows
                fourier_coeffs = returnFourierBasis_Eval(phases(ii),speed_vec(ii),incline_vec(ii),N_FOURIER);
                A_mat(ii,:) = fourier_coeffs;

            end
            b_ankle_torque_mat = ankle_torque;

            A_mat_master = [A_mat_master;A_mat];
            b_ankle_torque_master = [b_ankle_torque_master;b_ankle_torque_mat];
    
    
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

            for k = 1:length(ascent_ordered_step_strings)
                ascent_ordered_step_string = ascent_ordered_step_strings{k};

                %obtain actual inclines present
                incline_ascent_strings_act = fieldnames(sub_ramp_data.(ascent_ordered_step_string));

                if any(strcmp(incline_ascent_strings_act, incline_ascent_string))
            
                    trial_data = sub_ramp_data.(ascent_ordered_step_string).(incline_ascent_string);
                    avg_speeds = trial_data.avgSpeed; %there'll be one speed per stride
    
                    %extract kinetics
                    kinetics = return_kinetics_gt( trial_data);
                    times = trial_data.time;
                
                    ankle_torque = -kinetics.ankle_torque * avg_subj_mass;
                    phases = kinetics.phases;

                    

                    [N_strides, N_samples] = size(phases);
    
    
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

            
                    %reshape all the N_strides x N_samples matrixes to column vectors
                    phases = reshape(phases',[],1);
                    incline_vec = reshape(incline_vec',[],1);
                    ankle_torque = reshape(ankle_torque',[],1);

                    % generate speed vector               
                    speed_vec = repelem(avg_speeds, N_samples);

                    %threshold out ankle torques that were incorrectly
                    %recorded
                    threshold_mask = [];
                    if mean(abs(ankle_torque)) < 20
                        threshold_mask = 1:length(ankle_torque);
                    end
                    phases(threshold_mask) = [];
                    incline_vec(threshold_mask) = [];
                    ankle_torque(threshold_mask) = [];
                    speed_vec(threshold_mask) = [];

                    %plot for sanity
                    figure(200)
                    subplot(2,1,1)
                    plot(phases, ankle_torque)

                    
                    % set up regressor matrices
                    [rows,cols] = size(phases);
                
                    A_mat = zeros(rows,numFuncs);
        
                    for ii = 1:rows
                        fourier_coeffs = returnFourierBasis_Eval(phases(ii),speed_vec(ii),incline_vec(ii),N_FOURIER);
                        A_mat(ii,:) = fourier_coeffs;
        
                    end
                    b_ankle_torque_mat = ankle_torque;
        
                    A_mat_master = [A_mat_master;A_mat];
                    b_ankle_torque_master = [b_ankle_torque_master;b_ankle_torque_mat];
    
            

                end

            end


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

            for k = 1:length(descent_ordered_step_strings)
                descent_ordered_step_string = descent_ordered_step_strings{k};

                %obtain actual declines present
                incline_descent_strings_act = fieldnames(sub_ramp_data.(descent_ordered_step_string));

                if any(strcmp(incline_descent_strings_act, incline_descent_string))
            
                    trial_data = sub_ramp_data.(descent_ordered_step_string).(incline_descent_string);
                    avg_speeds = trial_data.avgSpeed; %there'll be one speed per stride
    
                    %extract kinetics
                    kinetics = return_kinetics_gt( trial_data);
                    times = trial_data.time;
                
                    ankle_torque = -kinetics.ankle_torque * avg_subj_mass;
                    phases = kinetics.phases;
            
                    [N_strides, N_samples] = size(phases);
    
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

            
                    %reshape all the N_strides x N_samples matrixes to column vectors
                    phases = reshape(phases',[],1);
                    incline_vec = reshape(incline_vec',[],1);
                    ankle_torque = reshape(ankle_torque',[],1);

                    % generate speed vector               
                    speed_vec = repelem(avg_speeds, N_samples);

                    %threshold out ankle torques that were incorrectly
                    %recorded
                    threshold_mask = [];
                    if mean(abs(ankle_torque)) < 20
                        threshold_mask = 1:length(ankle_torque);
                    end
                    phases(threshold_mask) = [];
                    incline_vec(threshold_mask) = [];
                    ankle_torque(threshold_mask) = [];
                    speed_vec(threshold_mask) = [];

                    %plot for sanity
                    figure(300)
                    subplot(2,1,1)
                    plot(phases, ankle_torque)

                    % set up regressor matrices
                    [rows,cols] = size(phases);
                
                    A_mat = zeros(rows,numFuncs);
        
                    for ii = 1:rows
                        fourier_coeffs = returnFourierBasis_Eval(phases(ii),speed_vec(ii),incline_vec(ii),N_FOURIER);
                        A_mat(ii,:) = fourier_coeffs;
        
                    end
                    b_ankle_torque_mat = ankle_torque;
        
                    A_mat_master = [A_mat_master;A_mat];
                    b_ankle_torque_master = [b_ankle_torque_master;b_ankle_torque_mat];
    
            

                end

            end

        end
    
    end
end

% strip out nans

idx = any(isnan(A_mat_master),2);
A_mat_master(idx,:) = [];
b_ankle_torque_master(idx,:) = [];

%% Regress
% constraint to force constant behavior at sL = 0
phase = linspace(0,1,100)';
lenP = length(phase);


A_eq_fourier_constAtZerosL = [];
% ramps = linspace(-10,10,3);
ramps = [0,10,-10];
for ramp = ramps
    
    A_eq_fourier_ramp = zeros(length(phase), numFuncs);
    
    
    for j= 1:length(phase)
        
        phase_j = phase(j);
        
        fourier_deriv_coeffs = returnFourierBasis_Eval(phase_j,0,ramp, N_FOURIER);


        A_eq_fourier_ramp(j,:) = fourier_deriv_coeffs;

    end
    
    
    A_eq_fourier_constAtZerosL = [A_eq_fourier_constAtZerosL;A_eq_fourier_ramp];

end  

[rows,~] = size(A_eq_fourier_constAtZerosL);
b_eq_fourier_constAtZerosL = zeros(rows,1);


A_eq_ankle_torque = [A_eq_fourier_constAtZerosL;];

b_eq_ankle_torque = [b_eq_fourier_constAtZerosL;];

params_ankle_torque = lsqlin(A_mat_master,b_ankle_torque_master,[],[],A_eq_ankle_torque, b_eq_ankle_torque);

%% Plot against speed
speed = 0:0.1:2;
phase = linspace(0,1,200)';

[X,Y] = meshgrid(phase,speed);

best_fit_ankle_torque_mat_311 = zeros(size(X));
best_fit_ankle_torque_mat_312 = zeros(size(X));
best_fit_ankle_torque_mat_313 = zeros(size(X));

for i = 1:length(phase)
    
    for j = 1:length(speed)
        phase_i = X(j,i);
        speed_j = Y(j,i);
        best_fit_ankle_torque_mat_311(j,i) = params_ankle_torque' * returnFourierBasis_Eval(phase_i,speed_j,10, N_FOURIER)';
        best_fit_ankle_torque_mat_312(j,i) = params_ankle_torque' * returnFourierBasis_Eval(phase_i,speed_j,0, N_FOURIER)';
        best_fit_ankle_torque_mat_313(j,i) = params_ankle_torque' * returnFourierBasis_Eval(phase_i,speed_j,-10, N_FOURIER)';


    end
    
    
end

figure(2)
subplot(1,3,1)
surf(phase, speed,best_fit_ankle_torque_mat_311)
subplot(1,3,2)
surf(phase, speed,best_fit_ankle_torque_mat_312)
subplot(1,3,3)
surf(phase, speed,best_fit_ankle_torque_mat_313)


%% plot against incline

incline = -10:0.1:10;
speed_k = 1;


[X,Y] = meshgrid(phase,incline);

best_fit_ankle_torque_mat_incline = zeros(size(X));
for i = 1:length(phase)
    
    for j = 1:length(incline)
%         i
%         j
        phase_i = X(j,i);
        incline_j = Y(j,i);
        best_fit_ankle_torque_mat_incline(j,i) = params_ankle_torque' * returnFourierBasis_Eval(phase_i,speed_k,incline_j, N_FOURIER)';


    end
    
    
end


figure(5)
surf(phase, incline,best_fit_ankle_torque_mat_incline)
title('Ankle Torque')

xlabel('percent gait')
ylabel('Incline, scaled (deg)')
zlabel('degrees'); 
view(60,60)

%% save
M = [params_ankle_torque'];
writematrix(M,'torque_profile_coeffs.csv')

