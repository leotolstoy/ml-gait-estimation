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

numSpeedFuncs = 2;
numStairFuncs = 2;
N_FOURIER = 20;
numPhaseFuncs = (length(1:1:N_FOURIER) * 2) + 1;
numFuncs = numStairFuncs*numSpeedFuncs*numPhaseFuncs;

DO_STAIRS_ASCENT = true;
DO_STAIRS_DESCENT = true;

if DO_STAIRS_ASCENT
    for i = 1:length(sub_ids)
        sub_id = sub_ids(i)
        sub_stairs_data = GT_Dataset_Normalized.(sub_ids(i)).stair;
        
        % extract the different incline ascents
        stairs_ascent_strings = fieldnames(sub_stairs_data.s1_ascent);
%         incline_ascent_strings = {'i18'}


        %loop through incline ascents
        for j = 1:length(stairs_ascent_strings)

            stairs_ascent_string = stairs_ascent_strings{j};

            stair_height = return_stair_height_ascent_gt(stairs_ascent_string);
                        
            %only extract the steps that are on the ramp, which don't have
            %this issue where the phases from the other leg are out of sync
            ascent_ordered_step_strings = {'w2s_ascent',...
                's2_ascent',...
                's4_ascent',...
                's2w_ascent',...
                'a2_ascent',...
                's1_ascent',...
                's3_ascent',...
                's5_ascent',...
                'a1_ascent'};
            
            %filter out missing fields
            ascent_ordered_step_strings = intersect(ascent_ordered_step_strings,...
                fieldnames(sub_stairs_data),'stable');

            for k = 1:length(ascent_ordered_step_strings)
                ascent_ordered_step_string = ascent_ordered_step_strings{k};

                %obtain actual inclines present
                incline_ascent_strings_act = fieldnames(sub_stairs_data.(ascent_ordered_step_string));

                if any(strcmp(incline_ascent_strings_act, stairs_ascent_string))
            
                    trial_data = sub_stairs_data.(ascent_ordered_step_string).(stairs_ascent_string);
                    avg_speeds = trial_data.avgSpeed; %there'll be one speed per stride
    
                    %extract kinetics
                    kinetics = return_kinetics_gt( trial_data);
                    times = trial_data.time;
                
                    ankle_torque = -kinetics.ankle_torque * avg_subj_mass;
                    phases = kinetics.phases;

                    

                    [N_strides, N_samples] = size(phases);
    
    
                    %generate incline vector
                    incline_vec = 0 * ones(size(phases));
                    
                    %generate stairs vector
                    stair_height_vec = stair_height * ones(size(phases));
                    if strcmp(ascent_ordered_step_string,'w2s_ascent')
                        stair_height_vec = linspace(0,stair_height,N_samples);
                        stair_height_vec = repmat(stair_height_vec,N_strides,1);
    
                    elseif strcmp(ascent_ordered_step_string,'s2w_ascent')
                        stair_height_vec = linspace(stair_height,0,N_samples);
                        stair_height_vec = repmat(stair_height_vec,N_strides,1);
    
                    elseif strcmp(ascent_ordered_step_string,'a1_ascent') || ...
                        strcmp(ascent_ordered_step_string,'a2_ascent')
                        stair_height_vec = 0 * ones(size(phases));
                    end

            
                    %reshape all the N_strides x N_samples matrixes to column vectors
                    phases = reshape(phases',[],1);
                    incline_vec = reshape(incline_vec',[],1);
                    stair_height_vec = reshape(stair_height_vec',[],1);
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
                    stair_height_vec(threshold_mask) = [];
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
                        stairFuncs = returnBezierLinear(stair_height_vec(ii));
                        speedFuncs = returnBezierLinear(speed_vec(ii));
                        phaseFuncs = returnFourier(phases(ii), N_FOURIER);
                         
                        fourier_coeffs = kron(stairFuncs,kron( speedFuncs,  phaseFuncs));
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

if DO_STAIRS_DESCENT
    for i = 1:length(sub_ids)
        sub_id = sub_ids(i)       
        sub_stairs_data = GT_Dataset_Normalized.(sub_ids(i)).stair;
        
        % extract the different decline ascents
        stairs_descent_strings = fieldnames(sub_stairs_data.s1_descent);

        %loop through incline ascents
        for j = 1:length(stairs_descent_strings)

            stairs_descent_string = stairs_descent_strings{j};      

            stair_height = return_stair_height_descent_gt(stairs_descent_string);
            
            %only extract the steps that are on the ramp, which don't have
            %this issue where the phases from the other leg are out of sync
            descent_ordered_step_strings = {'w2s_descent',...
                's2_descent',...
                's4_descent',...
                's2w_descent',...
                'a2_descent',...
                's1_descent',...
                's3_descent',...
                's5_descent',...
                'a1_descent'};
            
            %filter out missing fields
            descent_ordered_step_strings = intersect(descent_ordered_step_strings,...
                fieldnames(sub_stairs_data),'stable');

            for k = 1:length(descent_ordered_step_strings)
                descent_ordered_step_string = descent_ordered_step_strings{k};

                %obtain actual declines present
                incline_descent_strings_act = fieldnames(sub_stairs_data.(descent_ordered_step_string));

                if any(strcmp(incline_descent_strings_act, stairs_descent_string))
            
                    trial_data = sub_stairs_data.(descent_ordered_step_string).(stairs_descent_string);
                    avg_speeds = trial_data.avgSpeed; %there'll be one speed per stride
    
                    %extract kinetics
                    kinetics = return_kinetics_gt( trial_data);
                    times = trial_data.time;
                
                    ankle_torque = -kinetics.ankle_torque * avg_subj_mass;
                    phases = kinetics.phases;
            
                    [N_strides, N_samples] = size(phases);
    
                    %generate incline vector
                    incline_vec = 0 * ones(size(phases));

                    %generate stairs vector
                    stair_height_vec = stair_height * ones(size(phases));

                    if strcmp(descent_ordered_step_string,'w2s_descent')
                        stair_height_vec = linspace(0,stair_height,N_samples);
                        stair_height_vec = repmat(stair_height_vec,N_strides,1);
    
                    elseif strcmp(descent_ordered_step_string,'s2w_descent')
                        stair_height_vec = linspace(stair_height,0,N_samples);
                        stair_height_vec = repmat(stair_height_vec,N_strides,1);
    
                    elseif strcmp(descent_ordered_step_string,'a1_descent') || ...
                        strcmp(descent_ordered_step_string,'a2_descent')
                        stair_height_vec = 0 * ones(size(phases));
                    end

            
                    %reshape all the N_strides x N_samples matrixes to column vectors
                    phases = reshape(phases',[],1);
                    incline_vec = reshape(incline_vec',[],1);
                    stair_height_vec = reshape(stair_height_vec',[],1);
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
                    stair_height_vec(threshold_mask) = [];
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
                        stairFuncs = returnBezierLinear(stair_height_vec(ii));
                        speedFuncs = returnBezierLinear(speed_vec(ii));
                        phaseFuncs = returnFourier(phases(ii), N_FOURIER);
                         
                        fourier_coeffs = kron(stairFuncs,kron( speedFuncs,  phaseFuncs));
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

stair_heights = [0,0.2,-0.2];

for stair_height = stair_heights
    
    A_eq_fourier_stair_height = zeros(length(phase), numFuncs);
    
    
    for j= 1:length(phase)
        
        phase_j = phase(j);
        stairFuncs = returnBezierLinear(stair_height);
        speedFuncs = returnBezierLinear(0);
        phaseFuncs = returnFourier(phase_j, N_FOURIER);
        
        fourier_coeffs = kron( stairFuncs, kron( speedFuncs,  phaseFuncs));


        A_eq_fourier_stair_height(j,:) = fourier_coeffs;

    end
    
    
    A_eq_fourier_constAtZerosL = [A_eq_fourier_constAtZerosL;A_eq_fourier_stair_height];

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
        speedFuncs = returnBezierLinear(speed_j);
        phaseFuncs = returnFourier(phase_i, N_FOURIER);

           
        stairFuncs = returnBezierLinear(-0.2);
        fourier_coeffs = kron( stairFuncs, kron( speedFuncs,  phaseFuncs));
        best_fit_ankle_torque_mat_311(j,i) = params_ankle_torque' * fourier_coeffs';


        stairFuncs = returnBezierLinear(0);
        fourier_coeffs = kron( stairFuncs, kron( speedFuncs,  phaseFuncs));
        best_fit_ankle_torque_mat_312(j,i) = params_ankle_torque' * fourier_coeffs';


        stairFuncs = returnBezierLinear(0.2);
        fourier_coeffs = kron( stairFuncs, kron( speedFuncs,  phaseFuncs));
        best_fit_ankle_torque_mat_313(j,i) = params_ankle_torque' * fourier_coeffs';


    end
    
    
end

figure(2)
subplot(1,3,1)
surf(phase, speed,best_fit_ankle_torque_mat_311)
xlabel('phase')
ylabel('speed')

subplot(1,3,2)
surf(phase, speed,best_fit_ankle_torque_mat_312)
xlabel('phase')
ylabel('speed')

subplot(1,3,3)
surf(phase, speed,best_fit_ankle_torque_mat_313)
xlabel('phase')
ylabel('speed')


%% plot against stair_height

stair_height = -0.2:0.1:0.2;
speed = 1;


[X,Y] = meshgrid(phase,stair_height);

best_fit_ankle_torque_mat_stair_height = zeros(size(X));

for i = 1:length(phase)
    
    for j = 1:length(stair_height)
%         i
%         j
        phase_i = X(j,i);
        stair_height_j = Y(j,i);

        stairFuncs = returnBezierLinear(stair_height_j);
        speedFuncs = returnBezierLinear(speed);
        phaseFuncs = returnFourier(phase_i, N_FOURIER);
        fourier_coeffs = kron(stairFuncs, kron( speedFuncs,  phaseFuncs));

        best_fit_ankle_torque_mat_stair_height(j,i) = params_ankle_torque' * fourier_coeffs';


    end
    
    
end



figure(5)
surf(phase, stair_height,best_fit_ankle_torque_mat_stair_height)
title('Foot Angle')

xlabel('percent gait')
ylabel('stair height (m)')
zlabel('torque (Nm)'); 
view(60,60)


%% save
M = [params_ankle_torque'];
writematrix(M,'torque_profile_stairs_coeffs.csv')

