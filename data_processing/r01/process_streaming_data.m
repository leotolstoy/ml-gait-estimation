close all;clc;clearvars -except Normalized Streaming %removes all variables except for gaitcyle and continuous from the matlab workspace. Useful because loading these variables can take several minutes.
%load the InclineExperiment data from the folder location you specify
% load('Z:\your_file_location_here\InclineExperiment.mat') 
%This file is large and may take several minutes to load. We recommend not
%clearing the variable, and only loading it as many times as necessary.

%% Example: removing strides with outliers from kinematic data, and taking the mean

sub={'AB01','AB02','AB03','AB04','AB05','AB06','AB07','AB08','AB09','AB10'};

sub={'AB01'};


phase_vec = linspace(0,1,150)';
N_phase = length(phase_vec);

% figure(1)
% subplot(4,1,1)
% hold on
% subplot(4,1,2)
% hold on
% subplot(4,1,3)
% hold on
% subplot(4,1,4)
% hold on
% 
% figure(4)
% subplot(2,1,1)
% hold on
% subplot(2,1,2)
% hold on
% 
% figure(5)
% subplot(2,1,1)
% hold on
% subplot(2,1,2)
% hold on

SAMPLE_RATE_VICON_NOMINAL = 100;
DT_VICON_NOMINAL = 1/SAMPLE_RATE_VICON_NOMINAL;
LEFT_LEG_NUM = 1;
RIGHT_LEG_NUM = 2;



%four measurements
dim_input = 4;

% the four traditional gait states
% speed
% stairs
% stop
dim_output = 4 + 1 + 1 + 1;


stride_count_master = 0;
data_matrix_master = [];
for i = 1:length(sub) %loop through all subjects in 'sub'
    sub{i}

    %% PROCESS INCLINE
    %initialize matices    
    incline_fieldnames = fieldnames(Streaming.(sub{i}).Tread);
    
    %handle edge cases
    %AB08 lacks VelProf at i10, so skip 
    if strcmp(sub{i},'AB08')
        incline_fieldnames = setdiff(incline_fieldnames,{'i10'});
    end
    
    %AB10 lacks VelProf at i0, so skip 
    if  strcmp(sub{i},'AB10')
        incline_fieldnames = setdiff(incline_fieldnames,{'i0'});
    end
%     incline_fieldnames = {'d10'};

    
    for k = 1:length(incline_fieldnames)
        [treadmill_incline] = return_incline_streaming(incline_fieldnames{k});

        %extract treadmill speed + event markers
        cutPoints = Streaming.(sub{i}).Tread.(incline_fieldnames{k}).events.cutPoints;
        
        [start_cutPoints_ordered, cutPoints_order_idx] = sort(cutPoints(:,1));
        end_cutPoints_ordered = cutPoints(cutPoints_order_idx,2);
        tasks = Streaming.(sub{i}).Tread.(incline_fieldnames{k}).events.tasks;

        tasks_ordered = tasks(cutPoints_order_idx);
        treadmill_speed = Streaming.(sub{i}).Tread.(incline_fieldnames{k}).events.VelProf.cvel;

        %process left side
        L_foot_angle_data = Streaming.(sub{i}).Tread.(incline_fieldnames{k}).jointAngles.LFootProgressAngles(:,1);
        L_ankle_angle_data = Streaming.(sub{i}).Tread.(incline_fieldnames{k}).jointAngles.LAnkleAngles(:,1);
        
        L_foot_angle_data = processFootAngles(sub{i},incline_fieldnames{k},L_foot_angle_data);
        L_ankle_angle_data = processAnkleAngles(L_ankle_angle_data);
        L_shank_angle_data = L_foot_angle_data - L_ankle_angle_data;

        L_heel_pos_forward = Streaming.(sub{i}).Tread.(incline_fieldnames{k}).markers.LHEE(:,2);
        L_heel_pos_upward = Streaming.(sub{i}).Tread.(incline_fieldnames{k}).markers.LHEE(:,3);

        %negate forward heel pos if the incline is negative, since
        %otherwise (due to the world frame) the position is backwards
        %relative to forward motion

        if treadmill_incline < 0
            L_heel_pos_forward = -L_heel_pos_forward;
        end

        L_HS = Streaming.(sub{i}).Tread.(incline_fieldnames{k}).events.LHS;

        [processed_struct_L] = analyze_kinematics(L_foot_angle_data,L_shank_angle_data,...
            L_heel_pos_forward, L_heel_pos_upward,...
            L_HS, treadmill_speed, treadmill_incline, stride_count_master, ...
            LEFT_LEG_NUM, SAMPLE_RATE_VICON_NOMINAL);
        

        
        L_foot_angle_data = processed_struct_L.foot_angle_data;
        L_foot_vel_angle_data = processed_struct_L.foot_vel_angle_data;
        L_shank_angle_data = processed_struct_L.shank_angle_data;
        L_shank_vel_angle_data = processed_struct_L.shank_vel_angle_data;
        L_time_vec = processed_struct_L.time_vec;
        L_phase = processed_struct_L.phase;
        L_speed = processed_struct_L.speed;
        L_incline = processed_struct_L.incline;
        L_is_moving = processed_struct_L.is_moving;
        L_stride_count = processed_struct_L.stride_count;
        L_stride_count_vec = processed_struct_L.stride_count_vec;
        L_leg_side_num_vec = processed_struct_L.leg_side_num_vec;
        L_HSDetected = processed_struct_L.HSDetected;
        
        L_heel_pos_forward = processed_struct_L.heel_pos_forward;
        L_heel_vel_forward = processed_struct_L.heel_vel_forward;
        L_heel_acc_forward = processed_struct_L.heel_acc_forward;
        L_heel_pos_upward = processed_struct_L.heel_pos_upward;
        L_heel_vel_upward = processed_struct_L.heel_vel_upward;
        L_heel_acc_upward = processed_struct_L.heel_acc_upward;

        stride_count_master = L_stride_count;

        
        %process right side
        R_foot_angle_data = Streaming.(sub{i}).Tread.(incline_fieldnames{k}).jointAngles.RFootProgressAngles(:,1);
        R_ankle_angle_data = Streaming.(sub{i}).Tread.(incline_fieldnames{k}).jointAngles.RAnkleAngles(:,1);
        
        R_foot_angle_data = processFootAngles(sub{i},incline_fieldnames{k},R_foot_angle_data);
        R_ankle_angle_data = processAnkleAngles(R_ankle_angle_data);
        R_shank_angle_data = R_foot_angle_data - R_ankle_angle_data;

        R_heel_pos_forward = Streaming.(sub{i}).Tread.(incline_fieldnames{k}).markers.RHEE(:,2);
        R_heel_pos_upward = Streaming.(sub{i}).Tread.(incline_fieldnames{k}).markers.RHEE(:,3);

        if treadmill_incline < 0
            R_heel_pos_forward = -R_heel_pos_forward;
        end

        R_HS = Streaming.(sub{i}).Tread.(incline_fieldnames{k}).events.RHS;

        [processed_struct_R] = analyze_kinematics(R_foot_angle_data,R_shank_angle_data,...
            R_heel_pos_forward, R_heel_pos_upward,...
            R_HS, treadmill_speed, treadmill_incline, stride_count_master,...
            RIGHT_LEG_NUM, SAMPLE_RATE_VICON_NOMINAL);

        R_foot_angle_data = processed_struct_R.foot_angle_data;
        R_foot_vel_angle_data = processed_struct_R.foot_vel_angle_data;
        R_shank_angle_data = processed_struct_R.shank_angle_data;
        R_shank_vel_angle_data = processed_struct_R.shank_vel_angle_data;
        R_time_vec = processed_struct_R.time_vec;
        R_phase = processed_struct_R.phase;
        R_speed = processed_struct_R.speed;
        R_incline = processed_struct_R.incline;
        R_is_moving = processed_struct_R.is_moving;
        R_stride_count = processed_struct_R.stride_count;
        R_stride_count_vec = processed_struct_R.stride_count_vec;
        R_leg_side_num_vec = processed_struct_R.leg_side_num_vec;
        R_HSDetected = processed_struct_R.HSDetected;

        R_heel_pos_forward = processed_struct_R.heel_pos_forward;
        R_heel_vel_forward = processed_struct_R.heel_vel_forward;
        R_heel_acc_forward = processed_struct_R.heel_acc_forward;
        R_heel_pos_upward = processed_struct_R.heel_pos_upward;
        R_heel_vel_upward = processed_struct_R.heel_vel_upward;
        R_heel_acc_upward = processed_struct_R.heel_acc_upward;

        stride_count_master = R_stride_count;

        

        %concatenate left and right sides
        foot_angle_data = [L_foot_angle_data;R_foot_angle_data];
        foot_vel_angle_data = [L_foot_vel_angle_data;R_foot_vel_angle_data];
        shank_angle_data = [L_shank_angle_data;R_shank_angle_data];
        shank_vel_angle_data = [L_shank_vel_angle_data;R_shank_vel_angle_data];

        heel_pos_forward = [L_heel_pos_forward;R_heel_pos_forward];
        heel_pos_upward = [L_heel_pos_upward;R_heel_pos_upward];
        heel_acc_forward = [L_heel_acc_forward;R_heel_acc_forward];
        heel_acc_upward = [L_heel_acc_upward;R_heel_acc_upward];

        time_vec = [L_time_vec;R_time_vec+L_time_vec(end)];
        phase = [L_phase;R_phase];
        speed = [L_speed;R_speed];
        incline = [L_incline;R_incline];
        is_moving = [L_is_moving;R_is_moving];
        stride_count_vec = [L_stride_count_vec;R_stride_count_vec];
        HSDetected = [L_HSDetected;R_HSDetected];


        if strcmp(sub{i},'AB01')
            figure(1)
            subplot(4,1,1)
            plot(time_vec,foot_angle_data)
            hold on
             
    %         plot(time_vec,foot_vel_angle_data*1e-1)
            plot(time_vec,shank_angle_data)
    %         plot(time_vec,is_moving*10,'k','LineWidth',2)
            plot(time_vec,HSDetected*10,'k')
            
            subplot(4,1,2)
            plot(time_vec,speed,'b','LineWidth',2)
            hold on
            plot(time_vec,HSDetected*1,'k') 
    
            subplot(4,1,3)
            plot(time_vec,phase)
            hold on
            plot(time_vec,HSDetected*1,'k') 
            
            subplot(4,1,4)
            plot(time_vec,shank_vel_angle_data)
            hold on
            plot(time_vec,HSDetected*100,'k') 

            figure(2)
            subplot(2,1,1)
            plot(time_vec,heel_pos_forward)
            hold on
            plot(time_vec,HSDetected*0.5,'k')
            ylabel('heel pos forward')
            subplot(2,1,2)
            plot(time_vec,heel_pos_upward)
            hold on
            plot(time_vec,HSDetected*0.5,'k')
            ylabel('heel pos upward')

            figure(3)
            subplot(2,1,1)
            plot(time_vec,heel_acc_forward)
            hold on
            plot(time_vec,HSDetected*10,'k')
            ylabel('heel acc forward')

            subplot(2,1,2)
            plot(time_vec,heel_acc_upward)
            hold on
            plot(time_vec,HSDetected*10,'k')
            ylabel('heel acc upward')


            figure(4)
            hold on
            plot(phase(1:end), foot_angle_data(1:end),'o')

            figure(5)
            subplot(2,1,1)
            plot(phase,heel_pos_forward,'.')
            hold on
            ylabel('heel pos forward')
            subplot(2,1,2)
            plot(phase,heel_pos_upward,'.')
            ylabel('heel pos upward')

            figure(6)
            subplot(2,1,1)
            plot(phase,heel_acc_forward,'.')
            hold on
            ylabel('heel acc forward')

            subplot(2,1,2)
            plot(phase,heel_acc_upward,'.')
            ylabel('heel acc upward')


        end


        %update master matrices
        subj_id_master = i*ones(size(foot_angle_data));

        %set is_stairs vector to all zeros
        is_stairs_master = zeros(size(foot_angle_data));

        data_matrix = [foot_angle_data,...
                    foot_vel_angle_data,...
                    shank_angle_data,...
                    shank_vel_angle_data,...
                    heel_acc_forward,...
                    heel_acc_upward,...
                    phase,...
                    speed,...
                    incline,...
                    is_stairs_master,...
                    is_moving,...
                    stride_count_vec,...
                    HSDetected,...
                    subj_id_master];

        data_matrix_master = [data_matrix_master;data_matrix];

        
    end

    %% PROCESS STAIRS
    stair_fieldnames = fieldnames(Streaming.(sub{i}).Stair);
    for k = 1:length(stair_fieldnames)
        stair_fieldnames{k}
        incline_id = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).events.newID

        [treadmill_incline, stair_height] = return_incline_stair_height_streaming(incline_id)
        %set stair incline to zero
%         treadmill_incline = 0;

        %extract Stairmill speed + event markers
        cutPoints_cell_array = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).events.cutPoints;
        
        %extract the actual cutpoints from the cutPoints cell array
        [row, col] = size(cutPoints_cell_array);
        cutPoints = zeros(row, 2);
            
        L_stair_events = {};
        R_stair_events = {};

        for ii = 1:row
            if ~isempty(cutPoints_cell_array{ii,3})
                if strcmp('L',cutPoints_cell_array{ii,2})
                    L_stair_events = [L_stair_events;{cutPoints_cell_array{ii,1},cutPoints_cell_array{ii,3}}];
                elseif strcmp('R',cutPoints_cell_array{ii,2})
                    R_stair_events = [R_stair_events;{cutPoints_cell_array{ii,1},cutPoints_cell_array{ii,3}}];

                end
            end
            
        end

        % exclude certain subjects and trials
        if (strcmp(sub{i}, 'AB02') && strcmp(stair_fieldnames{k}, 's35dg_08')) ||...
                (strcmp(sub{i}, 'AB06') && strcmp(stair_fieldnames{k}, 's30dg_12'))

        else
            %calculate self-selected speed by taking initial and end position
            %of the right heel marker and dividing by the time elapsed
            R_com_pos_forward = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).markers.RPSI(:,2);
            R_com_pos_upward = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).markers.RPSI(:,3);
    
            %for some reason, some of the R_com_pos_forward vicon data are
             %zero, so remove them
            R_com_pos_forward_is_zero = R_com_pos_forward == 0;
            R_com_pos_forward(R_com_pos_forward_is_zero) = [];

            dist_raw = R_com_pos_forward(end) - R_com_pos_forward(1);
            total_dist = abs(dist_raw);
            total_time_stairs = length(R_com_pos_forward)/DT_VICON_NOMINAL;

            R_com_pos_upward(R_com_pos_forward_is_zero) = [];

            %numericall differentiate to obtain treadmill speed
            treadmill_speed = abs(diff(R_com_pos_forward))./DT_VICON_NOMINAL;
%             treadmill_speed = [treadmill_speed(1);treadmill_speed];

            

          

            figure(100)
            plot(R_com_pos_forward)
            hold on
            plot(diff(R_com_pos_forward)./DT_VICON_NOMINAL)
            
            plot(R_com_pos_upward)

            plot(treadmill_speed)
            legend('R_com_pos_forward', 'num deriv R_com_pos_forward','R_com_pos_upward','treadmill_speed')

            %take average
            treadmill_speed = mean(treadmill_speed);

            %set up boolean to negate the forward heel positions to account
            %for the direction the person was moving through the stairs
            negate_heel_pos_forward = false;
            raw_avg_speed = mean(diff(R_com_pos_forward)./DT_VICON_NOMINAL)
            if raw_avg_speed < 0
                negate_heel_pos_forward = true;
            end

    
            %process left side
            
            L_foot_angle_data = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).jointAngles.LFootProgressAngles(:,1);
            L_ankle_angle_data = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).jointAngles.LAnkleAngles(:,1);
            L_heel_pos_forward = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).markers.LHEE(:,2);
            L_heel_pos_upward = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).markers.LHEE(:,3);
            
            %change direction of forward heel pos to ensure that forward
            %motion resembles the forward motion of treadmilll walking
            if negate_heel_pos_forward
                L_heel_pos_forward = -L_heel_pos_forward;
            end

    
            %for some reason, some of the raw vicon foot angles are zero, so
            %remove them
            L_foot_angle_is_zero = L_foot_angle_data == 0;
            L_foot_angle_data(L_foot_angle_is_zero) = [];
            L_ankle_angle_data(L_foot_angle_is_zero) = [];
            L_heel_pos_forward(L_foot_angle_is_zero) = [];
            L_heel_pos_upward(L_foot_angle_is_zero) = [];
                    
            %re-zero foot and ankle angles
            L_foot_angle_data = processFootAngles(sub{i},stair_fieldnames{k},L_foot_angle_data);
            L_ankle_angle_data = processAnkleAngles(L_ankle_angle_data);
            L_shank_angle_data = L_foot_angle_data - L_ankle_angle_data;
            
            %extract raw HS indexes
            L_HS = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).events.LHS;
    
            %remove HS idxs that occurred when the foot angles were zero
            idxs_L_foot_angle_nonzero = find(~L_foot_angle_is_zero);
            L_HS(L_HS < idxs_L_foot_angle_nonzero(1)) = [];

            %strip out stair events that have HSs that occurred before foot
            %angles were nonzero
            [row, ~] = size(L_stair_events);
            idxs_to_remove = [];
            for ii = 1:row
                HS_idxs = L_stair_events{ii,2};
                if HS_idxs(1) < idxs_L_foot_angle_nonzero(1)
                    idxs_to_remove = [idxs_to_remove;ii];
                    
                end
            end
            L_stair_events(idxs_to_remove,:) = [];
            
            %adjust HS and stair events to account for missing frames from the removal of the
            %zero foot angles at beginning of data vector
            L_HS = L_HS - idxs_L_foot_angle_nonzero(1) + 1;
            [row, ~] = size(L_stair_events);
            for ii = 1:row
                L_stair_events{ii,2} = L_stair_events{ii,2} - idxs_L_foot_angle_nonzero(1) + 1;
            end
    
            [processed_struct_L] = analyze_stair_kinematics(L_foot_angle_data,L_shank_angle_data,...
                L_heel_pos_forward, L_heel_pos_upward,...
                L_HS,...
                treadmill_speed, treadmill_incline, stair_height, stride_count_master, L_stair_events, LEFT_LEG_NUM, SAMPLE_RATE_VICON_NOMINAL);
            
    
            
            L_foot_angle_data = processed_struct_L.foot_angle_data;
            L_foot_vel_angle_data = processed_struct_L.foot_vel_angle_data;
            L_shank_angle_data = processed_struct_L.shank_angle_data;
            L_shank_vel_angle_data = processed_struct_L.shank_vel_angle_data;
            L_time_vec = processed_struct_L.time_vec;
            L_phase = processed_struct_L.phase;
            L_speed = processed_struct_L.speed;
            L_incline = processed_struct_L.incline;
            L_is_moving = processed_struct_L.is_moving;
            L_stride_count = processed_struct_L.stride_count;
            L_stride_count_vec = processed_struct_L.stride_count_vec;
            L_leg_side_num_vec = processed_struct_L.leg_side_num_vec;
            L_HSDetected = processed_struct_L.HSDetected;
            L_heel_pos_forward = processed_struct_L.heel_pos_forward;
            L_heel_vel_forward = processed_struct_L.heel_vel_forward;
            L_heel_acc_forward = processed_struct_L.heel_acc_forward;
            L_heel_pos_upward = processed_struct_L.heel_pos_upward;
            L_heel_vel_upward = processed_struct_L.heel_vel_upward;
            L_heel_acc_upward = processed_struct_L.heel_acc_upward;
            L_is_stairs = processed_struct_L.is_stairs;
    
            stride_count_master = L_stride_count;
    
            %process right side
            R_foot_angle_data = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).jointAngles.RFootProgressAngles(:,1);
            R_ankle_angle_data = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).jointAngles.RAnkleAngles(:,1);
            R_heel_pos_forward = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).markers.RHEE(:,2);
            R_heel_pos_upward = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).markers.RHEE(:,3);

            %remove indices where the foot angle is zero
            R_foot_angle_is_zero = R_foot_angle_data == 0;
            R_foot_angle_data(R_foot_angle_is_zero) = [];
            R_ankle_angle_data(R_foot_angle_is_zero) = [];
            R_heel_pos_forward(R_foot_angle_is_zero) = [];
            R_heel_pos_upward(R_foot_angle_is_zero) = [];

            if negate_heel_pos_forward
                R_heel_pos_forward = -R_heel_pos_forward;
            end

    
            R_foot_angle_data = processFootAngles(sub{i},stair_fieldnames{k},R_foot_angle_data);
            R_ankle_angle_data = processAnkleAngles(R_ankle_angle_data);
            R_shank_angle_data = R_foot_angle_data - R_ankle_angle_data;
    
            R_HS = Streaming.(sub{i}).Stair.(stair_fieldnames{k}).events.RHS;
    
            %remove HS idxs that occurred when the foot angles were zero
            idxs_R_foot_angle_nonzero = find(~R_foot_angle_is_zero);
            R_HS(R_HS < idxs_R_foot_angle_nonzero(1)) = [];
            
            %strip out stair events that have HSs that occurred before foot
            %angles were nonzero
            [row, ~] = size(R_stair_events);
            idxs_to_remove = [];
            for ii = 1:row
                HS_idxs = R_stair_events{ii,2};
                if HS_idxs(1) < idxs_R_foot_angle_nonzero(1)
                    idxs_to_remove = [idxs_to_remove;ii];
                    
                end
            end
            R_stair_events(idxs_to_remove,:) = [];
            
            %adjust HS and stair events to account for missing frames from the removal of the
            %zero foot angles at beginning of data vector
            R_HS = R_HS - idxs_R_foot_angle_nonzero(1) + 1;
            [row, ~] = size(R_stair_events);
            for ii = 1:row
                R_stair_events{ii,2} = R_stair_events{ii,2} - idxs_R_foot_angle_nonzero(1) + 1;
            end


    
            [processed_struct_R] = analyze_stair_kinematics(R_foot_angle_data,R_shank_angle_data,...
                R_heel_pos_forward, R_heel_pos_upward,...
                R_HS,...
                treadmill_speed, treadmill_incline, stair_height, stride_count_master, R_stair_events, RIGHT_LEG_NUM, SAMPLE_RATE_VICON_NOMINAL);
    
            R_foot_angle_data = processed_struct_R.foot_angle_data;
            R_foot_vel_angle_data = processed_struct_R.foot_vel_angle_data;
            R_shank_angle_data = processed_struct_R.shank_angle_data;
            R_shank_vel_angle_data = processed_struct_R.shank_vel_angle_data;
            R_time_vec = processed_struct_R.time_vec;
            R_phase = processed_struct_R.phase;
            R_speed = processed_struct_R.speed;
            R_incline = processed_struct_R.incline;
            R_is_moving = processed_struct_R.is_moving;
            R_stride_count = processed_struct_R.stride_count;
            R_stride_count_vec = processed_struct_R.stride_count_vec;
            R_leg_side_num_vec = processed_struct_R.leg_side_num_vec;
            R_HSDetected = processed_struct_R.HSDetected;
            R_heel_pos_forward = processed_struct_R.heel_pos_forward;
            R_heel_vel_forward = processed_struct_R.heel_vel_forward;
            R_heel_acc_forward = processed_struct_R.heel_acc_forward;
            R_heel_pos_upward = processed_struct_R.heel_pos_upward;
            R_heel_vel_upward = processed_struct_R.heel_vel_upward;
            R_heel_acc_upward = processed_struct_R.heel_acc_upward;
            R_is_stairs = processed_struct_R.is_stairs;
    
            stride_count_master = R_stride_count;
    
            
    
            %concatenate left and right sides
            foot_angle_data = [L_foot_angle_data;R_foot_angle_data];
            foot_vel_angle_data = [L_foot_vel_angle_data;R_foot_vel_angle_data];
            shank_angle_data = [L_shank_angle_data;R_shank_angle_data];
            shank_vel_angle_data = [L_shank_vel_angle_data;R_shank_vel_angle_data];

            heel_pos_forward = [L_heel_pos_forward;R_heel_pos_forward];
            heel_pos_upward = [L_heel_pos_upward;R_heel_pos_upward];

            heel_acc_forward = [L_heel_acc_forward;R_heel_acc_forward];
            heel_acc_upward = [L_heel_acc_upward;R_heel_acc_upward];


            time_vec = [L_time_vec;R_time_vec+L_time_vec(end)];
            phase = [L_phase;R_phase];
            speed = [L_speed;R_speed];
            incline = [L_incline;R_incline];
            is_moving = [L_is_moving;R_is_moving];
            is_stairs = [L_is_stairs;R_is_stairs];
            stride_count_vec = [L_stride_count_vec;R_stride_count_vec];
            HSDetected = [L_HSDetected;R_HSDetected];
    
    
            if strcmp(sub{i},'AB01')% || true
                figure(101)
                subplot(4,1,1)
                plot(time_vec,foot_angle_data,'b','LineWidth',2)
                hold on
                 
        %         plot(time_vec,foot_vel_angle_data*1e-1)
    %             plot(time_vec,shank_angle_data)
        %         plot(time_vec,is_moving*10,'k','LineWidth',2)
    %             plot(time_vec,HSDetected*10,'k')
                legend('foot angles')
                
                subplot(4,1,2)
                plot(time_vec,is_stairs,'b','LineWidth',2)
                hold on
                plot(time_vec,HSDetected*1,'k') 
    
                legend('is_stairs')
        
                subplot(4,1,3)
                plot(time_vec,phase)
                hold on
                plot(time_vec,HSDetected*1,'k') 
                legend('phase')
                
                subplot(4,1,4)
                plot(time_vec,speed)
                hold on
                plot(time_vec,HSDetected*1,'k') 

                figure(102)
                subplot(2,1,1)
                plot(time_vec,heel_pos_forward,'.')
                hold on
                plot(time_vec,HSDetected*0.5,'k')
                ylabel('heel pos forward')
    
                subplot(2,1,2)
                plot(time_vec,heel_pos_upward,'.')
                hold on
                plot(time_vec,HSDetected*0.5,'k')
                ylabel('heel pos upward')


                figure(103)
                subplot(2,1,1)
                plot(time_vec,heel_acc_forward)
                hold on
                plot(time_vec,HSDetected*10,'k')
                ylabel('heel acc forward')
    
                subplot(2,1,2)
                plot(time_vec,heel_acc_upward)
                hold on
                plot(time_vec,HSDetected*10,'k')
                ylabel('heel acc upward')


                figure(105)
                subplot(2,1,1)
                plot(phase,heel_pos_forward,'.')
                hold on
                ylabel('heel pos forward')
                subplot(2,1,2)
                plot(phase,heel_pos_upward,'.')
                ylabel('heel pos upward')
    
                figure(106)
                subplot(2,1,1)
                plot(phase,heel_acc_forward,'.')
                hold on
                ylabel('heel acc forward')
    
                subplot(2,1,2)
                plot(phase,heel_acc_upward,'.')
                ylabel('heel acc upward')
    
            end
    
    
            %update master matrices
            subj_id_master = i*ones(size(foot_angle_data));
    
            data_matrix_stairs = [foot_angle_data,...
                        foot_vel_angle_data,...
                        shank_angle_data,...
                        shank_vel_angle_data,...
                        heel_acc_forward,...
                        heel_acc_upward,...
                        phase,...
                        speed,...
                        incline,...
                        is_stairs,...
                        is_moving,...
                        stride_count_vec,...
                        HSDetected,...
                        subj_id_master];
    
            data_matrix_master = [data_matrix_master;data_matrix_stairs];
        end

        
    end
    
    

end

%generate times and time steps
[rows,~] = size(data_matrix_master);
dt = 1/SAMPLE_RATE_VICON_NOMINAL;
dts = dt*ones(rows,1);
time = cumsum(dts);
% time = time - time(1);

data_matrix_master = [data_matrix_master,dts,time];

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

table_filename = 'r01_ordered_stairs_label.csv';
writetable(data_table_master,table_filename)













