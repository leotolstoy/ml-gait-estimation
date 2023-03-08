function kinetics = return_kinetics_gt(data_struct)


    ankle_torque = data_struct.jointMoments.ankle_moment;
    times = data_struct.time;
    
    %some ankle torques are nan, force those to be zero
    ankle_torque(isnan(ankle_torque)) = 0;

    %calculate phases by dividing the time vector by the total duration
    phases = times./times(:,end);

    kinetics.ankle_torque = ankle_torque;
    kinetics.phases = phases;


end








