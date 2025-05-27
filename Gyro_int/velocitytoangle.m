% 修改文件路径为相对路径
data = readtable('velocitytoangle.xls','Sheet',1); 
% fs = 100;          % 采样频率 (Hz)，未使用，已注释
t = table2array(data(2:end,1));  % 时间向量 (1秒)

x=table2array(data(2:end,2));
y=table2array(data(2:end,3));
z=table2array(data(2:end,4));

%%四元数求解
q=zeros(size(x,1),4);
q(1,:)=[1,0,0,0];
angle=zeros(size(x,1),3);
% 初始化第一个角度值
angle(1,:)=[0,0,0];
for i=2:size(x,1)
    q_temp=(t(i)-t(i-1))/2.*[2/(t(i)-t(i-1)) -x(i-1) -y(i-1) -z(i-1);
                          x(i-1) 2/(t(i)-t(i-1)) z(i-1) -y(i-1);
                          y(i-1) -z(i-1) 2/(t(i)-t(i-1)) x(i-1);
                          z(i-1) y(i-1) -x(i-1) 2/(t(i)-t(i-1))]*q(i-1,:)';
    % 四元数归一化
    q_temp = q_temp / norm(q_temp);
    q(i,:)=q_temp';

    yaw = atan2(2*(q_temp(1)*q_temp(4) + q_temp(2)*q_temp(3)), 1 - 2*(q_temp(3)^2 + q_temp(4)^2));
    pitch = asin(2*(q_temp(1)*q_temp(3) - q_temp(4)*q_temp(2)));
    roll = atan2(2*(q_temp(1)*q_temp(2) + q_temp(3)*q_temp(4)), 1 - 2*(q_temp(2)^2 + q_temp(3)^2));
    angle(i,:)=[rad2deg(yaw),rad2deg(pitch),rad2deg(roll)];
end


%%
figure;
plot(t,angle(:,1),"Color",'r');
hold on
plot(t,angle(:,2),"Color",'g');
plot(t,angle(:,3),"Color",'b');
legend('Yaw', 'Pitch', 'Roll');
xlabel('Time (s)');
ylabel('Angle (degrees)');
title('Gyro Integration Results');
grid on;


