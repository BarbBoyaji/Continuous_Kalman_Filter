clear all
close all

% some plotting defaults
blue =[0, 0.4470, 0.7410];
red = [0.8500, 0.3250, 0.0980];
orange = [0.9290, 0.6940, 0.1250];

%%%%% CONSTANTS %%%%%
tau= 2; %sec
tF = 10; %sec
R1 = 15e-6; %rad^2sec
R2 = 1.67e-3; %rad^2sec^3
b  = 1.52e-2;
Vc = 300; %ft/sec
E_a2T=100^2; %(ft/sec^2)^2
sigma_vel = 200;
sigma_ac = 100;

dt = 0.01;
time = 0:0.01:10;

%%%%% STATE VALUES %%%%%
F = [0, 1, 0; 0, 0, -1; 0, 0, -1/tau];
B = [0;1;0];
G = [0;0;1];

% process noise matrix
W = zeros(3);
W(3,3) = E_a2T;


%%%% INIT CONDS %%%%%
Pinit = [0,0,0; 0, sigma_vel^2, 0; 0, 0, sigma_ac^2];
Cov(:,:,1) = Pinit;
P = Pinit;

aP = 0;

%initial state
y0=0;
v0=randn(1)*sigma_vel;
a0= randn(1)*sigma_ac;
state(:,1) = [y0;v0;a0];

%initial estimate
xhat(:,1) = [0;0;0];

%%%% STORING VALS IN STRUCT %%%%
S = struct;
i = 1;

for t = time
    H = [1/(Vc*(tF-t)), 0, 0];  %state matrix
    V = R1+R2/(tF-t)^2;     %state variance
    waT = randn*sqrt(E_a2T/dt);   %process noise 
    nu = randn*sqrt(V/dt);        %measurement noise
    %remember that std dev = sqrt(Var/T)
    
    
    %calculate the kalman gains
    K(:,i) = P*H'*(1/V);
    S(i).K = P*H'*(1/V);
    
    if t < tF
        
        %incoming measurement
        z(i) = H*state(:,i) + nu;
        
        %state propagation
        state_dot = F*state(:,i) + B*aP + G*waT;
        state(:,i+1) = state(:,i)+state_dot*dt;
        
        %state estimation propagation
        xhat_dot = F*xhat(:,i) + P*H'*(1/V)*(z(i) - H*xhat(:,i));
        xhat(:,i+1) = xhat(:,i) + xhat_dot*dt;
        
        %propagate the covariance with simple euler
        Pdot = F*P + P*F' - (P*(H'*H)*P)*(1/V) + W;
        S(i).Pdot=Pdot;
        P = P+Pdot*dt;
        S(i+1).P=P;
        Cov(:,:,i+1) = P;  
        
        %store the error
        e(:,i+1) = state(:,i+1) - xhat(:,i+1);
        
        %calculate the residual
        res(i+1) = z(i)-H*xhat(:,i); %e(:,i+1)'*H*dt + nu;
    end
    i = i+1;
end

%% Do the Monte Carlo for Gauss-Markov Process and collect average covariance, average error, residual of all MC runs
MC_Runs = 500;
[P_ave, e_ave, r] = MAE_270B_Missile_Estimation_MonteCarlo(MC_Runs);

%% Use the residuals to verify their independence
orth = 0;
t1 = 800; %8 seconds
t2 = 500; %5 seconds
for realization = 1:MC_Runs
    orth = orth + r(realization, t1)*r(realization, t2);
end
orth = orth/MC_Runs;

realization_Resid = r(1,:);
corr = xcorr(realization_Resid);
tshifts = -(size(realization_Resid,2)-1):(size(realization_Resid,2)-1);

%% Do the Monte Carlo for Telegraph Signal and collect average covariance, average error, residual of all MC runs
MC_Runs = 500;
[P_ave_telegraph, e_ave_telegraph, r_telegraph, telegraph_states, telegraph_estimates] = MAE_270B_Missile_Estimation_TelegraphSignal(MC_Runs);

%%
%%%% PLOTTING %%%%%%


%% Kalman filter Gains
figure()
title("{Kalman Gains for Gauss-Markov Process}", 'interpreter','latex', 'fontsize', 16)
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
ylabel('Gain', 'interpreter','latex', 'fontsize', 12)
hold on
plot(time, K(1,:), 'linewidth', 2)
plot(time, K(2,:), 'linewidth', 2)
plot(time, K(3,:), 'linewidth', 2)
hold off
legend({'K1', 'K2', 'K3'});
saveas(gcf,'Kalman_Gains.png')

%% Standard Deviation (from covariance matrices)
figure()
title("Std. Dev ($\sigma$) of State for Gauss-Markov Process", 'interpreter','latex', 'fontsize', 16)
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
ylabel('$\sigma$', 'interpreter','latex', 'fontsize', 16)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
hold on
plot(time, sqrt(squeeze(Cov(1,1,:))), 'linewidth', 2)
plot(time, sqrt(squeeze(Cov(2,2,:))), 'linewidth', 2)
plot(time, sqrt(squeeze(Cov(3,3,:))), 'linewidth', 2)

txt11 = ['position error $ft$'];
text(5,35,txt11, 'interpreter', 'latex')
txt22 = ['velocity error $\frac{ft}{sec}$'];
text(5,65,txt22, 'interpreter', 'latex')
txt33 = ['acceleration error $\frac{ft}{sec^2}$'];
text(5,90,txt33, 'interpreter', 'latex')

hold off
legend({'\sigma_{11}', '\sigma_{22}', '\sigma_{33}'});

saveas(gcf,'StDevs.png')

%% State Estimation - Gauss Markov
figure()
set(gcf, 'Position', [294,105,833.5,705])
sgtitle("{State Estimation and State for Gauss-Markov Process}", 'interpreter','latex', 'fontsize', 16)

subplot(3,1,1)
title('Position', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
ylabel('$ft$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
hold on
plot(time, state(1,:), 'color', blue,'linewidth', 2)
plot(time, xhat(1,:), 'color', blue, 'linestyle',':', 'linewidth', 1.5)
hold off
legend({'$x$', '$\hat{x}$'}, 'interpreter', 'latex');

subplot(3,1,2)
title('Velocity', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
ylabel('$\frac{ft}{sec}$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
hold on
plot(time, state(2,:), 'color', red, 'linewidth', 2)
plot(time, xhat(2,:), 'color', red, 'linestyle', ':', 'linewidth', 1.5)
hold off
legend({'$x$', '$\hat{x}$'}, 'interpreter', 'latex');

subplot(3,1,3)
title('Acceleration', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
ylabel('$\frac{ft}{sec^2}$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
hold on
plot(time, state(3,:), 'color', orange, 'linewidth', 2)
plot(time, xhat(3,:), 'color', orange, 'linestyle', ':', 'linewidth', 1.5)
hold off
legend({'$x$', '$\hat{x}$'}, 'interpreter', 'latex');

saveas(gcf,'StateEstimationAndState.png')


%% State Estimation Errors - Gauss Markov
figure()
set(gcf, 'Position', [294,105,833.5,705])
sgtitle("{State Estimation Errors for Gauss-Markov Process}", 'interpreter','latex', 'fontsize', 16)

subplot(3,1,1)
title('Position Error', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
ylabel('$ft$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
plot(time, e(1,:), 'color', blue,'linewidth', 2)

subplot(3,1,2)
title('Velocity Error', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
ylabel('$\frac{ft}{sec}$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
plot(time, e(2,:), 'color', red, 'linewidth', 2)

subplot(3,1,3)
title('Acceleration Error', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
ylabel('$\frac{ft}{sec^2}$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
plot(time, e(3,:), 'color', orange, 'linewidth', 2)

saveas(gcf,'StateEstimationErrors.png')


%% Plot the average stand dev - Gauss Markov
figure()
set(gcf, 'Position', [294,105,833.5,705])
sgtitle("A Priori $\sigma$ and True $\sigma$ for Gauss-Markov Process", 'interpreter','latex', 'fontsize', 16)

subplot(3,1,1)
title('Position - Gauss-Markov', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
%ylabel('$ft$', 'interpreter','latex', 'fontsize', 12)
%ylh = get(gca,'ylabel');
%set(ylh, 'Rotation',0)
hold on
plot(time, P_ave(1,:), 'color', blue,'linewidth', 2)
plot(time, sqrt(squeeze(Cov(1,1,:))), 'color', blue, 'linestyle',':', 'linewidth', 1.5)
hold off
legend({'$actual$', '$a priori$'}, 'interpreter', 'latex');

subplot(3,1,2)
title('Velocity - Gauss-Markov', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
%ylabel('$\frac{ft}{sec}$', 'interpreter','latex', 'fontsize', 12)
%ylh = get(gca,'ylabel');
%set(ylh, 'Rotation',0)
hold on
plot(time, P_ave(2,:), 'color', red, 'linewidth', 2)
plot(time, sqrt(squeeze(Cov(2,2,:))), 'color', red, 'linestyle', ':', 'linewidth', 1.5)
hold off
legend({'$actual$', '$a priori$'}, 'interpreter', 'latex');

subplot(3,1,3)
title('Acceleration - Gauss-Markov', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
%ylabel('$\frac{ft}{sec^2}$', 'interpreter','latex', 'fontsize', 12)
%ylh = get(gca,'ylabel');
%set(ylh, 'Rotation',0)
hold on
plot(time, P_ave(3,:), 'color', orange, 'linewidth', 2)
plot(time, sqrt(squeeze(Cov(3,3,:))), 'color', orange, 'linestyle', ':', 'linewidth', 1.5)
hold off
legend({'$actual$', '$a priori$'}, 'interpreter', 'latex');

saveas(gcf,'AverageStDev_GaussMarkov.png')

%% Average error - Gauss Markov
figure()
set(gcf, 'Position', [294,105,833.5,705])
sgtitle("Average Error Across Monte Carlo Realizations for Gauss-Markov", 'interpreter','latex', 'fontsize', 16)

subplot(3,1,1)
title('Position Error- Gauss-Markov', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)

hold on
plot(time, e_ave(1,:), 'color', blue,'linewidth', 2)
plot(time, sqrt(squeeze(Cov(1,1,:))), 'color', blue, 'linestyle',':', 'linewidth', 1.5)
plot(time, -sqrt(squeeze(Cov(1,1,:))), 'color', blue, 'linestyle',':', 'linewidth', 1.5)
ylabel('$ft$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
hold off
legend({'$error$'}, 'interpreter', 'latex');

subplot(3,1,2)
title('Velocity Error - Gauss-Markov', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
hold on
plot(time, e_ave(2,:), 'color', red, 'linewidth', 2)
plot(time, sqrt(squeeze(Cov(2,2,:))), 'color', red, 'linestyle', ':', 'linewidth', 1.5)
plot(time, -sqrt(squeeze(Cov(2,2,:))), 'color', red, 'linestyle', ':', 'linewidth', 1.5)
ylabel('$\frac{ft}{sec}$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
hold off
legend({'$error$'}, 'interpreter', 'latex');

subplot(3,1,3)
title('Acceleration Error - Gauss-Markov', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
hold on
plot(time, e_ave(3,:), 'color', orange, 'linewidth', 2)
plot(time, sqrt(squeeze(Cov(3,3,:))), 'color', orange, 'linestyle', ':', 'linewidth', 1.5)
plot(time, -sqrt(squeeze(Cov(3,3,:))), 'color', orange, 'linestyle', ':', 'linewidth', 1.5)
ylabel('$\frac{ft}{sec^2}$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
hold off
legend({'$error$'}, 'interpreter', 'latex');

saveas(gcf,'AverageError_GaussMarkov.png')

%% Further proof that residual is uncorrelated - Gauss Markov
 % justified by using signal correlation
figure()
plot(tshifts, corr, 'k-')
title('Auto-Correlation of Residual for One Realization', 'interpreter', 'latex');
ylabel('signal correlation', 'interpreter', 'latex')
xlabel('time shift')

saveas(gcf,'ResidualAutoCorrelation.png')


%% State Estimation - Telegraph Signal
figure()
set(gcf, 'Position', [294,105,833.5,705])
sgtitle("{State Estimation and State for Telegraph Signal}", 'interpreter','latex', 'fontsize', 16)

subplot(3,1,1)
title('Position', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
ylabel('$ft$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
hold on
plot(time, telegraph_states(1,:), 'color', blue,'linewidth', 2)
plot(time, telegraph_estimates(1,:), 'color', blue, 'linestyle',':', 'linewidth', 1.5)
hold off
legend({'$x$', '$\hat{x}$'}, 'interpreter', 'latex');

subplot(3,1,2)
title('Velocity', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
ylabel('$\frac{ft}{sec}$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
hold on
plot(time, telegraph_states(2,:), 'color', red, 'linewidth', 2)
plot(time, telegraph_estimates(2,:), 'color', red, 'linestyle', ':', 'linewidth', 1.5)
hold off
legend({'$x$', '$\hat{x}$'}, 'interpreter', 'latex');

subplot(3,1,3)
title('Acceleration', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
ylabel('$\frac{ft}{sec^2}$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
hold on
plot(time, telegraph_states(3,:), 'color', orange, 'linewidth', 2)
plot(time, telegraph_estimates(3,:), 'color', orange, 'linestyle', ':', 'linewidth', 1.5)
hold off
legend({'$x$', '$\hat{x}$'}, 'interpreter', 'latex');

saveas(gcf,'StateEstimation_Telegraph.png')

%% Plot the average stand dev - Telegraph Signal
figure()
set(gcf, 'Position', [294,105,833.5,705])
sgtitle("A Priori $\sigma$ and True $\sigma$ for Telegraph Signal", 'interpreter','latex', 'fontsize', 16)

subplot(3,1,1)
title('Position - Telegraph Signal', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
%ylabel('$ft$', 'interpreter','latex', 'fontsize', 12)
%ylh = get(gca,'ylabel');
%set(ylh, 'Rotation',0)
hold on
plot(time, P_ave_telegraph(1,:), 'color', blue,'linewidth', 2)
plot(time, sqrt(squeeze(Cov(1,1,:))), 'color', blue, 'linestyle',':', 'linewidth', 1.5)
hold off
legend({'$actual$', '$a priori$'}, 'interpreter', 'latex');

subplot(3,1,2)
title('Velocity - Telegraph Signal', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
%ylabel('$\frac{ft}{sec}$', 'interpreter','latex', 'fontsize', 12)
%ylh = get(gca,'ylabel');
%set(ylh, 'Rotation',0)
hold on
plot(time, P_ave_telegraph(2,:), 'color', red, 'linewidth', 2)
plot(time, sqrt(squeeze(Cov(2,2,:))), 'color', red, 'linestyle', ':', 'linewidth', 1.5)
hold off
legend({'$actual$', '$a priori$'}, 'interpreter', 'latex');

subplot(3,1,3)
title('Acceleration - Telegraph Signal', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
%ylabel('$\frac{ft}{sec^2}$', 'interpreter','latex', 'fontsize', 12)
%ylh = get(gca,'ylabel');
%set(ylh, 'Rotation',0)
hold on
plot(time, P_ave_telegraph(3,:), 'color', orange, 'linewidth', 2)
plot(time, sqrt(squeeze(Cov(3,3,:))), 'color', orange, 'linestyle', ':', 'linewidth', 1.5)
hold off
legend({'$actual$', '$a priori$'}, 'interpreter', 'latex');

saveas(gcf,'AverageStDev_Telegraph.png')

%% Average error - Telegraph Signal
figure()
set(gcf, 'Position', [294,105,833.5,705])
sgtitle("Average Error Across Monte Carlo Realizations for Telegraph Signal", 'interpreter','latex', 'fontsize', 16)

subplot(3,1,1)
title('Position Error - Telegraph Signal', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)

hold on
plot(time, e_ave_telegraph(1,:), 'color', blue,'linewidth', 2)
plot(time, sqrt(squeeze(Cov(1,1,:))), 'color', blue, 'linestyle',':', 'linewidth', 1.5)
plot(time, -sqrt(squeeze(Cov(1,1,:))), 'color', blue, 'linestyle',':', 'linewidth', 1.5)
ylabel('$ft$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
hold off
legend({'$error$'}, 'interpreter', 'latex');

subplot(3,1,2)
title('Velocity Error - Telegraph Signal', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
hold on
plot(time, e_ave_telegraph(2,:), 'color', red, 'linewidth', 2)
plot(time, sqrt(squeeze(Cov(2,2,:))), 'color', red, 'linestyle', ':', 'linewidth', 1.5)
plot(time, -sqrt(squeeze(Cov(2,2,:))), 'color', red, 'linestyle', ':', 'linewidth', 1.5)
ylabel('$\frac{ft}{sec}$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
hold off
legend({'$error$'}, 'interpreter', 'latex');

subplot(3,1,3)
title('Acceleration Error - Telegraph Signal', 'interpreter', 'latex');
xlabel('time (s)', 'interpreter','latex', 'fontsize', 12)
hold on
plot(time, e_ave_telegraph(3,:), 'color', orange, 'linewidth', 2)
plot(time, sqrt(squeeze(Cov(3,3,:))), 'color', orange, 'linestyle', ':', 'linewidth', 1.5)
plot(time, -sqrt(squeeze(Cov(3,3,:))), 'color', orange, 'linestyle', ':', 'linewidth', 1.5)
ylabel('$\frac{ft}{sec^2}$', 'interpreter','latex', 'fontsize', 12)
ylh = get(gca,'ylabel');
set(ylh, 'Rotation',0)
hold off
legend({'$error$'}, 'interpreter', 'latex');

saveas(gcf,'AverageError_Telegraph.png')
