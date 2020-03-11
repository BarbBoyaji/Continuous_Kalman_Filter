
function [P_ave, e_ave, r, telegraph_states, telegraph_estimates] = MAE_270B_Missile_Estimation_TelegraphSignal(MC)

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
aP = 0;


% process noise matrix
W = zeros(3);
W(3,3) = E_a2T;


%%%% INIT CONDS %%%%%
Pinit = [0,0,0; 0, sigma_vel^2, 0; 0, 0, sigma_ac^2];
Cov(:,:,1) = Pinit;

%constant for telegraph signal
lambda = .25; %hertz


%initial estimate
xhat(:,1) = [0;0;0];

%%%% STORING VALS IN STRUCT %%%%
S = struct;

for k = 1:MC
    i = 1;
    rng('shuffle')
    clear state;
    
    % Random Telegraph for initial state
    aT = 100; %(ft/sec^2)^2
    if rand(1) > 0.5
        aT = -aT;
    end
    %defaults for initializing switched time
    reached_switch = true;
    tswitch = (0 - 1/lambda*log(rand(1)));
    
    %initial state
    y0=0;
    v0=randn(1)*sigma_vel;
    a0= aT;
    state(:,1) = [y0;v0;a0];
    P = Pinit;
    
    for t = time
        H = [1/(Vc*(tF-t)), 0, 0];  %state matrix
        V = R1+R2/(tF-t)^2;     %state variance
        waT = randn*sqrt(E_a2T/dt);   %process noise
        nu = randn*sqrt(V/dt);        %measurement noise
        %remember that std dev = sqrt(Var/T)
        
        %monitor for reaching switch time
        if (tswitch <= t)
            aT = -aT;
            reached_switch = true;
        end
        
        %calculate next switch if we're not reached the next switch time
        %yet
        if reached_switch
            tswitch = (t - 1/lambda*log(rand(1)));
            reached_switch = false;
        end
        

        %calculate the kalman gains
        K(:,i) = P*H'*(1/V);
        S(k).K(:,i) = P*H'*(1/V);

        if t < tF

            %incoming measurement
            z(i) = H*state(:,i) + nu;

            %state propagation
            state_dot = F*state(:,i) + B*aP + G*waT;
            state(:,i+1) = state(:,i)+state_dot*dt;
            
            %impose the random telegraph signal
            state(3,i+1) = aT;
            
            S(k).state(:,i+1) = state(:,i+1);
            
            %state estimation propagation
            xhat_dot = F*xhat(:,i) + P*H'*(1/V)*(z(i) - H*xhat(:,i));
            xhat(:,i+1) = xhat(:,i) + xhat_dot*dt;
            S(k).xhat(:,i+1) = xhat(:,i+1);
            
            
            %propagate the covariance with simple euler
            Pdot = F*P + P*F' - (P*(H'*H)*P)*(1/V) + W;
            P = P+Pdot*dt;
            S(k).P(:,:,i+1)=P;
            Cov(:,:,i+1) = P;

            %store the error
            e(:,i+1) = state(:,i+1) - xhat(:,i+1);
            S(k).e(:,i+1) = e(:,i+1);
            
            %store the residual
            r(k, i+1) = z(i)-H*xhat(:,i);

        end





        i = i+1;
    end
    %realization loop
    
end

telegraph_states = state;
telegraph_estimates = xhat;
%% Calculations
%neede for verifying that the actual error variance (e) computed in the simulation
%matches the a priori error variance in the Kalman filter gain (P)

%compute the average error and covariance
e_ave = [0;0;0];
P_ave = [0;0;0];
for j =1:MC
   e_ave = e_ave + S(j).e;
   P_ave = P_ave + S(j).e.^2;
end
e_ave = e_ave/MC;
P_ave = sqrt(P_ave/(MC));

end
