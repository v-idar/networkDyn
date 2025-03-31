%% 
clc 
clear all
close all

% Transition rate matrix
Lambda = [0 2/5 1/5 0 0;
          0 0 3/4 1/4 0;
          1/2 0 0 1/2 0;
          0 0 1/3 0 2/3;
          0 1/3 0 1/3 0];

% Accumulated transition matrix
P_sum = [2/5 4/5 5/5 0 0;
        0 0 3/4 4/4 0;
        1/2 0 0 2/2 0;
        0 0 1/3 0 3/3;
        0 1/3 0 2/3 3/3];



% rate vector
omega = Lambda * ones(5, 1);

% Transition matrix 
P = diag(1./omega) * Lambda;

% Phat vector
Phat = Lambda / max(omega);
for i = 1:5
    Phat(i,i) = 1-sum(Phat(i,:));
end

% space vector
space = ["o" "a" "b" "c" "d"];

%% 1a) och b)
%start pos
pos = 1;

% time to wait in each node
zzz = 1./omega;

% starttime & nbrOfIteration
iter = 1000000;
time = zeros(1, iter);

% go for iter numbers of iterations
for p = 1:iter

    % iteration
    run = true;
    while run == true
    % random value
    u = rand(1);

        % Checking which node the particle transitions to. 
        time(1, p) = time(1, p) + 1;
        for i=1:5
            if u < P_sum(pos, i)
            pos = i;
    
            if pos == 5 % stopping position
                run = false;
            end
            break    
            end
        end
    end
end 
fprintf("mean time for 1 000 000 iterations: %f \n", mean(time))

%% 1b Theoretical return times %% 
% eigenvector of P transpose corresponding to eigenvalue 1
[V, D] = eigs(Phat');
inv_dist = V(: , 1);
inv_dist = inv_dist / sum(inv_dist);

% Return times
E = 1 ./(omega .* inv_dist);
E(2)

%% 1c) Simulated hitting times 

% time to wait in each node
zzz = 1./omega;

% starttime & nbrOfIteration
iter = 1000000;
time = zeros(1, iter);

% go for iter numbers of iterations
for p = 1:iter
    % start position
    pos = 1;

    % iteration
    run = true;
    while run == true
    % random value
    u = rand(1);

        % Checking which node the particle transitions to. 
        time(1, p) = time(1, p) + 1;
        for i=1:5
            if u <= P_sum(pos, i)
            pos = i;
    
            if pos == 5 % stopping position
                run = false;
            end
            break    
            end
        end
    end
end 
fprintf("mean time for 1 000 000 iterations: %f \n", mean(time))
%% 1d) Theoretical hitting time
zzz = 1./omega;
zzz(5) = 0;
P(5, :) = [0,0,0,0,0];
z = (diag([1, 1, 1, 1, 1])-P) \ zzz;

fprintf("Theoretical hitting time from o-d: \t %0.2f seconds \n", z(1))



%% 2 a) Visualization of line graph changing color
figure
set(gcf,'color','white')

% adjacency matrix
W = diag(ones(9, 1), 1) + diag(ones(9, 1), -1);
zzz = 0.001;

% number of iterations or expected running time
Tmax = 200;
x = ones(10, 1); % x is one for the current particle state
cords = [[1:10]'  zeros(10,1)];
% Potential function: 
U_t = zeros(1, Tmax);

% Plot the graph and the initial states with red
subplot(211)
gplot(W,cords,'-k');
hold on

% Initialise grid
for i = 1:10
    if x(i, 1) == 1
        scatter(cords(i,1),cords(i,2),200,'markeredgecolor','k','markerfacecolor', 'r');
    else
        scatter(cords(i,1),cords(i,2),200,'markeredgecolor','k','markerfacecolor', 'g');
    end

end
set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w')


% Iterations
for t = 1:Tmax
    n = randi(10,1); % choose which node that wakes up
    eta = t/100;     % invers of noise
    prob = [0 0]; % first entry is green, second entry is red
    % numerator
    numsum = 0;
    for a = 0:1
        c = x==a;
        %for j =find(W(n,:)~=0)
        %numsum = numsum + W(n,j) * (x(j) == s);
        %end
        numsum = W(n,:)*c;
    prob(a+1) = exp(-eta * numsum);
    end
    prob = prob ./ sum(prob);
    F = cumsum(prob);
    u = rand(1);

   x(n) = (find(F>u, 1) - 1); %update the state vector

   
    
    % plot the new location of the node
%     subplot(211)
%     for k = 1:10
%         if x(k) == 1
%             scatter(cords(k,1),cords(k,2),200,'markeredgecolor','k','markerfacecolor', 'r');
%         else
%             scatter(cords(k,1),cords(k,2),200,'markeredgecolor','k','markerfacecolor', 'g');
%         end
%     end
%     
    

    for i = 1:10
        for j = find(W(i,:)~=0)
            U_t(t) = U_t(t) + 1/2 * W(i,j) * (x(i) == x(j));
        end
    end
end
for k = 1:10
     if x(k) == 1
             scatter(cords(k,1),cords(k,2),200,'markeredgecolor','k','markerfacecolor', 'r');
     else
             scatter(cords(k,1),cords(k,2),200,'markeredgecolor','k','markerfacecolor', 'g');
     end
end
     

subplot(212)
tvec = [0 1:Tmax];
plot(tvec(1:end-1),U_t((1:Tmax)))

%% part B %%
load -ASCII coord.mat;
load -ASCII wifi.mat;

figure
set(gcf,'color','white')

% adjacency matrix
coord;

% number of iterations or expected running time
Tmax = 1000;
colors = ['r', 'g', 'b', 'y' 'm' 'c' 'w' 'b' ];

% Potential function: 
U_t = zeros(1, Tmax);

% Plot the graph and the initial states with red
subplot(211)
gplot(wifi,coord,'-k');
hold on

% Initialise vector 
x = ones(100, 1);


% Initialise grid
subplot(211)
for i = 1:length(coord)
        scatter(coord(i,1),coord(i,2), 10,'markeredgecolor','k','markerfacecolor', colors(x(i)));   
end
hold on
set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w')


% Cost function
Cost = diag(2*ones(8,1)) + diag(ones(7,1), 1) + diag(ones(7,1), -1);
% Potential function
U_t = zeros(1, Tmax);



% Iterations
for t=1:Tmax
    n = randi(100,1); % choose which node that wakes up
    eta = log(t+1);     % invers of noise
    prob = zeros(10, 1); %
    % numerator
    numsum = 0;
    for a = 1:8
        c = 0;
        for j =find(wifi(n,:)~=0)
            c = c + wifi(n,j) * Cost(a, x(j));
        end
    prob(a) = exp(-eta * c);
    end

    prob = prob ./ sum(prob);
    F = cumsum(prob);
    u = rand(1);

   
   % calculate potential
    for i = 1:100
        for j = find(wifi(i,:)~=0)
            U_t(t) = U_t(t) + 1/2 * wifi(i,j) * (Cost(x(i), x(j)));
        end
    end
    
   % update state vector
   x(n) = (find(F>u, 1)); %update the state vector  

end

% update graph
for i = 1:length(coord)
        scatter(coord(i,1),coord(i,2), 10,'markeredgecolor','k','markerfacecolor', colors(x(i)));   
end



subplot(212)
tvec = [0 1:Tmax];
plot(tvec(1:end-1),U_t((1:Tmax)))



