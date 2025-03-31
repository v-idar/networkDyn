%% Task 1.1 Epidemic on a known graph 
clear all, close all, clc
% Create adjacency matrix and state vector
n = 500;
W = zeros(n);
W = W + diag(ones(n-1,1),1); % add ones on the +1 off-diagonal
W = W + diag(ones(n-1,1),-1); % add ones on the -1 off-diagonal
W = W + diag(ones(n-2,1),2); % add ones on the +2 off-diagonal
W = W + diag(ones(n-2,1),-2); % add ones on the -2 off-diagonal
W = W + diag(ones(1,1),n-1); % add ones on the +n-1 off-diagonal
W = W + diag(ones(1,1),1-n); % add ones on the -n+1 off-diagonal
W = W + diag(ones(2,1),n-2); % add ones on the +n-2 off-diagonal
W = W + diag(ones(2,1),2-n); % add ones on the -n+2 off-diagonal
W = sparse(W); % transform it into a sparse matrix




S = 0;
I = 1;
R = 2;
V = 3;

beta = 0.3; % Probability that the desease spead to an infected node,
rho = 0.7; % Probability that a node recovers
k = 4;
% W = create_graph(n,k);
G = graph(W); % convert W into a graph (might not work on all MATLAB versions.

% Simulation
iter = 100;  % Number of iterations
nbr_weeks = 15; % Number of weeks
new_inf = zeros(iter,16); % Vector storing the newly infected
nbr_infected = zeros(iter,16); % Vector storing the nbr infected each week
nbr_recovered = zeros(iter,16); % Vector storing the nbr recovered each week
nbr_susceptible = zeros(iter,16); % Vector storing the nbr susceptible each week
nbr_vaccinated = zeros(iter,16); % Vector storing the nbr vaccinated each week

for t = 1:iter
    % Initialize state vector
    X = zeros(n,1);
    infected = randperm(n,10);
    X(infected, 1) = 1;
    nbr_infected(:,1) = 10;
    nbr_susceptible(:, 1) = 490;
    

    % Vaccination 
    Vacc = [0, 5, 15, 25, 35, 45, 55, 60, 60, 60, 60, 60, 60, 60, 60]; % Percent of population to be vaccinated
    %Vacc = [5, 9, 16, 24, 32, 40, 47, 54, 59, 60, 60, 60, 60, 60, 60, 60];
    Vacc = [0, diff(Vacc)]; % Percent to be vaccinated each week 

    P_vacc = ones(n,1)/n; % Probability to get vaccinated
    

    for w = 2:nbr_weeks+1
        m  = W * (X == I); % number of infected neighbors
        P_i = (X==S).* (1 - (1-beta).^m); % Probability that a node gets infected
        P_r = (X==I) * rho; % Probability that an infected node recovers
        u = rand(n, 1);
        
        X(find(P_i-u > 0)) = I; % update vector with infected nodes
        new_inf(t, w) = sum(P_i > u); % total number of infected people
    
        X(find(P_r-u > 0)) = R; % update vector with recovered node
        newly_rec = sum(P_r > u); % number of recovered nodes

%         vaccinated_nbr = round(Vacc(w-1)/100 * n);
%         for p = 1:vaccinated_nbr
%             vaccinated = randsample(n, 1, true, P_vacc); % Uppdating the population with the vaccinated people
%             P_vacc(vaccinated) = 0;
%             X(vaccinated) = V;
%         end

        % Adding nbr of S, I and R each week
        nbr_susceptible(t, w) = sum(X == S);
        nbr_infected(t, w) = sum(X == I);
        nbr_recovered(t, w) = sum(X == R);
        nbr_vaccinated(t, w) = sum(X == V);
    end
    
end

% Calculating means
avg_new_inf = mean(new_inf, 1);
avg_sus = mean(nbr_susceptible, 1);
avg_inf = mean(nbr_infected, 1);
avg_rec = mean(nbr_recovered, 1);
avg_vac = mean(nbr_vaccinated, 1);

x = (0:nbr_weeks); % x vector
% Plot 1
figure()
subplot(2,1,1)
plot(x, avg_new_inf);
title('Newly infected people')
xlabel('Week')
ylabel('Nbr of people')
legend('newly infected')

% Plot 2
subplot(2,1,2)
plot(x, avg_rec, 'g', x, avg_inf, 'r', x, avg_sus, 'b')
% , x, avg_vac, "p"
xlabel('Week')
ylabel('Nbr of people')
title('Nbr of susceptible, infected and recovered')
legend('rec', 'inf', 'sus', "vac")

% fprintf('done\n')


%% Task 2, i copy pasted the code and made some adjustments from task 1
clear all, close all, clc
% Create adjacency matrix and state vector
n = 500;

S = 0;
I = 1;
R = 2;
V = 3;

beta = 0.3; % Probability that the desease spead to an infected node,
rho = 0.7; % Probability that a node recovers
k = 6;
W = create_graph(n,k);
G = graph(W); % convert W into a graph (might not work on all MATLAB versions.

% Simulation
iter = 100;  % Number of iterations
nbr_weeks = 15; % Number of weeks
new_inf = zeros(iter,16); % Vector storing the newly infected
nbr_infected = zeros(iter,16); % Vector storing the nbr infected each week
nbr_recovered = zeros(iter,16); % Vector storing the nbr recovered each week
nbr_susceptible = zeros(iter,16); % Vector storing the nbr susceptible each week
nbr_vaccinated = zeros(iter,16); % Vector storing the nbr vaccinated each week

for t = 1:iter
    % Initialize state vector
    X = zeros(n,1);
    infected = randperm(n,10);
    X(infected, 1) = 1;
    nbr_infected(:,1) = 10;
    nbr_susceptible(:, 1) = 490;
    

    % Vaccination 
    Vacc = [0, 5, 15, 25, 35, 45, 55, 60, 60, 60, 60, 60, 60, 60, 60]; % Percent of population to be vaccinated
    %Vacc = [5, 9, 16, 24, 32, 40, 47, 54, 59, 60, 60, 60, 60, 60, 60, 60];
    Vacc = [0, diff(Vacc)]; % Percent to be vaccinated each week 

    P_vacc = ones(n,1)/n; % Probability to get vaccinated
    

    for w = 2:nbr_weeks+1
        m  = W * (X == I); % number of infected neighbors
        P_i = (X==S).* (1 - (1-beta).^m); % Probability that a node gets infected
        P_r = (X==I) * rho; % Probability that an infected node recovers
        u = rand(n, 1);
        
        X(find(P_i-u > 0)) = I; % update vector with infected nodes
        new_inf(t, w) = sum(P_i > u); % total number of infected people
    
        X(find(P_r-u > 0)) = R; % update vector with recovered node
        newly_rec = sum(P_r > u); % number of recovered nodes

%         vaccinated_nbr = round(Vacc(w-1)/100 * n);
%         for p = 1:vaccinated_nbr
%             vaccinated = randsample(n, 1, true, P_vacc); % Uppdating the population with the vaccinated people
%             P_vacc(vaccinated) = 0;
%             X(vaccinated) = V;
%         end

        % Adding nbr of S, I and R each week
        nbr_susceptible(t, w) = sum(X == S);
        nbr_infected(t, w) = sum(X == I);
        nbr_recovered(t, w) = sum(X == R);
        nbr_vaccinated(t, w) = sum(X == V);
    end
    
end

% Calculating means
avg_new_inf = mean(new_inf, 1);
avg_sus = mean(nbr_susceptible, 1);
avg_inf = mean(nbr_infected, 1);
avg_rec = mean(nbr_recovered, 1);
avg_vac = mean(nbr_vaccinated, 1);

x = (0:nbr_weeks); % x vector
% Plot 1
figure()
subplot(2,1,1)
plot(x, avg_new_inf);
title('Newly infected people')
xlabel('Week')
ylabel('Nbr of people')
legend('newly infected')

% Plot 2
subplot(2,1,2)
plot(x, avg_rec, 'g', x, avg_inf, 'r', x, avg_sus, 'b')
% , x, avg_vac, "p"
xlabel('Week')
ylabel('Nbr of people')
title('Nbr of susceptible, infected and recovered')
legend('rec', 'inf', 'sus', "vac")

% fprintf('done\n')




%% Task 3 i copy pasted the code and made some adjustments from task 2, uncommented on the vaccination rows
clear all, close all, clc
% Create adjacency matrix and state vector
n = 500;

S = 0;
I = 1;
R = 2;
V = 3;

beta = 0.3; % Probability that the desease spead to an infected node,
rho = 0.7; % Probability that a node recovers
k = 6;
W = create_graph(n,k);
G = graph(W); % convert W into a graph (might not work on all MATLAB versions.

% Simulation
iter = 100;  % Number of iterations
nbr_weeks = 15; % Number of weeks
new_inf = zeros(iter,16); % Vector storing the newly infected
nbr_infected = zeros(iter,16); % Vector storing the nbr infected each week
nbr_recovered = zeros(iter,16); % Vector storing the nbr recovered each week
nbr_susceptible = zeros(iter,16); % Vector storing the nbr susceptible each week
nbr_vaccinated = zeros(iter,16); % Vector storing the nbr vaccinated each week

for t = 1:iter
    % Initialize state vector
    X = zeros(n,1);
    infected = randperm(n,10);
    X(infected, 1) = 1;
    nbr_infected(:,1) = 10;
    nbr_susceptible(:, 1) = 490;
    

    % Vaccination 
    Vacc = [0, 5, 15, 25, 35, 45, 55, 60, 60, 60, 60, 60, 60, 60, 60]; % Percent of population to be vaccinated
    %Vacc = [5, 9, 16, 24, 32, 40, 47, 54, 59, 60, 60, 60, 60, 60, 60, 60];
    Vacc = [0, diff(Vacc)]; % Percent to be vaccinated each week 

    P_vacc = ones(n,1)/n; % Probability to get vaccinated
    

    for w = 2:nbr_weeks+1
        m  = W * (X == I); % number of infected neighbors
        P_i = (X==S).* (1 - (1-beta).^m); % Probability that a node gets infected
        P_r = (X==I) * rho; % Probability that an infected node recovers
        u = rand(n, 1);
        
        X(find(P_i-u > 0)) = I; % update vector with infected nodes
        new_inf(t, w) = sum(P_i > u); % total number of infected people
    
        X(find(P_r-u > 0)) = R; % update vector with recovered node
        newly_rec = sum(P_r > u); % number of recovered nodes

        vaccinated_nbr = round(Vacc(w-1)/100 * n);
        for p = 1:vaccinated_nbr
            vaccinated = randsample(n, 1, true, P_vacc); % Uppdating the population with the vaccinated people
            P_vacc(vaccinated) = 0;
            X(vaccinated) = V;
        end

        % Adding nbr of S, I and R each week
        nbr_susceptible(t, w) = sum(X == S);
        nbr_infected(t, w) = sum(X == I);
        nbr_recovered(t, w) = sum(X == R);
        nbr_vaccinated(t, w) = sum(X == V);
    end
    
end

% Calculating means
avg_new_inf = mean(new_inf, 1);
avg_sus = mean(nbr_susceptible, 1);
avg_inf = mean(nbr_infected, 1);
avg_rec = mean(nbr_recovered, 1);
avg_vac = mean(nbr_vaccinated, 1);
newly_vac = [0, diff(avg_vac)];

x = (0:nbr_weeks); % x vector
% Plot 1
figure()
subplot(2,1,1)
plot(x, avg_new_inf, "r", x, newly_vac, 'b');
title('Newly infected people')
xlabel('Week')
ylabel('Nbr of people')
legend('newly infected', 'newly vaccinated')

% Plot 2
subplot(2,1,2)
plot(x, avg_rec, 'g', x, avg_inf, 'r', x, avg_sus, 'b', x, avg_vac, "p")

xlabel('Week')
ylabel('Nbr of people')
title('Nbr of susceptible, infected and recovered')
legend('rec', 'inf', 'sus', "vac")

% fprintf('done\n')





%% Task 4 optimizing the input parameters
clear all, close all, clc
run = true;

S = 0;
I = 1;
R = 2;
V = 3;

% Simulation
iter = 10;  % Number of iterations
nbr_weeks = 15; % Number of weeks
n = 934;
nbr_real_infected = [1, 1, 3, 5, 9, 17, 32, 32, 17, 5, 2, 1, 0, 0, 0, 0];
Vacc_given = [5, 9, 16, 24, 32, 40, 47, 54, 59, 60, 60, 60, 60, 60, 60, 60];
Vacc = [5, diff(Vacc_given)]; % Percent to be vaccinated each week  


% Initializing best parameters
best_beta_batch = 0;
best_rho_batch = 0;
best_k_batch = 0;
best_beta = 0;
best_rho = 0;
best_k = 0;


% Increments 
d_beta = 0.2; % Gradient beta
d_rho = 0.2; % Gradient rho
d_k = 4;     % Gradient k
RMS_min = inf; % Min RMS
epoch = 0;

% Plot real infected vector
x = (0:nbr_weeks); % x vector
figure();
plot(x, nbr_real_infected, 'r');
title('Newly infected people')
xlabel('Week')
ylabel('Nbr of people')
legend('newly infected')
hold on

n_half = 0; % number of times to half the gradient of the parameters


% Iterating until parameters are
while run
    epoch = epoch + 1;
    
    % First values for parameters or updating values
    if epoch == 1
        beta = 0.3; % Probability that the desease speads to an infected node,
        rho = 0.6; % Probability that a node recovers
        k = 10; % Average degree on randomly generated graph
    else
        beta = best_beta_batch;
        rho = best_rho_batch;
        k = best_k_batch;
    end

    % Stop criterion or minimizing gradient
    if best_beta == best_beta_batch && best_rho == best_rho_batch && best_k == best_k_batch && epoch ~= 1
        % Dividing the gradient in two
        if n_half < 4
            d_beta = d_beta/2;
            d_rho = d_rho/2;
            d_k = d_k/2;
            if d_k < 1
                d_k = 1;
            end

            n_half = n_half +1;
        else
            run = false;
            break;
        end
    end

    % Updating best vector
    best_beta = best_beta_batch;
    best_k = best_k_batch;
    best_rho = best_rho_batch;
 
    
    % Updating the set of the parameters
    beta_vec = [beta-d_beta, beta, beta+d_beta]; % beta vector with the set of possible parameters
    rho_vec = [rho-d_rho, rho, rho+d_rho]; 
    k_vec = [k-d_k, k, k+d_k];
    if k_vec(1) < 1
        k_vec(1) = 1;
    end

    for i = 1:3
        % Create random graph
        k = k_vec(i);
        W = create_graph(n, k);
        for j=1:3
            rho = rho_vec(j);
            for l=1:3              
                beta = beta_vec(l);
                
                new_inf = zeros(iter,nbr_weeks + 1); % Vector storing the newly infected
                new_inf(:,1) = 1; % New infected week 1 is 1
                nbr_infected = zeros(iter, nbr_weeks + 1); % Vector storing the nbr infected each week
                nbr_recovered = zeros(iter, nbr_weeks + 1); % Vector storing the nbr recovered each week
                nbr_susceptible = zeros(iter, nbr_weeks + 1); % Vector storing the nbr susceptible each week
                nbr_vaccinated = zeros(iter, nbr_weeks + 1); % Vector storing the nbr vaccinated each week
                
                % iterating 10 times
                for t = 1:iter
    
                    % Initialize state vector and adding 10 randomly infected
                    % people
                    X = zeros(n,1);
                    infected = randperm(n, 1);
                    X(infected, 1) = I;
                    nbr_infected(:,1) = 1;
                    nbr_susceptible(:, 1) = n-1;           
                    P_vacc = ones(n,1) / n; % Probability to get vaccinated
                    
                    % Simulating each week
                    for w = 2:nbr_weeks+1
                        m  = W * (X == I); % number of infected neighbors
                        P_i = (X==S).* (1 - (1-beta).^m); % Probability that a node gets infected
                        P_r = (X==I) * rho; % Probability that an infected node recovers
                            
                        u = rand(n, 1); % Random variable between 0 and 1
                        X(P_i-u > 0) = I; % update vector with infected nodes
                        new_inf(t, w) = sum(P_i > u); % total number of infected people
                    
                        X(P_r -u > 0) = R; % update vector with recovered node
                        newly_rec = sum(P_r -u > 0); % number of recovered nodes


                        vaccinated_nbr = round(Vacc(w-1)/100 * n);
                        for p = 1:vaccinated_nbr
                            vaccinated = randsample(n, 1, true, P_vacc); % Uppdating the population with the vaccinated people
                            P_vacc(vaccinated) = 0;
                            X(vaccinated) = V;
                        end
                
                        % Adding nbr of S, I and R each week
                        nbr_susceptible(t, w) = sum(X == S);
                        nbr_infected(t, w) = sum(X == I);
                        nbr_recovered(t, w) = sum(X == R);
                        nbr_vaccinated(t, w) = sum(X == V);
                    end
                end
                

                % Calculating root mean square
                avg_new_inf = mean(new_inf, 1);
%                 RMS = rms(nbr_real_infected-avg_new_inf);
                RMS = sqrt(mean((nbr_real_infected - avg_new_inf).^2));

                % Checking stop criterion and updating parameters
                if RMS < RMS_min 
                    RMS_min = RMS;

                    % Setting new best parameters
                    best_beta_batch = beta;
                    best_rho_batch = rho;
                    best_k_batch = k;
                    
                    % Plot the new infection curve
                    plot(x, avg_new_inf);
                    pause(0.2)

                    % Calculate all the new values needed if it were to be
                    % the last iteration
                    % Calculating means
                    avg_new_inf_best = mean(new_inf, 1);
                    avg_sus = mean(nbr_susceptible, 1);
                    avg_inf = mean(nbr_infected, 1);
                    avg_rec = mean(nbr_recovered, 1);
                    avg_vac = mean(nbr_vaccinated, 1);



                end
            end
        end
    end
    fprintf("epoch: %d \n \t beta = %6.2f \n \t rho = %6.2f \n \t k = %d \n ", epoch, best_beta_batch, best_rho_batch, best_k_batch)
end


% Plotting the number of infected, simulation vs real
figure()
subplot(2,1,1)
plot(x, avg_new_inf_best, x, nbr_real_infected)
legend("simulation", "real")
title("Nbr of infected people each week, simulation and real values")

% Plotting the total susceptible, infected, recovered and vaccinated 
subplot(2,1,2)
plot(x, avg_rec, 'g', x, avg_inf, 'r', x, avg_sus, 'b', x, avg_vac, "p")
fprintf("best parameters\n k = %d, beta = %1.2f, rho = %1.2f", best_k_batch, best_beta_batch, best_rho_batch)
xlabel('Week')
ylabel('Nbr of people')
title('Nbr of susceptible, infected and recovered')
legend('rec', 'inf', 'sus', "vac")



%% Task 1.2 Generate a random graph
function W = create_graph(n, k)
% n = 493;
% k = 10; % The wanted average degree of the graph
% delta_k = 1;

nbr_nodes = n-k-1; % Number of nodes
k0 = k + 1; % Starting position 

% Creating a connected graph
W = ones(k0) - diag(ones(k0, 1), 0);
g = graph(W);
% plot(g)
% title("Initial graph, k = 4")


for n = 1:nbr_nodes
    
    w = sum(W, 2); % calculating out-degree of each node
    P = w./sum(w); % Probability vector P 
    pop_size = length(P);
    c = k/2;
    if mod(k,2) ~= 0
        if mod(n, 2) ~= 0
            c = floor(k/2);
        else
            c = ceil(k/2);
        end
    end

    for i = 1:c
        % calculate the probability for link between nodes and 
        % Adjacency matrix for the new graph
        neighbor = randsample((1:pop_size), 1, true, full(P)); % Selecting a random node from the population
        P(neighbor) = 0;
        W(pop_size+1, neighbor) = 1; % Updating the adjacency matrix with the 
                                     % selected node
        W(neighbor, pop_size+1 ) = 1;
    end
end

% fprintf("Statistics for the final graph: \n Nbr of nodes: %6.2f \n average degree: %6.2f", length(W), sum(w)/length(w))
% figure()
% g = graph(W);
% plot(g)
% title("Final graph, k = 4")
end
