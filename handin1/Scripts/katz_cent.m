load("IOdownload.mat")

%% Sweden data
swe = io.swe2000; % Load data
W = adjacency(g, "weighted");
beta = 0.15;

%% Sweden, The katz centrality with mu as the unit vector
n = length(W);
mu = ones(n,1);

[~, val] = eigs(W);
z = (eye(n) - (1-beta) / val(1) * W' ) \ mu * beta;
[~, I_katz] = sort(z, "descend");

% Printing the results
disp('The three most central sectors in Sweden are (katz centrality 1): ')
for i = 1:3
fprintf('\t Sector: %s\n', name{I_katz(i)})
end
fprintf('\n')

%% Sweden, The katz centrality with different mu
n = length(W);
mu = zeros(n,1);
mu(31) = 1;

[~, val] = eigs(W);
z = (eye(n) - (1-beta) / val(1) * W' ) \ mu * beta;
[~, I_katz] = sort(z, "descend");

% Printing the results
disp('The three most central sectors in Sweden are (katz centrality 2): ')
for i = 1:3
fprintf('\t Sector: %s\n', name{I_katz(i)})
end
fprintf('\n')

%% Indonesia data
idn = io.idn2000; % Load data
W = adjacency(g, "weighted");
beta = 0.15;

%% Indonesia, The katz centrality with mu as the unit vector
n = length(W); % number of nodes
mu = ones(n,1);

[~, val] = eigs(W);
z = (eye(n) - (1-beta) / val(1) * W' ) \ mu * beta;
[~, I_katz] = sort(z, "descend");

% Printing the results
disp('The three most central sectors in Indonesia are (katz centrality 1): ')
for i = 1:3
fprintf('\t Sector: %s\n', name{I_katz(i)})
end
fprintf('\n')

%% Indonesia, The katz centrality with different mu
n = length(W);
mu = zeros(n,1);
mu(31) = 1;

[~, val] = eigs(W);
z = (eye(n) - (1-beta) / val(1) * W' ) \ mu * beta;
[v, I_katz] = sort(z, "descend");

% Printing the results
disp('The three most central sectors in Indonesia are (katz centrality 2): ')
for i = 1:3
fprintf('\t Sector: %s\n', name{I_katz(i)})
end
fprintf('\n')
