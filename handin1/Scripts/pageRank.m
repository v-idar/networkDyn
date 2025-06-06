clear all
clc

%%
load -ascii twitter.mat
load -ascii users.mat
W = spconvert(twitter);
W = W((1:6881), (1:6881))  % Ignoring the rows that make the matrix asymmetric
W(1,1) = 1;                 % adding a selfloop
P = sparse(diag(sum(W,2))\W);

digraph(W)
no_followers = setdiff((1:6893), twitter(:,2));

%% Pagerank centrality for twitter influence
beta = 0.15;
% PageRank centrality
[n, ~] = size(W);
z = sparse(zeros(n,1));
Ps = sparse(eye(n));

% Parameters
beta = 0.15;

% initialize mu k for equation (2.20) in the lecture notes
mu = ones(n,1);
k = 0;
dz = beta*mu;
z = z + dz;
norm(dz)
% Iterate through the geometric sum
while norm(dz) > 0.1
k = k + 1;
Ps = sparse(P' * Ps);
dz = beta*(1-beta)^(k)*Ps*mu;
z = z + dz;
end

% Sort the nodes pageRank centralities and extract the 5 nodes with most
% centrality
[~, I] = sort(z, "descend");
central_users = users(I);

% Print the results
disp('The 5 most central users are: ')
for i=1:5
    fprintf("\t TwitterId%d: %d, \t Index in vector: %d \n",i, central_users(i), I(i))
end
% Comment: we get the most central node to be the first node which we might
% not be entirely correct since we do not track the outdegree of that node
% so I assume it is set to zero. 

%% Simulation of discrete time consensus value with two stubborn nodes
s1 = 3000; % Index of stubborn node 1
s2 = 3001; % Index of stubborn node 2
u1 = 1;
u2 = 0; 

stubborn = [I(s1), I(s2)];          % Stubborn nodes
regular = setdiff(1:n, stubborn);   % Regular nodes
u = [u1; u2];                       % Value of stubborn nodes

niter = 2000;  % Number of iterations

% Submatrices
Q = P(regular, regular);
E = P(regular, stubborn);
x = 0.5 * ones(n,niter);
x(stubborn,1) = u;

% Iterate through the simulation corresponding to equation (6.19) in the
% lecturenotes
for i = 2:niter
x(regular, i) = Q*x(regular, i-1) + E*x(stubborn, i-1);
x(stubborn, i) = x(stubborn, i-1);
end

xplot = x([stubborn, I((1:10), :)'], :); % Choosing the nodes to plot, choosing the 10 with the highest centrality
plot(xplot')    
xlabel(['Iterations']);
ylabel(['Opinion']);
ylim([-0.1 1.1]);
titlestring = sprintf("Stubborn nodes on Index [%d %d] with u = [%d; %d]'", s1, s2, u1, u2);
disp(titlestring);
title([titlestring]);

% Plotting the histograms of the last iteration
figure()
hist(x(:,end-1)')
xlabel('bins of histogram with width 0.1');
ylabel('count');
title('Histogram of counts of opinion', titlestring)

%% Investigate how the choice of nodes with respect to their pageRank change the stationary opinion 
% distribution
Q = P(regular,regular);
E = P(regular, stubborn);

x_stat = ones(n,1);
x_stat(regular) = (eye(n-2) - Q) \ E * u;
x_stat(stubborn) = u;
figure()
hist(x_stat')
xlabel('bins of histogram with width 0.1');
ylabel('count');
title('Histogram of counts of opinion', titlestring)

%% Stationary opinion vector
x = 0.5 * ones(n,1);
x(stubborn,1) = u;
[eigvec, eigval] = eigs(P')
alpha = eigvec(:,1)' * x;
