%% load data
clear all
clc

load -ascii capacities.mat
load -ascii flow.mat
load -ascii traveltime.mat
load -ascii traffic.mat
%% Create graph
[s, ~] = find(traffic > 0);
[t, ~] = find(traffic < 0);

G = graph(s, t, traveltime);
nbrOfLinks = size(traffic, 2);

%% 1 a) shortest path
p = plot(G, 'EdgeLabel', round(G.Edges.Weight, 2));
[path, D] = shortestpath(G, 1, 17);
highlight(p, path,'EdgeColor','red')
fprintf("\t shortest path %d \n", path, D)

%% 1 b) max flow
G_cap = graph(s, t, capacities)
mf = maxflow(G_cap, 1, 17);
fprintf("\t maxflow %d \n", mf)
    
%% 1 c) external inflow or outflow at each node
nu = traffic * flow;

%% 1 d) social optimum
nu(2:16) = 0;
nu(17) = -nu(1);
cvx_begin
    variable f(nbrOfLinks)
    minimize sum( traveltime .* capacities .* inv_pos(1 - f./ capacities) - traveltime .* capacities)
    subject to
    traffic * f == nu
    f <= capacities
    f >= 0
cvx_end
f
%% 1 e) Wardrop equilibrium 
cvx_begin
    variable fwar(nbrOfLinks)
    minimize sum( traveltime .* capacities .* log( (capacities - fwar) ./ capacities ) *-1 )
    subject to
    traffic * fwar == nu
    fwar <= capacities
    fwar >= 0
cvx_end
fwar
%%  1 f) Wardrop equilibrium with tolls
w = f .* traveltime .* capacities .* inv_pos(capacities - f) .* inv_pos(capacities -f);
cvx_begin
    variable fwar_toll(nbrOfLinks)
    minimize sum( traveltime .* capacities .* (log(capacities) - log(capacities - fwar_toll)) + fwar_toll .* w  )
    subject to
    traffic * fwar_toll == nu
    fwar_toll <= capacities
    fwar_toll >= 0
cvx_end
fwar_toll

%sum(fwar), sum(fwar_toll))
%% 1 g) Total cost is additive delay from the total delay at freeflow,
cvx_begin
    variable f_2(nbrOfLinks)
    minimize sum( traveltime .* quad_over_lin(f_2, capacities - f_2, 0))
    traffic * f_2 == nu
    0 <= f_2 <= capacities
cvx_end
w = f_2 .* traveltime .* capacities ./ (capacities - f_2).^2 - traveltime
f_2
%% 1 g) 2 construct tolls to such that the new wardrop equilibrium 
cvx_begin 
    variable f_2_war(nbrOfLinks)
    D = traveltime .* capacities .* (log(capacities) - log(capacities - f_2_war));
    minimize sum( D + f_2_war .* w)
    traffic * f_2_war == nu
0 <= f_2_war <= capacities
cvx_end
f_2_war
