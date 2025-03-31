load("IOdownload.mat")

%% Sweden eigenvector centrality
swe = io.swe2000; % Load data
g = digraph(swe); % Create a directed graph

% Create subgraph of the largest connected component
[bin, binsize] = conncomp(g);
% idx = binsize(bin) == max(binsize);
idx = find(binsize(bin) == max(binsize));
SG = subgraph(g, idx);
plot(SG);

% Fidning the largets eigenvalue and corresponding 
W = adjacency(SG, 'weighted'); % Removed "weighted"
[vec, val] = eigs(W');
[~, I] = sort(diag(val), "descend");
[~, I_sort] = sort(vec(:,1));
real_idx = idx(I_sort);

% Printing the results
disp('The three most central sectors in Sweden are (eig): ')
for i = 1:3
fprintf('\t Sector: %s\n', name{real_idx(i)})
end
fprintf('\n')

%% Indonesia eigenvector centrality
idn = io.idn2000; % Load data
g = digraph(idn); % Create a directed graph

% Create subgraph of the largest connected component
[bin, binsize] = conncomp(g); 
%idx = binsize(bin) == max(binsize);
idx = find(binsize(bin) == max(binsize));
SG = subgraph(g, idx);
plot(SG);

% Fidning the largets eigenvalue and corresponding 
W = adjacency(SG, 'weighted');
[vec, val] = eigs(W');
[~, I] = sort(diag(val), "descend");
[~, I_sort] = sort(vec(:,1));

real_idx = idx(I_sort);

% Printing the results
disp('The three most central sectors in Indonesia are (eig): ')
for i = 1:3
fprintf('\t Sector: %s\n', name{real_idx(i)})
end
fprintf('\n')

