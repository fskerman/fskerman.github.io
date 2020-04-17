%%%%% Task 1 %%%%%

filename = "ia-email-univ.mtx";
fid = fopen(filename);

fgetl(fid); % comment

line = fgetl(fid); % # vertices, .., # edges
metadata = str2num(line);

V = metadata(1);
E = metadata(3);

n1 = zeros(1, E);
n2 = zeros(1, E);

for i = 1:E
    nodes = str2num(fgetl(fid));
    n1(i) = nodes(1);
    n2(i) = nodes(2);
end

fclose(fid);

%%%%% Task 2 %%%%%

G = graph(n1, n2);
deg = degree(G);
[y, x] = groupcounts(deg);

figure(1);
plot(x, y);
title('Degree distribution');

%%%%% Task 3 %%%%%

% v = randsample(V,50); % doesn't work without the Machine Learning Toolbox
sample_size = 50;
indices = randperm(V);
S = indices(1:sample_size);
deg2 = deg(S);
[y, x] = groupcounts(deg2);

figure(2);
plot(x, y);
title('Degree distribution for a sample of 50 of data');

%%%%% Task 4 %%%%%

S2 = [];
for i = 1:length(S)
    S2 = [S2; neighbors(G, S(i))];
end
S2 = unique(S2);

deg3 = deg(S2);
[y, x] = groupcounts(deg3);

figure(3);
plot(x, y);
title('Degree distribution for neighbors of S');

