%Modified from code by Toshi and Loren Shore at: https://blogs.mathworks.com/loren/2015/09/30/can-we-predict-a-breakup-social-network-analysis-with-matlab/

clear all
load karate.mat
G = graph(edges(:,1), edges(:,2));      % create a graph from edges
G.Nodes = table(name);                  % name the nodes
figure                                  % visualize the graph
plot(G);
title('Zachary''s Karate Club')

D = degree(G);                          % get degrees per node
mu = mean(D);                           % average degrees
figure
histogram(D);                           % plot histogram
hold on
title('Karate Club Degree Distribution')
xlabel('degrees (# of connections)'); ylabel('# of nodes');

