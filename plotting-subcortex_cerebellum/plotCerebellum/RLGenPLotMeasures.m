clear
close all
% Reading reference gradients matirx for 3 PCs
measures = readtable('results/k3/fc_correlations/fc_connectivity_measures.tsv', ...
    'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
measures = measures(strcmp(measures.network, 'Cerebellum'), :);
fig_path = fullfile(pwd, "figures/measures");
vals = measures.strength;
filename = strcat("strength");
fig = figure("Name", "strength");
colormap(getViridisCmap);
plotCerebellum(vals, [1.34, 3.41]);
print(fig, fullfile(fig_path, filename), '-dsvg');

vals = measures.participation;
filename = strcat("participation");
fig = figure("Name", "participation");
colormap(getViridisCmap);
plotCerebellum(vals, [0, 0.86]);
print(fig, fullfile(fig_path, filename), '-dsvg');

vals = measures.module_degree;
filename = strcat("module_degree");
fig = figure("Name", "module_degree");
colormap(getViridisCmap);
plotCerebellum(vals, [-2.51, 2.51]);
print(fig, fullfile(fig_path, filename), '-dsvg');