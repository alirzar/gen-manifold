clear
close all
left_early = readtable('results/k3/behavior/lefttransfer-early-corrected_error_correlations.tsv', ...
    'FileType','text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
right_early = readtable('results/k3/behavior/rightlearning-early-corrected_error_correlations.tsv', ...
    'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule','preserve');
cmap = readmatrix('colormap.csv');
% Define the path for saving figures
fig_path = fullfile(pwd, "figures/behavior");
left_early_cereb_regions = left_early(433:464, :);
right_early_cereb_regions = right_early(433:464, :);

fig = figure("Name", "lefttransfer-early-corrected_error_corr");
colormap(cmap);
plotCerebellum(left_early_cereb_regions.r, [-0.4, 0.4]);

% Save the figure as a SVG file
filename = "lefttransfer-early-corrected_error_correlation_map";
print(fig, fullfile(fig_path, filename), '-dsvg')

fig = figure("Name", 'rightlearning-early-corrected_error_corr');
colormap(cmap)
plotCerebellum(right_early_cereb_regions.r, [-0.4, 0.4])

filename = 'rightlearning-early-corrected_error_correlation_map';
print(fig, fullfile(fig_path, filename), '-dsvg')