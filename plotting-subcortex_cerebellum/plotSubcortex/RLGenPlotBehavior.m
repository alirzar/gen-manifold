clear
close all
left_early = readtable('results/k3/behavior/lefttransfer-early-corrected_error_correlations.tsv', ...
    'FileType','text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
right_early = readtable('results/k3/behavior/rightlearning-early-corrected_error_correlations.tsv', ...
    'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule','preserve');
sub_labels = load("subcorticalLabels.mat");
cmap = readmatrix('colormap.csv');
label_map = containers.Map(sub_labels.subcorticalLabels, 1:32);

% Define the path for saving figures
fig_path = fullfile(pwd, "figures/behavior");
% Obtain the rows corresponding to subcortical regions from the mixed ANOVA stats
left_sub_regions = left_early(401:432, :);

rvals = table(sub_labels.subcorticalLabels, zeros(32, 1), 'VariableNames', {'roi', 'r'});
for i = 1:height(left_sub_regions)
    roi = left_sub_regions.roi{i};  % Get the 'roi' value
    rvals.r(label_map(roi)) = left_sub_regions.r(i);
end

fig = figure("Name", "lefttransfer-early-corrected_error_corr");
colormap(cmap);
plotSubcortex(rvals.r, [-0.45, 0.45]);
% Save the figure as a PNG file
filename = "lefttransfer-early-corrected_error_correlation_map";
print(fig, fullfile(fig_path, filename), '-dsvg')

right_sub_regions = right_early(401:432, :);

rvals = table(sub_labels.subcorticalLabels, zeros(32, 1), 'VariableNames', {'roi', 'r'});
for i = 1:height(right_sub_regions)
    roi = right_sub_regions.roi{i};
    rvals.r(label_map(roi)) = right_sub_regions.r(i);
end

fig = figure("Name", 'rightlearning-early-corrected_error_corr');
colormap(cmap)
plotSubcortex(rvals.r, [-0.45, 0.45])

filename = 'rightlearning-early-corrected_error_correlation_map';
print(fig, fullfile(fig_path, filename), '-dsvg')