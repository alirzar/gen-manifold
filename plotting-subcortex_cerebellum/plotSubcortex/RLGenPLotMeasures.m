clear
close all
% Reading reference gradients matirx for 3 PCs
measures = readtable('results/k3/fc_correlations/fc_connectivity_measures.tsv', ...
    'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
measures = measures(strcmp(measures.network, 'Subcortex'), :);
fig_path = fullfile(pwd, "figures/measures/");

% Load subcortical labels from a MAT file
sub_labels = load("subcorticalLabels.mat");
% Create a map to associate subcortical labels with indices
label_map = containers.Map(sub_labels.subcorticalLabels, 1:32);

% Initialize a table for values with zeros
vals = table(sub_labels.subcorticalLabels, zeros(32, 1), 'VariableNames', {'roi', 'strength'});
% Map roi values in resSub to indices using the label_map
roi_indices = cellfun(@(roi) label_map(roi), measures.roi);
% Update the values in the tvals table using the mapped indices
vals.strength(roi_indices) = measures.strength;
fig = figure("Name", "strength");
colormap(getViridisCmap);
plotSubcortex(vals.strength, [1.34, 3.41]);
colorbar
print(fig, fullfile(fig_path, "strength"), '-dsvg');

vals = table(sub_labels.subcorticalLabels, zeros(32, 1), 'VariableNames', {'roi', 'participation'});
% Map roi values in resSub to indices using the label_map
roi_indices = cellfun(@(roi) label_map(roi), measures.roi);
% Update the values in the tvals table using the mapped indices
vals.participation(roi_indices) = measures.participation;
fig = figure("Name", "participation");
colormap(getViridisCmap);
plotSubcortex(vals.participation, [0, 0.86]);
colorbar
print(fig, fullfile(fig_path, "participation"), '-dsvg');

vals = table(sub_labels.subcorticalLabels, zeros(32, 1), 'VariableNames', {'roi', 'module_degree'});
% Map roi values in resSub to indices using the label_map
roi_indices = cellfun(@(roi) label_map(roi), measures.roi);
% Update the values in the tvals table using the mapped indices
vals.module_degree(roi_indices) = measures.module_degree;
fig = figure("Name", "module_degree");
colormap(getViridisCmap);
plotSubcortex(vals.module_degree, [-2.51, 2.51]);
colorbar
print(fig, fullfile(fig_path, "module_degree"), '-dsvg');