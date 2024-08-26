clear
close all
% Reading reference gradients matirx
ref_grad = readtable('results/k3/reference_gradient.tsv', ...
    'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
ref_ecc = readtable('results/k3/ref_ecc.tsv', ...
    'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
ref_ecc = ref_ecc(401:432, :);
ref_grad = ref_grad(401:432, :);
ref_grad.Properties.VariableNames{1} = 'roi';
sub_labels = load("subcorticalLabels.mat");
label_map = containers.Map(sub_labels.subcorticalLabels, 1:32);
fig_path = fullfile(pwd, "figures/reference");
num_pc = 3;
for pc = 1:num_pc
    labels = table(sub_labels.subcorticalLabels, zeros(32, 1), 'VariableNames', {'roi', 'loading'});
    for r = 1:height(ref_grad)
        roi = ref_grad.roi{r};  % Get the 'roi' value
        loading_val = ref_grad{r, strcat("g", num2str(pc))};  % Convert cell value to double
        labels.loading(label_map(roi)) = loading_val;
    end
    filename = strcat("gradients_PC", num2str(pc));
    fig = figure("Name", "PC" + num2str(pc));
    colormap(getTwilightShiftedCmap);
    plotSubcortex(-labels.loading, [-2.5, 2.5]);
    print(fig, fullfile(fig_path, filename), '-dsvg');
end
labels = table(sub_labels.subcorticalLabels, zeros(32, 1), 'VariableNames', {'roi', 'distance'});
for r = 1:height(ref_ecc)
    roi = ref_ecc.roi{r};  % Get the 'roi' value
    dist_val = ref_ecc{r, "distance"};  % Convert cell value to double
    labels.distance(label_map(roi)) = dist_val;
end
filename = strcat("ref_ecc");
fig = figure("Name", "ref_ecc");
colormap(getViridisCmap);
plotSubcortex(labels.distance, [-1, 4]);
print(fig, fullfile(fig_path, filename), '-dsvg');