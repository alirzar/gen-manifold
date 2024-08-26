clear
close all
% Reading reference gradients matirx for 3 PCs
ref_grad = readtable('results/k3/reference_gradient.tsv', ...
    'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
ref_ecc = readtable('results/k3/ref_ecc.tsv', ...
    'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
ref_ecc = ref_ecc(433:464, :);
ref_grad = ref_grad(433:464, :);
ref_grad.Properties.VariableNames{1} = 'roi';
fig_path = fullfile(pwd, "figures/reference");
num_pc = 3;
for pc = 1:num_pc
    labels = ref_grad{:, strcat("g", num2str(pc))};  % Convert cell value to double
    filename = strcat("gradients_PC", num2str(pc));
    fig = figure("Name", "PC" + num2str(pc));
    colormap(getTwilightShiftedCmap);
    plotCerebellum(labels, [-4, 4]);
    print(fig, fullfile(fig_path, filename), '-dsvg');
end
filename = strcat("ref_ecc");
fig = figure("Name", "ref_ecc");
colormap(getViridisCmap);
plotCerebellum(ref_ecc.distance, [-.1, 3.3]);
print(fig, fullfile(fig_path, filename), '-dsvg');

