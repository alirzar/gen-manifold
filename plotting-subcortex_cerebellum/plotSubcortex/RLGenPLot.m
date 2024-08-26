clear
close all
sub_labels = load("subcorticalLabels.mat");
label_map = containers.Map(sub_labels.subcorticalLabels, 1:32);
% Define the path for saving figures
fig_path = fullfile(pwd, "figures/generalization");
pc = 3;
% Reading ANOVA + postHoc ttest results from text files
filename = sprintf('results/k%d/ecc_anova_stats.tsv', pc);
anova_stats = readtable(filename, 'FileType', 'text', 'Delimiter',...
    '\t', 'VariableNamingRule', 'preserve'); 
% Obtain the rows corresponding to subcortical regions from the mixed ANOVA stats
subcortical_regions = anova_stats(anova_stats.roi_ix > 400 & anova_stats.roi_ix < 433, :);
uniqueSources = unique(anova_stats.Source);
% Loop over each unique source
for i = 1:length(uniqueSources)
    s = uniqueSources{i};
    temp_table = subcortical_regions(strcmp(subcortical_regions.Source, s), :);
    fvals = table(sub_labels.subcorticalLabels, zeros(32, 1), 'VariableNames', {'roi', 'F'});
    for r = 1:height(temp_table)
        if temp_table.sig_corrected(r) == 1
            roi = temp_table.roi{r};  % Get the 'roi' value
            fvals.F(label_map(roi)) = temp_table.F(r);
        end
    end

    % If there are any significant results, create a figure and plot them
    if nnz(fvals.F) > 0
        fig = figure("Name", "anova_" + s);
        colormap(getCmap);
        plotSubcortex(fvals.F, [6, 32]);

        % Save the figure as a PNG file
        figname = sprintf("_anova_%dPCs", pc);
        figname = s + figname;
        print(fig, fullfile(fig_path, figname), '-dsvg')
    end
end
ttest_stats = readtable('results/k3/ecc_ttest_stats.tsv', ...
    'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule','preserve');
uniqueContrasts = unique(ttest_stats.Contrast);
for c = uniqueContrasts'
    if strcmp(c, "time * hand")
        temp_table = ttest_stats(strcmp(ttest_stats.Contrast, c{1}), :);
        uniqueTime = unique(temp_table.time);
        for t = uniqueTime'
            temp_table = ttest_stats(strcmp(ttest_stats.time, t{1}), :);
            prefix = strcat(c{1}, "_", t{1});
            plotPairwise(temp_table, fig_path, prefix)
        end
    else
        temp_table = ttest_stats(strcmp(ttest_stats.Contrast, c{1}), :);
        prefix = strcat(c{1}, "_");
        plotPairwise(temp_table, fig_path, prefix)
    end
end
fig = figure("Name", 'zeros');
filename = 'subcortex_zeros';
colormap('gray')
plotSubcortex(0.85.*ones(1, 32)', [0, 1])
print(fig, fullfile(pwd, filename), '-dsvg')

