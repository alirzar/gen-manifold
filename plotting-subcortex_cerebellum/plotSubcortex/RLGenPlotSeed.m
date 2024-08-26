clear
close all
% Define the paths for figures and results
figPath = fullfile(pwd, "figures/seed/");
resPath = fullfile(pwd, "results/k3/seed/");
% Pattern for matching files - in this case, files with a '.tsv' extension
pattern = '*.tsv';
% Get a list of files in the specified directory matching the pattern
fileList = dir(fullfile(resPath, pattern));
% Loop through each file in the fileList
for i = 1:length(fileList)
    % Read the table from the file with specific options for file type and delimiter
    res = readtable(fullfile(resPath, fileList(i).name), ...
        "FileType","text",  'Delimiter', '\t', 'VariableNamingRule', 'preserve');

    % Load subcortical labels from a MAT file
    sub_labels = load("subcorticalLabels.mat");
    % Create a map to associate subcortical labels with indices
    label_map = containers.Map(sub_labels.subcorticalLabels, 1:32);
    % Filter results for specific roi indices
    resSub = res(res.roi_ix > 400 & res.roi_ix < 433, :);
    % Initialize a table for T values with zeros
    tvals = table(sub_labels.subcorticalLabels, zeros(32, 1), 'VariableNames', {'roi', 'T'});
    % Map roi values in resSub to indices using the label_map
    roi_indices = cellfun(@(roi) label_map(roi), resSub.roi);
    % Update the T values in the tvals table using the mapped indices
    tvals.T(roi_indices) = resSub.T;
    % Create a new figure for plotting, naming it after the current file
    fig = figure('Name', fileList(i).name);
    % Set the colormap for the figure
    colormap(getSeismicCmap(256, 0, 1));
    % Create a figure filename by removing the '.tsv' extension and adding ''
    figName = strcat(erase(fileList(i).name, ".tsv"), "");
    % Determine the maximum absolute value for color scaling
    cmax = ceil(max(abs(min(tvals.T)), max(tvals.T)));
    if contains(figName, "lh") || contains(figName, "rh")
        parts = strsplit(figName, '_');
        region = parts{end};
        tvals(strcmp(tvals.roi, region), "T") = {100};
    end
    if contains(figName, 'RH_DorsAttnB_FEF') || contains(figName, 'RH_SomMotB_S2')
        figName = strrep(figName, 'right_vs_left', 'left_vs_right');
        tvals.T = -tvals.T;
    end
    % Plot the subcortex data with color scaling
    colorbar
    plotSubcortex(tvals.T, [-cmax, cmax])
    % Save the figure to the specified path with high resolution
    print(fig, fullfile(figPath, figName), '-dsvg');
end

