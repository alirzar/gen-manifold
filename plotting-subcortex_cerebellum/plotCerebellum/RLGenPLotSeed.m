% Cerebellum
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

    resCer = res(res.roi_ix > 432, :);
    % Determine the maximum absolute value for color scaling
    cmax = ceil(max(abs(min(resCer.T)), max(resCer.T)));
    % Create a new figure for plotting, naming it after the current file
    fig = figure('Name', fileList(i).name);
    % Set the colormap for the figure
    colormap(getSeismicCmap(256, 0, 1));
    % Create a figure filename by removing the '.tsv' extension and adding ''
    figName = strcat(erase(fileList(i).name, ".tsv"), "");
    if contains(figName, "region")
        region = regexp(figName, 'region\d+', 'match');
        resCer(strcmp(resCer.roi, region), "T") = {100};
    end
    if contains(figName, "RH_DorsAttnB_FEF") || contains(figName, 'RH_SomMotB_S2')
        figName = strrep(figName, 'right_vs_left', 'left_vs_right');
        resCer.T = -resCer.T;
    end
    % Plot the cerebellum data with color scaling
    plotCerebellum(resCer.T, [-cmax, cmax])
    % Save the figure to the specified path with high resolution
    print(fig, fullfile(figPath, figName), '-dsvg');
end