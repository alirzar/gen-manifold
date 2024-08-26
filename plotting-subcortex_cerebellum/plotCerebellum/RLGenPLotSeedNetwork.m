clear
close all
% Define the paths for figures and results
figPath = fullfile(pwd, "figures/seed-network/");
resPath = fullfile(pwd, "results/k3/seed-network/");
% Pattern for matching files - in this case, files with a '.tsv' extension
pattern = '*error_correlations.tsv';
cmap = readmatrix('colormap.csv');
% Get a list of files in the specified directory matching the pattern
fileList = dir(fullfile(resPath, pattern));
% Loop through each file in the fileList
for i = 1:length(fileList)
    % Read the table from the file with specific options for file type and delimiter
    res = readtable(fullfile(resPath, fileList(i).name), ...
        "FileType","text",  'Delimiter', '\t', 'VariableNamingRule', 'preserve');
    resCereb = res(contains(res.network, 'region'), :);
    % Extract the numeric part of the regions
    region_numbers = cellfun(@(x) str2double(x(strfind(x, 'region')+length('region'):end)), resCereb.network);
    % Sort the regions based on the numeric part
    [~, sorted_indices] = sort(region_numbers);
    % Sort the regions array using the sorted indices
    resCereb = resCereb(sorted_indices, :);
    % Create a new figure for plotting, naming it after the current file
    fig = figure('Name', fileList(i).name);
    % Set the colormap for the figure
    % colormap(getSeedNetCmap(256, 0, 1));
    colormap(cmap)
    % Create a figure filename by removing the '.tsv' extension and adding ''
    figName = strcat(erase(fileList(i).name, ".tsv"), "");
    % Plot the subcortex data with color scaling
    plotCerebellum(resCereb.r, [-0.4, 0.4])
    % Save the figure to the specified path with high resolution
    print(fig, fullfile(figPath, figName), '-dsvg');
end

function cmap = getSeedNetCmap(m, start_frac, end_frac)
    if nargin < 1
        m = 256; % Default number of colors
    end
    if nargin < 3
        start_frac = 1;
        end_frac = 0;
    end

    % Full seismic colormap (blue-white-red)
    mid = floor(m/2);
    r = [(0:1:mid-1)/mid, ones(1, mid)];
    g = [(0:1:mid-1)/mid, (mid-1:-1:0)/mid];
    b = [ones(1, mid), (mid-1:-1:0)/mid];

    seismic = [r', g', b'];

    % Change the last color to yellow
    %seismic(end, :) = [1, 1, 0];  % RGB for yellow

    % Calculate start and end indices for the subset
    start_idx = max(1, floor(start_frac * m));
    end_idx = min(m, ceil(end_frac * m));

    % Extract the subset
    cmap = seismic(start_idx:end_idx, :);
end
