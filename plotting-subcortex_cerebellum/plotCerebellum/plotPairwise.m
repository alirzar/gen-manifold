function f = plotPairwise(data, fig_path, prefix)
if nargin < 3
    prefix = ""; % Default prefix is an empty string
end

groups = findgroups(data.A, data.B);

% Initialize an empty cell array to store the processed tables
list_ = {};
order = epoch_order;
% Loop over each group
for g = unique(groups)'
    % Select the rows of the current group
    group_data = data(groups == g, :);
    % Determine the order of conditions in the contrast
    if order(group_data.A{1}) > order(group_data.B{1})
        contrast = [group_data.A{1}, '_vs_', group_data.B{1}];
        T_values = group_data.T;
        group_data = removevars(group_data,'Contrast');
        group_data.new_T = T_values;
        group_data = addvars(group_data, T_values, 'NewVariableNames', {contrast}, 'Before', 'roi_ix');
    else
        contrast = [group_data.B{1}, '_vs_', group_data.A{1}];
        T_values = -group_data.T;
        group_data = removevars(group_data,'Contrast');
        group_data = addvars(group_data, T_values, 'NewVariableNames', {contrast}, 'Before', 'roi_ix');
    end
        
    % Append the processed table to the list
    list_{end+1} = group_data;
end

for i = 1:numel(list_)
    temp_table = list_{i};
    temp_table = temp_table(temp_table.roi_ix > 432, :);
    tvals = zeros(32, 1);
    if nnz(temp_table.sig_corrected) > 0
        temp_table{:,2}(temp_table.sig_corrected ~= 1) = 0;
        temp_table{:, 2} = (temp_table{:, 2} - min(temp_table{:, 2})) / (max(temp_table{:, 2}) - min(temp_table{:, 2}));
        tvals(temp_table.roi_ix - 432) = temp_table{:, 2};
        fig = figure('Name', temp_table.Properties.VariableNames{2});
        colormap(getRedBlueCmap);
        plotCerebellum(tvals, [-.5, .5])
        disp(prefix)
        filename = strcat(prefix, 'ecc_ttests_', temp_table.Properties.VariableNames{2});
        print(fig, fullfile(fig_path, filename), '-dsvg');
    end
end


function order = epoch_order()
    order = containers.Map(...
        {'rest', 'leftbaseline', 'rightbaseline', ...
         'rightlearning-early', 'rightlearning-late', ...
         'lefttransfer-early', 'lefttransfer-late', ...
         'baseline', 'early', 'late', ...
         'left', 'right'}, ...
        [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 1, 2]);
end
end