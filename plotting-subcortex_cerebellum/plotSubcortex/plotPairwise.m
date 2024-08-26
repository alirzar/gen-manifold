function f = plotPairwise(data, fig_path, prefix)
if nargin < 3
    prefix = ""; % Default prefix is an empty string
end

sub_labels = load("subcorticalLabels.mat");
label_map = containers.Map(sub_labels.subcorticalLabels, 1:32);

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

for i = 1:length(list_)
    temp_table = list_{i};
    temp_table = temp_table(temp_table.roi_ix > 400 & temp_table.roi_ix < 433, :);
    tvals = table(sub_labels.subcorticalLabels, zeros(32, 1), 'VariableNames', {'roi', 'T'});
    if nnz(temp_table.sig_corrected) > 0    
        for r = 1:height(temp_table)
            if temp_table.sig_corrected(r) == 1
                roi = temp_table.roi{r};  % Get the 'roi' value
                tvals.T(label_map(roi)) = temp_table{r, 2};
            end
        end

        fig = figure('Name', temp_table.Properties.VariableNames{2});
        colormap(getRedBlueCmap);
        plotSubcortex(tvals.T, [-6, 6])
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