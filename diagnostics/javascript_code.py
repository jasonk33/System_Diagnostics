pipeline_selection = """
    
        // Get data from input sources
        var inds = cb_obj.selected.indices;
        var d1 = cb_obj.data;
        var d2 = selected_source_points.data;
        var d4 = selected_source_points_plot.data;
        var d5 = source_progression.data;
        var d6 = selected_source_points_line_plot.data;
        var d7 = selected_source_points_avg_pipe.data;
        var d8 = source_pruning.data;
        var d9 = selected_source_pruning.data;
    
        // Initialize empty array for selected points data
        IDs = [];
        d4['ID'] = [];
        d4['validation_error'] = [];
        d4['train_error'] = [];
        d4['sample_size'] = [];
        d4['color'] = [];
        d4['name'] = [];
        
        // Initialize empty array for selected points data
        for (var key in d1){
            d2[key] = []
        }
        
        // Add IDs
        for (var i = 0; i < inds.length; i++) {
            curr_ID = d1['ID'][inds[i]]
            IDs.push(curr_ID)
            for (var key in d1){
                d2[key].push(d1[key][inds[i]])
            }
        }
        
        // Initialize empty array for selected points data (pruning)
        d9['ID'] = []
        d9['train_pruning_error'] = [];
        d9['ipa_pruning_error'] = [];
        d9['ipa2_pruning_error'] = [];
        d9['train_pruning_samp_size'] = [];
        d9['ipa_pruning_samp_size'] = [];
        d9['ipa2_pruning_samp_size'] = [];
        d9['color'] = [];
        
        // Iterate over pruning data
        for (var ii = 0; ii < d8['train_pruning_error'].length; ii++) {
        
            // Iterate over IDs (of selected points)
            for (var iii = 0; iii < IDs.length; iii++) {
            
                // If ID matches, add data to selected points data (pruning)
                if (IDs[iii] == d8['ID'][ii]) {
                    d9['ID'].push(IDs[iii])
                    d9['train_pruning_error'].push(d8['train_pruning_error'][ii])
                    d9['ipa_pruning_error'].push(d8['ipa_pruning_error'][ii])
                    d9['ipa2_pruning_error'].push(d8['ipa2_pruning_error'][ii])
                    d9['train_pruning_samp_size'].push(d8['train_pruning_samp_size'][ii])
                    d9['ipa_pruning_samp_size'].push(d8['ipa_pruning_samp_size'][ii])
                    d9['ipa2_pruning_samp_size'].push(d8['ipa2_pruning_samp_size'][ii])
                    d9['color'].push(d8['color'][ii])
                }
            }
        }
        
        // Initialize empty dictionaries for group data
        score_groups = {};
        score_groups_train = {};
        samp_groups = {};
        color_groups = {};
        name_groups = {};
        samp_score_avg = {};
        samp_score_avg_train = {};
        
        // Iterate over progression data
        for (var ii = 0; ii < d5['validation_error'].length; ii++) {
        
            // Iterate over IDs (selected points)
            for (var iii = 0; iii < IDs.length; iii++) {
            
                // If ID matches, add data
                if (IDs[iii] == d5['ID'][ii]) {
                    validation_error = d5['validation_error'][ii]
                    train_error = d5['train_error'][ii]
                    sample_size = d5['sample_size'][ii]
                    color = d5['color'][ii]
                    name = d5['name'][ii]
                    ID = d5['ID'][ii]
                    d4['ID'].push(ID)
                    d4['validation_error'].push(validation_error)
                    d4['train_error'].push(train_error)
                    d4['sample_size'].push(sample_size)
                    d4['color'].push(color)
                    d4['name'].push(name)
                    
                    // If ID in group
                    if (ID in score_groups) {
                        score_groups[ID].push(validation_error)
                        score_groups_train[ID].push(train_error)
                        samp_groups[ID].push(sample_size)
                    } else {
                        score_groups[ID] = [validation_error]
                        score_groups_train[ID] = [train_error]
                        samp_groups[ID] = [sample_size]
                        color_groups[ID] = color
                        name_groups[ID] = name
                    }
                    
                    // If sample size in score average
                    if (sample_size in samp_score_avg) {
                        samp_score_avg[sample_size].push(validation_error)
                        samp_score_avg_train[sample_size].push(train_error)
                    } else {
                        samp_score_avg[sample_size] = [validation_error]
                        samp_score_avg_train[sample_size] = [train_error]
                    }
                    
                }
            }
        }
        
        // Initialize empty arrays for partial score data
        part_score_line = [];
        part_samp_line = [];
        part_score_line_train = [];
        line_colors = [];
        names = [];
        IDS = [];
        
        // Iterate over groups
        for(var key in score_groups) {
            var score = score_groups[key]
            var score_train = score_groups_train[key]
            var samp = samp_groups[key]
            part_score_line.push(score)
            part_score_line_train.push(score_train)
            part_samp_line.push(samp)
            line_colors.push(color_groups[key])
            names.push(name_groups[key])
            IDS.push(key)
        }
        
        // Add progression data
        d6['validation_error'] = part_score_line;
        d6['train_error'] = part_score_line_train;
        d6['sample_size'] = part_samp_line;
        d6['color'] = line_colors;
        d6['name'] = names;
        d6['ID'] = IDS;
        
        // Initialize empty arrays for average data
        part_score_avg = [];
        part_score_avg_train = [];
        part_samp_avg = [];
        names_avg = [];
        IDS_avg = [];
        
        // Iterate over average data
        for(var key in samp_score_avg) {
            part_samp_avg.push(key)
            var scores = samp_score_avg[key]
            var scores_train = samp_score_avg_train[key]
            sum = scores.reduce(function(a, b) { return a + b; });
            sum_train = scores_train.reduce(function(a, b) { return a + b; });
            avg = sum / scores.length;
            avg_train = sum_train / scores_train.length;
            part_score_avg.push(avg)
            part_score_avg_train.push(avg_train)
            names_avg.push("NA")
            IDS_avg.push("NA")
        }
        
        // Add data
        d7['validation_error'] = part_score_avg;
        d7['train_error'] = part_score_avg_train;
        d7['sample_size'] = part_samp_avg;
        d7['name'] = names_avg;
        d7['ID'] = IDS_avg;
        
        // Emit changes to source data
        selected_source_points.change.emit();
        selected_source_points_plot.change.emit();
        selected_source_points_line_plot.change.emit();
        selected_source_points_avg_pipe.change.emit();
        selected_source_pruning.change.emit();
    """

file_input = """
import * as p from "core/properties"
import {LayoutDOM, LayoutDOMView} from "models/layouts/layout_dom"

export class FileInputView extends LayoutDOMView
  initialize: (options) ->
    super(options)
    input = document.createElement("input")
    input.type = "file"
    input.onchange = () =>
      @model.value = input.value
    @el.appendChild(input)

export class FileInput extends LayoutDOM
  default_view: FileInputView
  type: "FileInput"
  @define {
    value: [ p.String ]
  }
"""