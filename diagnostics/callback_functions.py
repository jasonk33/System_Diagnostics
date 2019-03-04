from widgets import *

# Callback function to make the legend visible for given plot
def show_legend_callback(figure, button_value):
    if button_value == "Yes":
        figure.legend.visible = True
    elif button_value == "No":
        figure.legend.visible = False

        
# Callback function to show the best overall or the average of the selected pipelines for incremental results
def incremental_pipe_callback(pipe_points, pipe_line, button_value):
    if button_value == "Yes":
        pipe_points.visible = True
        pipe_line.visible = True
    elif button_value == "No":
        pipe_points.visible = False
        pipe_line.visible = False
        

# Callback function to show where training would be pruned
def pruning_cutoff_callback(pruning_glyph, button_value):
    if button_value == "Yes":
        pruning_glyph.visible = True
    elif button_value == "No":
        pruning_glyph.visible = False

def change_num_pipes(num_pipes, num_pipes_text_value, optimizer_progression_range):
    num_pipes.value = int(num_pipes_text_value)
    optimizer_progression_range.value = "{},{}".format(num_pipes_text_value,num_pipes.end)