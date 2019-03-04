import numpy as np
import pandas as pd
import json
import os
from bokeh.layouts import layout, widgetbox, column, row
from bokeh.models.widgets import TextInput, Button, TableColumn, Paragraph, DataTable, Panel, Tabs, Select, RadioButtonGroup, CheckboxButtonGroup, Slider
from bokeh.io import curdoc, export_svgs, export_png
from bokeh.plotting import figure, output_file, save
from bokeh.models.tools import BoxZoomTool, HoverTool
from bokeh.models import ColumnDataSource, CustomJS, Button, LayoutDOM, Range1d, Legend, LegendItem
from bokeh.core.properties import String
from bokeh.palettes import Viridis256, Plasma256
from random import shuffle
from time import sleep,time
import re
from PIL import Image
import shutil
from functools import partial

from javascript_code import *
from callback_functions import *
from widgets import *
from utils import *

########
"""
Add pruning cutoffs to data
"""
########

PLOT_HEIGHT = 700
PLOT_WIDTH = 1000
colors = (Viridis256 + Plasma256)
shuffle(colors)
colors = colors * 100
tag_counts = {"random-general":0, "meta-learning":0, "random-data-specific":0, "cost_model-weighed_random":0}

all_params = ['max_features','learning_rate','C','power_t','dual','n_neighbors','max_samples','penalty','max_depth','tol','average','epsilon','p','weights','subsample','min_samples_leaf','bootstrap','min_samples_split','criterion','loss','alpha','n_estimators','max_iter','min_child_weight']


def upload_data(num_pipes, num_pipes_text, optimizer_progression_range):
    global data
    path = 'aps-i2pa/' + input.value[12:]
    with open(path) as f:
        data = json.load(f)
    num_poss_pipes = len(data)
    num_pipes.value = num_poss_pipes
    num_pipes_text.value = str(num_poss_pipes)
    num_pipes.end = num_poss_pipes
    optimizer_progression_range.value = "1,{}".format(num_poss_pipes)
    munge_data()
    pipeline_results_boxplot_figure.y_range.factors = source.data['methods']
    min_samp = min(source_progression.data['sample_size'])
    max_samp = max(source_progression.data['sample_size'])
    if max_samp < min_samp:
        max_samp = min_samp
        min_samp = 0
    incremental_results_validation_figure.x_range = Range1d(min_samp, max_samp)


def munge_data(pipe_num=None, return_data=False, return_tag_counts=False):
    pipeline_dict = {}
    pipeline_dict_keys = ['Name', 'Accuracy', 'validation_error', 'train_error', 'sample_size', 'ID', 'color', 'boxplot_color', 'training_time', 'train_pruning', 'ipa_pruning', 'ipa2_pruning', 'best_so_far']
    for name in pipeline_dict_keys:
        pipeline_dict[name] = []

    scores = []
    for idx,pipeline in enumerate(data):
        progression = json.loads(pipeline['metrics']['progression'])
        score = list(progression['validation_error'].values())[-1]
        if score is not None and progression is not None:
            scores.append(score)
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    x0 = max(Q1 - 5 * IQR, min(scores))
    x1 = min(Q3 + 5 * IQR, max(scores))
    pipeline_results_boxplot_figure.x_range = Range1d(x0, x1)
    for param in all_params:
        pipeline_dict[param] = []
    best_so_far = np.inf
    global tag_counts
    tag_counts = {"random-general":0, "meta-learning":0, "random-data-specific":0, "cost_model-weighed_random":0}
    for idx,pipeline in enumerate(data):
        progression = json.loads(pipeline['metrics']['progression'])
        score = list(progression['validation_error'].values())[-1]
        if (score is not None) and (progression is not None) and (score > Q1 - 5*IQR) and (score < Q3 + 5*IQR):
            best_so_far = min(best_so_far, score)
            pipeline_dict['best_so_far'].append(best_so_far)
#             train_pruning = pipeline['metrics']['train_pruning']
#             ipa_pruning = pipeline['metrics']['ipa_pruning']
#             ipa2_pruning = pipeline['metrics']['ipa2_pruning']
            train_pruning = list(progression['sample_size'].values())[-1]
            ipa_pruning = list(progression['sample_size'].values())[-1]
            ipa2_pruning =  list(progression['sample_size'].values())[-1]
            name = pipeline['pipeline']['steps'][-1]['primitive']['name']
            tag = pipeline['tags']['pipeline_arm']
            if tag == "random-general":
                pipeline_dict['boxplot_color'].append("Blue")
            elif tag == "meta-learning":
                pipeline_dict['boxplot_color'].append("Purple")
            elif tag == "random-data-specific":
                pipeline_dict['boxplot_color'].append("Green")
            elif tag == "cost_model-weighed_random":
                pipeline_dict['boxplot_color'].append("Red")
            else:
                pipeline_dict['boxplot_color'].append("Black")
                
            pipeline_dict['Accuracy'].append(score)
            pipeline_dict['Name'].append(name)
            validation_error = list(progression['validation_error'].values())
            train_error = list(progression['train_error'].values())
            sample_sizes = list(progression['sample_size'].values())
            training_time = list(progression['train_time'].values())[-1]
            pipeline_dict['validation_error'].append(validation_error)
            pipeline_dict['train_error'].append(train_error)
            pipeline_dict['sample_size'].append(sample_sizes)
            pipeline_dict['ID'].append(idx)
            pipeline_dict['color'].append(colors[idx])
            pipeline_dict['training_time'].append(training_time)
            
            pipeline_dict['train_pruning'].append(train_pruning)
            pipeline_dict['ipa_pruning'].append(ipa_pruning)
            pipeline_dict['ipa2_pruning'].append(ipa2_pruning)
            
            if 'humanReadableParameters' in pipeline['pipeline']['steps'][-1]['primitive']:
                pipe_params = pipeline['pipeline']['steps'][-1]['primitive']['humanReadableParameters']
            else:
                pipe_params = {}
            for param in all_params:
                if param in pipe_params:
                    pipeline_dict[param].append(pipe_params[param])
                else:
                    pipeline_dict[param].append(None)
                    
    pipe_type = set(pipeline_dict['Name'])
    pipe_type_color = {}
    for idx,pipe in enumerate(pipe_type):
        pipe_type_color[pipe] = colors[idx]
    pipeline_dict['estimator_color'] = []
    for pipe in pipeline_dict['Name']:
        pipeline_dict['estimator_color'].append(pipe_type_color[pipe])

    pipeline_data = pd.DataFrame(pipeline_dict)
    if not pipe_num:
        pipe_num = num_pipes.value
    pipeline_data = pipeline_data[pipeline_data['ID'] < pipe_num]
    
    name = pipeline_dict['Name']
    score = pipeline_dict['Accuracy']
    validation_error = pipeline_dict['validation_error']
    train_error = pipeline_dict['train_error']
    sample_size = pipeline_dict['sample_size']
    ID = pipeline_dict['ID']
    ID_repeat = []
    part_err = []
    part_train_err = []
    samp_size = []
    colors_partial = []
    name_partial = []
    
    train_pruning_error = []
    ipa_pruning_error = []
    ipa2_pruning_error = []
    ID_pruning = []
    color_pruning = []
    for idx,part_validation_error in enumerate(validation_error):
        ID_pruning.append(ID[idx])
        color_pruning.append(colors[idx])
        for idx2,err in enumerate(part_validation_error):
            samp_size_single_run = int(sample_size[idx][idx2])
            train_err_single_run = train_error[idx][idx2]
            part_err.append(err)
            part_train_err.append((train_error[idx][idx2]))
            ID_repeat.append(ID[idx])
            samp_size.append(samp_size_single_run)
            colors_partial.append(colors[idx])
            name_partial.append(pipeline_dict['Name'][idx])
            if samp_size_single_run == pipeline_dict['train_pruning'][idx]:
                train_pruning_error.append(train_err_single_run)
            if samp_size_single_run == pipeline_dict['ipa_pruning'][idx]:
                ipa_pruning_error.append(train_err_single_run)
            if samp_size_single_run == pipeline_dict['ipa2_pruning'][idx]:
                ipa2_pruning_error.append(train_err_single_run)
    
    progression_data = {'ID': ID_repeat, 'validation_error': part_err, 'train_error': part_train_err, 'sample_size': samp_size, 'color': colors_partial, 'name': name_partial}
    
    pruning_data = {'ID': ID_pruning, 'train_pruning_error': train_pruning_error, 'train_pruning_samp_size': pipeline_dict['train_pruning'], 'ipa_pruning_error': ipa_pruning_error, 'ipa_pruning_samp_size': pipeline_dict['ipa_pruning'], 'ipa2_pruning_error': ipa2_pruning_error, 'ipa2_pruning_samp_size': pipeline_dict['ipa2_pruning'], 'color': color_pruning}
    
    source_pruning.data = pruning_data
    
    source_progression.data = progression_data

    methods, q1, q2, q3, upper, lower = munge_boxplot_data(pipeline_data, statistic)
    
    source_data = dict(
        methods = methods,
        q1 = q1.Accuracy,
        q2 = q2.Accuracy,
        q3 = q3.Accuracy,
        upper = upper.Accuracy,
        lower = lower.Accuracy
    )
    source.data = source_data
    
    source_dict = {'x':name, 'y':score, 'color':pipeline_dict['boxplot_color'], 'validation_error':validation_error, 'train_error':train_error, 'sample_size': sample_size, 'ID': ID, 'training_time': pipeline_dict['training_time'], 'estimator_color': pipeline_dict['estimator_color'], 'size' : [6] * len(name), 'best_so_far': pipeline_dict['best_so_far']}
    
    for param in all_params:
        source_dict[param] = pipeline_dict[param]
        
    if not pipe_num:
        pipe_num = num_pipes.value
    removable_ids = sorted([idx for idx,item in enumerate(source_dict['ID']) if item >= pipe_num], reverse=True)
    for idx in removable_ids:
        for key in source_dict.keys():
            del source_dict[key][idx]
            
    for color in source_dict['color']:
        if color == "Blue":
            tag_counts["random-general"] += 1
        elif color == "Purple":
            tag_counts["meta-learning"] += 1
        elif color == "Green":
            tag_counts["random-data-specific"] += 1
        elif color == "Red":
            tag_counts["cost_model-weighed_random"] += 1
            
    last_pipe = np.argmax(source_dict['ID'])
    source_dict['size'][last_pipe] = 17
    #source_dict['color'][last_pipe] = "Red"
    
    source_points.data = source_dict
    
    best_pipe_idx = np.argmin(pipeline_dict['Accuracy'])
    num_partials = len(pipeline_dict['sample_size'][best_pipe_idx])
    
    source_points_best_pipe.data = {'sample_size': pipeline_dict['sample_size'][best_pipe_idx], 'validation_error': pipeline_dict['validation_error'][best_pipe_idx], 'train_error':pipeline_dict['train_error'][best_pipe_idx], 'color': [pipeline_dict['color'][best_pipe_idx]]*num_partials, 'name': [pipeline_dict['Name'][best_pipe_idx]]*num_partials, 'ID': [pipeline_dict['ID'][best_pipe_idx]]*num_partials}
    
    source_point_training_time_best.data = {'training_time': [pipeline_dict['training_time'][best_pipe_idx]], 
                                            'score': [pipeline_dict['Accuracy'][best_pipe_idx]], 'x': [pipeline_dict['Name'][best_pipe_idx]]}
    
    if return_data:
        if return_tag_counts:
            return source_data, source_dict, tag_counts
        else:
            return source_data, source_dict
    elif return_tag_counts:
        return tag_counts


class FileInput(LayoutDOM):
    __implementation__ = file_input
    value = String()

input = FileInput()

# def upload():
#     print(input.value)
upload_botton = Button(label="Upload")
upload_botton.on_click(partial(upload_data, num_pipes, num_pipes_text, optimizer_progression_range))


source = ColumnDataSource(data=dict(methods=[], q1=[], q2=[], q3=[], upper=[], lower=[]))
source_points = ColumnDataSource(data=dict(x=[], y=[], color=[], sample_size=[], validation_error=[], train_error=[], ID=[], training_time=[], estimator_color=[], size=[], best_so_far=[]))
selected_source_points = ColumnDataSource(data=dict(x=[], y=[], color=[], sample_size=[], validation_error=[], train_error=[]))
selected_source_points_plot = ColumnDataSource(data=dict(ID=[], validation_error=[], train_error=[], sample_size=[], color=[], name=[]))
selected_source_points_line_plot = ColumnDataSource(data=dict(ID=[], validation_error=[], train_error=[], sample_size=[], color=[], name=[]))
source_progression = ColumnDataSource(data=dict(ID=[], validation_error=[], train_error=[], sample_size=[], color=[], name=[], train_pruning=[], ipa_pruning=[], ipa2_pruning=[]))
source_pruning = ColumnDataSource(data=dict(ID=[], train_pruning_error=[], ipa_pruning_error=[], ipa2_pruning_error=[], train_pruning_samp_size=[], ipa_pruning_samp_size=[], ipa2_pruning_samp_size=[], color=[]))
selected_source_pruning = ColumnDataSource(data=dict(ID=[], train_pruning_error=[], ipa_pruning_error=[], ipa2_pruning_error=[], train_pruning_samp_size=[], ipa_pruning_samp_size=[], ipa2_pruning_samp_size=[], color=[]))
source_points_best_pipe = ColumnDataSource(data=dict(color=[], sample_size=[], validation_error=[], train_error=[], ID=[], name=[]))
selected_source_points_avg_pipe = ColumnDataSource(data=dict(ID=[], validation_error=[], train_error=[], sample_size=[], name=[]))
source_point_training_time_best = ColumnDataSource(data=dict(score=[], training_time=[], x=[]))

source_points.callback = CustomJS(args=dict(selected_source_points=selected_source_points,selected_source_points_plot=selected_source_points_plot, source_progression=source_progression, selected_source_points_line_plot=selected_source_points_line_plot, selected_source_points_avg_pipe=selected_source_points_avg_pipe, source_pruning=source_pruning, selected_source_pruning=selected_source_pruning), code=pipeline_selection)

updateVals = CustomJS(args=dict(source_progression=source_progression, source_pruning=source_pruning), code="""""")

update_plot_button = widgetbox(Button(label="This Button Does Nothing", callback=updateVals))
update_plot_button2 = widgetbox(Button(label="This Button Does Nothing", callback=updateVals))

TOOLTIPS = [
    ("ID", "@ID"),
    ("Validation Error", "@validation_error"),
    ("Train Error", "@train_error"),
    ("Sample Size", "@sample_size"),
]

hover = HoverTool(names=["circle", "best_circle", "avg_circle"], tooltips=TOOLTIPS)
hover_train = HoverTool(names=["circle_train", "best_circle_train", "avg_circle_train"], tooltips=TOOLTIPS)
pipeline_results_boxplot_figure = figure(tools="pan,wheel_zoom,reset,box_select,save", background_fill_color="#EFE8E2", title="", y_range=source.data['methods'], plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT)
incremental_results_validation_figure = figure(tools=["pan,wheel_zoom,reset,box_select", hover], background_fill_color="#EFE8E2", title="", plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT)
incremental_results_validation_figure.circle(x='sample_size', y='validation_error', size=10, color='color', source=selected_source_points_plot, name="circle")
incremental_results_validation_figure.multi_line(xs='sample_size', ys='validation_error', color='color', line_width=3, source=selected_source_points_line_plot)
best_pipe_plot = incremental_results_validation_figure.circle(x='sample_size', y='validation_error', size=20, color='green', source=source_points_best_pipe, name="best_circle")
best_pipe_line_plot = incremental_results_validation_figure.line(x='sample_size', y='validation_error', line_width=6, color='green', source=source_points_best_pipe)
avg_pipe_plot = incremental_results_validation_figure.circle(x='sample_size', y='validation_error', size=20, color='red', source=selected_source_points_avg_pipe, name="avg_circle")
avg_pipe_line_plot = incremental_results_validation_figure.line(x='sample_size', y='validation_error', line_width=6, color='red', source=selected_source_points_avg_pipe)
training_time_figure = figure(tools=["pan,wheel_zoom,reset,box_select"], background_fill_color="#EFE8E2", title="", plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, tooltips=[("Name", "@x")])
training_time_figure.scatter(x='training_time', y='y', color='estimator_color', source=source_points, legend='x')
training_time_figure.legend.location = "top_right"
training_time_figure.x_range = Range1d(0, 10)

incremental_results_training_figure = figure(tools=["pan,wheel_zoom,reset,box_select", hover_train], background_fill_color="#EFE8E2", title="", plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT)
incremental_results_training_figure.circle(x='sample_size', y='train_error', size=10, color='color', source=selected_source_points_plot, name="circle_train")
incremental_results_training_figure.multi_line(xs='sample_size', ys='train_error', color='color', line_width=3, source=selected_source_points_line_plot)
train_pruning_glyph = incremental_results_training_figure.x(x='train_pruning_samp_size', y='train_pruning_error', color='color', size=33, line_width=5, source=selected_source_pruning)
ipa_pruning_glyph = incremental_results_training_figure.x(x='ipa_pruning_samp_size', y='ipa_pruning_error', color='color', size=33, line_width=5, source=selected_source_pruning)
ipa2_pruning_glyph = incremental_results_training_figure.x(x='ipa2_pruning_samp_size', y='ipa2_pruning_error', color='color', size=33, line_width=5, source=selected_source_pruning)
best_pipe_plot_train = incremental_results_training_figure.circle(x='sample_size', y='train_error', size=20, color='green', source=source_points_best_pipe, name="best_circle_train")
best_pipe_line_plot_train = incremental_results_training_figure.line(x='sample_size', y='train_error', line_width=6, color='green', source=source_points_best_pipe)
avg_pipe_plot_train = incremental_results_training_figure.circle(x='sample_size', y='train_error', size=20, color='red', source=selected_source_points_avg_pipe, name="avg_circle_train")
avg_pipe_line_plot_train = incremental_results_training_figure.line(x='sample_size', y='train_error', line_width=6, color='red', source=selected_source_points_avg_pipe)
optimizer_improvement_figure = figure(tools=["pan,wheel_zoom,reset,box_select"], background_fill_color="#EFE8E2", title="", plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, tooltips=[("Name", "@x")])
optimizer_improvement_figure.line(x='ID', y='best_so_far', line_width=3, color='Blue', source=source_points, name="best_so_far_line")
optimizer_improvement_figure.circle(x='ID', y='best_so_far', size=7, color='Red', source=source_points, name="best_so_far_circle")
optimizer_incremental_results_figure = figure(tools=["pan,wheel_zoom,reset,box_select"], background_fill_color="#EFE8E2", title="", plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, tooltips=[("Name", "@x")])
p6_circles = optimizer_incremental_results_figure.circle(x='ID', y='y', size=7, color='estimator_color', source=source_points, name="pipe_num_vs_score", legend='x')
optimizer_incremental_results_figure.legend.location = "top_right"

# Initialize glyphs to be hidden
best_pipe_plot.visible=best_pipe_line_plot.visible=avg_pipe_plot.visible=avg_pipe_line_plot.visible=train_pruning_glyph.visible=ipa_pruning_glyph.visible=ipa2_pruning_glyph.visible=best_pipe_plot_train.visible=best_pipe_line_plot_train.visible=avg_pipe_plot_train.visible=avg_pipe_line_plot_train.visible=optimizer_incremental_results_figure.legend.visible=False


legend_circles_boxplot = []
for color in ["Blue", "Purple", "Green", "Red"]:
    legend_circles_boxplot.append(pipeline_results_boxplot_figure.circle([], [], color=color, fill_alpha=0.75, size=6))

legend = Legend(items=[("Random General", [legend_circles_boxplot[0]]), ("Meta Learning" , [legend_circles_boxplot[1]]), ("Random Data Specific" , [legend_circles_boxplot[2]]), ("Cost Model Weighed Random" , [legend_circles_boxplot[3]])])
pipeline_results_boxplot_figure.add_layout(legend)
pipeline_results_boxplot_figure.legend.location = "top_right"

make_plot(pipeline_results_boxplot_figure, source, source_points)

def change_plot(change_y_range=True):
    munge_data()
    
    pipeline_results_boxplot_figure.legend.items = [LegendItem(label="Random General: {}".format(tag_counts['random-general']), renderers=[legend_circles_boxplot[0]]), LegendItem(label="Meta Learning: {}".format(tag_counts['meta-learning']), renderers=[legend_circles_boxplot[1]]), LegendItem(label="Random Data Specific: {}".format(tag_counts['random-data-specific']), renderers=[legend_circles_boxplot[2]]), LegendItem(label="Cost Model Weighed Random: {}".format(tag_counts['cost_model-weighed_random']), renderers=[legend_circles_boxplot[3]])]
    source.data = source.data
    sleep(0.5)
    
    if change_y_range:
        pipeline_results_boxplot_figure.y_range.factors = source.data['methods']
    
    
def run_optimizer_progression(optimizer_progression_range, optimizer_progression_size, optimizer_progression_max_time, num_pipes, optimizer_progression_speed):
    start_time = time()
    start_stop = optimizer_progression_range.value.split(",")
    for num_pipe in range(int(start_stop[0]),int(start_stop[1])+1,int(optimizer_progression_size.value)):
        if time() - start_time > float(optimizer_progression_max_time.value):
            break
        num_pipes.value = num_pipe
        sleep(float(optimizer_progression_speed.value))
        
    sleep(1)
    num_pipes.value = int(start_stop[1])
    
    
def generate_gif(input, gif_x_range, source, num_pipes, optimizer_progression_size, pipeline_results_boxplot_figure, legend_circles_boxplot, gif_speed):
    print("Beginning Generating Gif")
    
    try:
        os.mkdir("Plots")
    except:
        pass
    try:
        shutil.rmtree('Plots/{}'.format(input.value[12:-5]), ignore_errors=True)
    except:
        pass
    os.mkdir('Plots/{}'.format(input.value[12:-5]))
    gif_x_range_vals = str.split(gif_x_range.value, ",")

    munge_data()
    all_classifiers = source.data['methods']
    
    pipe_vals = list(range(num_pipes.start,num_pipes.end+1,int(optimizer_progression_size.value)))
    if pipe_vals[-1] != num_pipes.end:
        pipe_vals.append(num_pipes.end)
    for num_pipe in pipe_vals:
        source_data, source_points_data, tag_counts = munge_data(num_pipe, return_data=True, return_tag_counts=True)
#         p.y_range.factors = source_data['methods']
        
        plot = figure(tools="", background_fill_color="#EFE8E2", title="", y_range=all_classifiers, plot_width=1000, plot_height=600)
        
        plot.circle([], [], color="Blue", fill_alpha=0.75, size=6, legend="Random General: {}".format(tag_counts['random-general']))
        plot.circle([], [], color="Purple", fill_alpha=0.75, size=6, legend="Meta Learning: {}".format(tag_counts['meta-learning']))
        plot.circle([], [], color="Green", fill_alpha=0.75, size=6, legend="Random Data Specific: {}".format(tag_counts['random-data-specific']))
        plot.circle([], [], color="Red", fill_alpha=0.75, size=6, legend="Cost Model Weighed Random: {}".format(tag_counts['cost_model-weighed_random']))
        
        pipeline_results_boxplot_figure.legend.items = [LegendItem(label="Random General: {}".format(tag_counts['random-general']), renderers=[legend_circles_boxplot[0]]), LegendItem(label="Meta Learning: {}".format(tag_counts['meta-learning']), renderers=[legend_circles_boxplot[1]]), LegendItem(label="Random Data Specific: {}".format(tag_counts['random-data-specific']), renderers=[legend_circles_boxplot[2]]), LegendItem(label="Cost Model Weighed Random: {}".format(tag_counts['cost_model-weighed_random']), renderers=[legend_circles_boxplot[3]])]
        source.data = source.data
        
        make_plot_for_saving(plot, source_data, source_points_data)
        
        plot.x_range = Range1d(float(gif_x_range_vals[0]), float(gif_x_range_vals[1]))
        export_png(plot, filename="Plots/{}/plot_{}.png".format(input.value[12:-5], num_pipe))
        
        print("Finished Pipeline {} of {}".format(num_pipe, num_pipes.end))
    num_pipes.value = num_pipes.end
    
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]
    
    plots = [x for x in os.listdir('Plots/{}/'.format(input.value[12:-5])) if not x.startswith('.')]
    plots.sort(key=natural_keys)
    
    im1 = Image.open('Plots/{}/{}'.format(input.value[12:-5], plots[0]))
    images = [Image.open('Plots/{}/{}'.format(input.value[12:-5], plot_name)) for plot_name in plots[1:]]

    im1.save("Gifs/{}.gif".format(input.value[12:-5]), save_all=True, append_images=images, duration=int(gif_speed.value), loop=0)
    
    print("Finished Generating Gif")
    
    
# metric.on_change("value", lambda attr, old, new: change_plot())
statistic.on_change("value", lambda attr, old, new: change_plot())
num_pipes.on_change("value", lambda attr, old, new: change_plot(change_y_range=False))
run_optimizer_progression_button.on_click(partial(run_optimizer_progression, optimizer_progression_range, optimizer_progression_size, optimizer_progression_max_time, num_pipes, optimizer_progression_speed))
generate_gif_button.on_click(partial(generate_gif, input, gif_x_range, source, num_pipes, optimizer_progression_size, pipeline_results_boxplot_figure, legend_circles_boxplot, gif_speed))
num_pipes_text.on_change("value", lambda attr, old, new: change_num_pipes(num_pipes, num_pipes_text.value, optimizer_progression_range))


columns = [
    TableColumn(field="ID",title="ID"),
    TableColumn(field="x", title="Classifier"),
    TableColumn(field="y",title="Score")]

for param in all_params:
    columns.append(TableColumn(field=param, title=param))
    
data_tables = []
for _ in range(6):
    data_tables.append(widgetbox(DataTable(source=selected_source_points, columns=columns,sizing_mode = 'scale_width', fit_columns=False, width=1200)))


show_legend_option.on_change("value", lambda attr, old, new: show_legend_callback(training_time_figure, show_legend_option.value))
show_legend_option2.on_change("value", lambda attr, old, new: show_legend_callback(pipeline_results_boxplot_figure, show_legend_option2.value))
show_legend_option3.on_change("value", lambda attr, old, new: show_legend_callback(optimizer_incremental_results_figure, show_legend_option3.value))
show_avg_pipeline.on_change("value", lambda attr, old, new: incremental_pipe_callback(avg_pipe_plot, avg_pipe_line_plot, show_avg_pipeline.value))
show_avg_pipeline_train.on_change("value", lambda attr, old, new: incremental_pipe_callback(avg_pipe_plot_train, avg_pipe_line_plot_train, show_avg_pipeline_train.value))
show_best_pipeline.on_change("value", lambda attr, old, new: incremental_pipe_callback(best_pipe_plot, best_pipe_line_plot, show_best_pipeline.value))
show_best_pipeline_train.on_change("value", lambda attr, old, new: incremental_pipe_callback(best_pipe_plot_train, best_pipe_line_plot_train, show_best_pipeline_train.value))
show_train_pruning.on_change("value", lambda attr, old, new: pruning_cutoff_callback(train_pruning_glyph, show_train_pruning.value))
show_ipa_pruning.on_change("value", lambda attr, old, new: pruning_cutoff_callback(ipa_pruning_glyph, show_ipa_pruning.value))
show_ipa2_pruning.on_change("value", lambda attr, old, new: pruning_cutoff_callback(ipa2_pruning_glyph, show_ipa2_pruning.value))


buttons1 = widgetbox(show_legend_option2, statistic, optimizer_progression_speed, optimizer_progression_max_time, optimizer_progression_size, num_pipes_text, num_pipes, optimizer_progression_range, run_optimizer_progression_button, gif_x_range, gif_speed, generate_gif_button)
layer1 = layout(row(pipeline_results_boxplot_figure, buttons1), data_tables[0], sizing_mode='fixed')
layer2 = layout(row(incremental_results_validation_figure, column(widgetbox(show_best_pipeline), widgetbox(show_avg_pipeline))), data_tables[1], update_plot_button, sizing_mode='fixed')
layer3 = layout(row(training_time_figure, widgetbox(show_legend_option)), data_tables[2], sizing_mode='fixed')
layer4 = layout(row(incremental_results_training_figure, column(widgetbox(show_best_pipeline_train), widgetbox(show_avg_pipeline_train), widgetbox(show_train_pruning), widgetbox(show_ipa_pruning), widgetbox(show_ipa2_pruning))), data_tables[3], update_plot_button2, sizing_mode='fixed')
layer5 = layout(optimizer_improvement_figure, data_tables[4], sizing_mode='fixed')
layer6 = layout(row(optimizer_incremental_results_figure, widgetbox(show_legend_option3)), data_tables[5], sizing_mode='fixed')
tab1 = Panel(child=layer1,title="Pipeline Results")
tab2 = Panel(child=layer2,title="Pipeline Incremental Results (Validation)")
tab3 = Panel(child=layer3,title="Training Times")
tab4 = Panel(child=layer4,title="Pipeline Incremental Results (Training)")
tab5 = Panel(child=layer5,title="Optimizer Improvement")
tab6 = Panel(child=layer6,title="Optimizer Incremental Results")
tabs = Tabs(tabs=[ tab1, tab2, tab4, tab3, tab5, tab6])

curdoc().add_root(tabs)
curdoc().add_root(column(input, upload_botton))


# optimizer_progression_size.value = "10"
# try:
#     os.mkdir('Plots/tests2')
# except:
#     pass
# try:
#     os.mkdir('Gifs/tests2')
# except:
#     pass
# input.value = "            tests2/mimic.json"
# upload_data()
# generate_gif()
# for i in range(0,10):
#     input.value = "            tests2/mimic_{}.json".format(i)
#     upload_data()
#     generate_gif()
    
    
def main():
    pass

if __name__ == '__main__':
    main()
