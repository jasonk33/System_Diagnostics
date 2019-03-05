from bokeh.models.widgets import TextInput, Button, Select, Slider
from constants import *

# metric = Select(title="Sort By Metric", value="Score", options=["Score", "None"])

# Statistic to sort by
statistic = Select(title="Sort By Statistic", value="Median", options=["Median", "Mean", "Max", "Min", "Standard Deviation"])

# Number of pipelines to display in boxplot (first x pipelines tested)
num_pipes_text = TextInput(value="100", title="Number of Pipelines Tested (Text)")

# Range of pipelines to progess over for boxplot visualization
optimizer_progression_range = TextInput(value="1,100", title="Optimizer Progression Range (Start,End)")

# Slider for number of pipelines tested being displayed in boxplots
num_pipes = Slider(start=1, end=100, value=100, step=1, title="Number of Pipelines Tested")

# Button to loop over the range of pipelines tested for boxplot visualization
run_optimizer_progression_button = Button(label="Run Optimizer Progression", button_type="success")

# Speed at which the loop runs for iterating over range of pipelines tested
optimizer_progression_speed = TextInput(value="1", title="Optimizer Progression Delay (Seconds)")

# Maximum amount of time to loop over range of pipelines (to avoid getting caught in a long loop)
optimizer_progression_max_time = TextInput(value="10", title="Max Time For Optimizer Progression (Sec)")

# Number of pipelines per iteration through range in boxplot visualization
optimizer_progression_size = TextInput(value="1", title="Number of Pipelines Per Progression")

# Button to generate gif of boxplots iterating over range of number of pipelines
generate_gif_button = Button(label="Generate Gif", button_type="success")

# X range for gif
gif_x_range = TextInput(value="0,1", title="Set Gif x-axis (left,right)")

# Speed between image in gif
gif_speed = TextInput(value="500", title="Gif Speed (milliseconds)")

# Show best pipeline for incremental results (validation)
show_best_pipeline = Select(title="Show Best Pipeline (Green)", value=NO_OPTION, options=[NO_OPTION, YES_OPTION])

# Show average of all pipelines selected for incremental results (validation)
show_avg_pipeline = Select(title="Show Average Selected Pipeline (Red)", value=NO_OPTION, options=[NO_OPTION, YES_OPTION])

# Show legend button
show_legend_option = Select(title="Show Legend", value=YES_OPTION, options=[YES_OPTION, NO_OPTION])

# Show legend button
show_legend_option2 = Select(title="Show Legend", value=YES_OPTION, options=[YES_OPTION, NO_OPTION])

# Show legend button
show_legend_option3 = Select(title="Show Legend", value=NO_OPTION, options=[YES_OPTION, NO_OPTION])

# Show best pipeline for incremental results (training)
show_best_pipeline_train = Select(title="Show Best Pipeline (Green)", value=NO_OPTION, options=[NO_OPTION, YES_OPTION])

# Show average of all pipelines selected for incremental results (training)
show_avg_pipeline_train = Select(title="Show Average Selected Pipeline (Red)", value=NO_OPTION, options=[NO_OPTION, YES_OPTION])

# Show where pruning would have stopping the training
show_train_pruning = Select(title="Show Train Pruning Cutoff", value=NO_OPTION, options=[NO_OPTION, YES_OPTION])

# Show where pruning would have stopping the training
show_ipa_pruning = Select(title="Show IPA Pruning Cutoff", value=NO_OPTION, options=[NO_OPTION, YES_OPTION])

# Show where pruning would have stopping the training
show_ipa2_pruning = Select(title="Show IPA2 Pruning Cutoff", value=NO_OPTION, options=[NO_OPTION, YES_OPTION])

