from bokeh.models.widgets import TextInput, Button, Select, Slider

statistic = Select(title="Sort By Statistic", value="Median", options=["Median", "Mean", "Max", "Min", "Standard Deviation"])
# metric = Select(title="Sort By Metric", value="Score", options=["Score", "None"])
num_pipes_text = TextInput(value="100", title="Number of Pipelines Tested (Text)")
optimizer_progression_range = TextInput(value="1,100", title="Optimizer Progression Range (Start,End)")
num_pipes = Slider(start=1, end=100, value=100, step=1, title="Number of Pipelines Tested")
run_optimizer_progression_button = Button(label="Run Optimizer Progression", button_type="success")
optimizer_progression_speed = TextInput(value="1", title="Optimizer Progression Delay (Seconds)")
run_optimizer_progression_button = Button(label="Run Optimizer Progression", button_type="success")
optimizer_progression_max_time = TextInput(value="10", title="Max Time For Optimizer Progression (Sec)")
optimizer_progression_size = TextInput(value="1", title="Number of Pipelines Per Progression")
generate_gif_button = Button(label="Generate Gif", button_type="success")
gif_x_range = TextInput(value="0,1", title="Set Gif x-axis (left,right)")
gif_speed = TextInput(value="500", title="Gif Speed (milliseconds)")

show_best_pipeline = Select(title="Show Best Pipeline (Green)", value="No", options=["No", "Yes"])

show_avg_pipeline = Select(title="Show Average Selected Pipeline (Red)", value="No", options=["No", "Yes"])

show_legend_option = Select(title="Show Legend", value="Yes", options=["Yes", "No"])

show_legend_option2 = Select(title="Show Legend", value="Yes", options=["Yes", "No"])

show_legend_option3 = Select(title="Show Legend", value="No", options=["Yes", "No"])

show_best_pipeline_train = Select(title="Show Best Pipeline (Green)", value="No", options=["No", "Yes"])

show_avg_pipeline_train = Select(title="Show Average Selected Pipeline (Red)", value="No", options=["No", "Yes"])

show_train_pruning = Select(title="Show Train Pruning Cutoff", value="No", options=["No", "Yes"])

show_ipa_pruning = Select(title="Show IPA Pruning Cutoff", value="No", options=["No", "Yes"])

show_ipa2_pruning = Select(title="Show IPA2 Pruning Cutoff", value="No", options=["No", "Yes"])