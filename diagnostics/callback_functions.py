from widgets import *

def show_legend(p3):
    if show_legend_option.value == "Yes":
        p3.legend.visible = True
    elif show_legend_option.value == "No":
        p3.legend.visible = False
        
def show_legend2(p):
    if show_legend_option2.value == "Yes":
        p.legend.visible = True
    elif show_legend_option2.value == "No":
        p.legend.visible = False
        
def show_legend3(p6):
    if show_legend_option3.value == "Yes":
        p6.legend.visible = True
    elif show_legend_option3.value == "No":
        p6.legend.visible = False

def best_pipe(best_pipe_plot, best_pipe_plot_train, best_pipe_line_plot, best_pipe_line_plot_train):
    if show_best_pipeline.value == "Yes":
        best_pipe_plot.visible = True
        best_pipe_line_plot.visible = True
    elif show_best_pipeline.value == "No":
        best_pipe_plot.visible = False
        best_pipe_line_plot.visible = False
    
    if show_best_pipeline_train.value == "Yes":
        best_pipe_plot_train.visible = True
        best_pipe_line_plot_train.visible = True
    elif show_best_pipeline_train.value == "No":
        best_pipe_plot_train.visible = False
        best_pipe_line_plot_train.visible = False

def avg_pipe(avg_pipe_plot, avg_pipe_plot_train, avg_pipe_line_plot, avg_pipe_line_plot_train):
    if show_avg_pipeline.value == "Yes":
        avg_pipe_plot.visible = True
        avg_pipe_line_plot.visible = True
    elif show_avg_pipeline.value == "No":
        avg_pipe_plot.visible = False
        avg_pipe_line_plot.visible = False
        
    if show_avg_pipeline_train.value == "Yes":
        avg_pipe_plot_train.visible = True
        avg_pipe_line_plot_train.visible = True
    elif show_avg_pipeline_train.value == "No":
        avg_pipe_plot_train.visible = False
        avg_pipe_line_plot_train.visible = False
        
def pruning_cutoff(train_pruning_glyph, ipa_pruning_glyph, ipa2_pruning_glyph):
    if show_train_pruning.value == "Yes":
        train_pruning_glyph.visible = True
    elif show_train_pruning.value == "No":
        train_pruning_glyph.visible = False
    
    if show_ipa_pruning.value == "Yes":
        ipa_pruning_glyph.visible = True
    elif show_ipa_pruning.value == "No":
        ipa_pruning_glyph.visible = False
        
    if show_ipa2_pruning.value == "Yes":
        ipa2_pruning_glyph.visible = True
    elif show_ipa2_pruning.value == "No":
        ipa2_pruning_glyph.visible = False