import pandas as pd

def make_plot(figure, source, source_points):
    
    # stems
    figure.segment("upper", "methods", "q3", "methods", line_color="black", source=source)
    figure.segment("lower", "methods", "q1", "methods", line_color="black", source=source)

    # boxes
    figure.hbar("methods", 0.7, "q2", "q3", fill_color="#E08E79", line_color="black", fill_alpha=.75, source=source)
    figure.hbar("methods", 0.7, "q1", "q2", fill_color="#3B8686", line_color="black", fill_alpha=.75, source=source)

    # whiskers (almost-0 height rects simpler than segments)
    figure.rect("lower", "methods", 0.01, 0.2, line_color="black", source=source)
    figure.rect("upper", "methods", 0.01, 0.2, line_color="black", source=source)

    figure.circle('y', 'x', size='size', color='color', alpha=0.33, source=source_points, name="pipe")
    

    figure.ygrid.grid_line_color = None
    figure.xgrid.grid_line_color = "white"
    figure.grid.grid_line_width = 2
    figure.yaxis.major_label_text_font_size="12pt"
    figure.legend.location = "top_right"
    
def make_plot_for_saving(figure, source_data, source_point_data):
    
    # stems
    figure.segment(source_data["upper"], source_data["methods"], source_data["q3"], source_data["methods"], line_color="black")
    figure.segment(source_data["lower"], source_data["methods"], source_data["q1"], source_data["methods"], line_color="black")

    # boxes
    figure.hbar(source_data["methods"], 0.7, source_data["q2"], source_data["q3"], fill_color="#E08E79", line_color="black", fill_alpha=.75)
    figure.hbar(source_data["methods"], 0.7, source_data["q1"], source_data["q2"], fill_color="#3B8686", line_color="black", fill_alpha=.75)

    # whiskers (almost-0 height rects simpler than segments)
    figure.rect(source_data["lower"], source_data["methods"], 0.01, 0.2, line_color="black")
    figure.rect(source_data["upper"], source_data["methods"], 0.01, 0.2, line_color="black")

    figure.circle(source_point_data['y'], source_point_data['x'], size=source_point_data['size'], color=source_point_data['color'], alpha=0.75, name="pipe")

    figure.ygrid.grid_line_color = None
    figure.xgrid.grid_line_color = "white"
    figure.grid.grid_line_width = 2
    figure.yaxis.major_label_text_font_size="12pt"
    figure.legend.location = "top_right"
    
def munge_boxplot_data(pipeline_data, statistic):
        groups = pipeline_data.groupby('Name')
        sort_stat = statistic.value
        ascending = False
        if sort_stat == "Median":
            groups_sort = pipeline_data.groupby('Name').median().reset_index()
        elif sort_stat == "Mean":
            groups_sort = pipeline_data.groupby('Name').mean().reset_index()
        elif sort_stat == "Max":
            groups_sort = pipeline_data.groupby('Name').max().reset_index()
        elif sort_stat == "Min":
            groups_sort = pipeline_data.groupby('Name').min().reset_index()
            ascending = False
        elif sort_stat == "Standard Deviation":
            groups_sort = pipeline_data.groupby('Name').std().reset_index()
        sorter = list(groups_sort.sort_values('Accuracy', ascending=ascending)['Name'])
        methods = list((groups.groups.keys()))
        methods = sorter
        q1 = groups.quantile(q=0.25)
        q2 = groups.quantile(q=0.5)
        q3 = groups.quantile(q=0.75)
        q1['name'] = pd.Categorical(q1.index, sorter)
        q1 = q1.sort_values('name').iloc[:,0:1]
        q2['name'] = pd.Categorical(q2.index, sorter)
        q2 = q2.sort_values('name').iloc[:,0:1]
        q3['name'] = pd.Categorical(q3.index, sorter)
        q3 = q3.sort_values('name').iloc[:,0:1]
        iqr = q3 - q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr
        
        qmin = groups.quantile(q=0.00)
        qmax = groups.quantile(q=1.00)
        qmin['name'] = pd.Categorical(qmin.index, sorter)
        qmin = qmin.sort_values('name').iloc[:,0:1]
        qmax['name'] = pd.Categorical(qmax.index, sorter)
        qmax = qmax.sort_values('name').iloc[:,0:1]
        upper.Accuracy = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'Accuracy']),upper.Accuracy)]
        lower.Accuracy = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'Accuracy']),lower.Accuracy)]
        
        return methods, q1, q2, q3, upper, lower