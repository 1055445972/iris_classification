from django.shortcuts import render
from pyecharts import options as opts
from .getResult import get_confusion_matrix
from pyecharts.charts import HeatMap
from pyecharts.charts import HeatMap
from pyecharts import options as opts
from pyecharts.faker import Faker
from copy import deepcopy
import random

def home(request):
    min_dist_cm,knn_cm,id3_cm,min_dist_accuracy,knn_accuracy,id3_accuracy= get_confusion_matrix()
    print(min_dist_cm)
    print(knn_cm)
    print(id3_cm)
    xlabels = ["山鸢尾","变色鸢尾","维吉尼亚鸢尾"]
    ylabels = xlabels[::-1]
    c1 = (
        HeatMap()
        .add_xaxis(xlabels)
        .add_yaxis(
            "",
            ylabels,
            min_dist_cm,
            label_opts=opts.LabelOpts(is_show=True, position="inside"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="最小距离分类器类 准确率:"+str(min_dist_accuracy),pos_left='center'),
            visualmap_opts=opts.VisualMapOpts(pos_left='right',min_=0, 
            max_=1,
            range_color=["#0FF", "#50F"])
        )
    )
    chart_render1 = deepcopy(c1.render_embed(pos_left='right'))

    
    c2 = (
        HeatMap()
        .add_xaxis(xlabels)
        .add_yaxis(
            "",
            ylabels,
            knn_cm,
            label_opts=opts.LabelOpts(is_show=True, position="inside"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="k近邻分类器类 准确率:"+str(knn_accuracy),pos_left='center'),
            visualmap_opts=opts.VisualMapOpts(pos_left='right',min_=0, 
            max_=1,
            range_color=["#006", "#140"]),
        )
    )
    chart_render2 = deepcopy(c2.render_embed())

    value = [[i, j, random.randint(0, 50)] for i in range(24) for j in range(7)]
    c3 = (
        HeatMap()
        .add_xaxis(xlabels)
        .add_yaxis(
            "",
            ylabels,
            id3_cm,
            label_opts=opts.LabelOpts(is_show=True, position="inside"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="ID3分类器 准确率:"+str(id3_accuracy),pos_left='center'),
            visualmap_opts=opts.VisualMapOpts(pos_left='right',min_=0, 
            max_=1,
            range_color=["#F1F", "#50F"]),
        )
    )
    chart_render3 = deepcopy(c3.render_embed())


    return render(request, 'index.html', {'chart_render1': chart_render1,
                                          'chart_render2': chart_render2,
                                          'chart_render3': chart_render3,})

home(None)
