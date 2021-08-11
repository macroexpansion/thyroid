def scalars_layout(metric, class_metric):
    classes = ["tirad 3", "tirad 4"]
    tmp = {}
    for idx, item in enumerate(classes):
        tmp.update({classes[idx]: class_metric[idx]})
    return {"macro": metric, **tmp}


data = scalars_layout(1, [2, 3])
print(data)
