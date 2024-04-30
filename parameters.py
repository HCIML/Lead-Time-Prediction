import pathlib
cache_location = pathlib.Path.joinpath(pathlib.Path.home(), pathlib.Path('IBP_ML').stem, "Inventory")

meta_params = {
"features": ['OriginSite', 'DestinationSite', 'material', 'DAYS', 'Mode'],
"granularity": ['OriginSite', 'DestinationSite', 'Mode'],
"time_level": 'WEEKS',
"label": 'LEADTIME',
"date": 'DATETIME',
"today": "2021-01-01",
"date_format": '%Y-%m-%d',
"history": 50,
"confidence": .99,
"outliers": True,
"analyze": True,
"chart": 30, # False, 30
"predict": False,
"min_group_size": 10,
"demo": True,
}

outlier_params = {
    "remove_outliers": True,
    "cap": None,
    "outlier_removal_method": 'confidence',
    "visualize_outliers_flag": True,
    "vis_sort_by_count": True,
}

box_plot = {
    "group": meta_params["granularity"],
    "min_group_sample": None,  # minimum group size to consider
    "remove_small_groups": 10,
    "visualize": 30,  # chunk size
}

forest = {
    "group": meta_params["granularity"],  # groups to be analyzed for outliers
    "min_group_sample": None,  # groups of less size are considered all as inliers
    "remove_small_groups": 10,  # groups of less size are considered all as outliers (removed completely)
    "contamination": 0.001,  # percentage of outliers (per group, if groups are given), default: 'auto'
    "max_group_outliers": 100,  # outlier size limit per group
    "visualize": 30,  # chunk size
    "density": True,
    "scale": "width",
    "smoothing_kernel": .1,  # density smoothing kernel
    "scatter": None  # 'o'
}

confidence = {
    "group": meta_params["granularity"],
    "significance": 3
}