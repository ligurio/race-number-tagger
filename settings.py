
MODEL_FILE = 'data/model.h5'
MODEL_WEIGHTS_FILE = 'data/model-weights.h5'
CSV_RESULTS = 'data/raw_results.csv'
ORIGINAL_IMAGES = 'data/original_data'
IMAGE_SUBSETS = { 'train': 50, 'validate': 35, 'test': 15 }  # proportion of images in subsets (percents)
BOX_HEIGHT = 157    # 75th percentile 190.25, mean average 153.22972972972974
BOX_WIDTH = 68      # 75th percentile 118.0, mean average 110.66216216216216
PERCENTILE = 75
