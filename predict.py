
# ======================================================================================================================
# Detects and segments disease symptoms in image crops
# ======================================================================================================================

from leaf import models
from leaf.inference import Predictor
from leaf.visualization import FlattenedVisualizer, Path
from leaf import get_model_urls_for_config, download_models_for_config, download_test_images
import glob
from pathlib import Path

# pre-download models and test image
urls = get_model_urls_for_config(config_name='flattened_leaves', config_path='config')
downloaded = download_models_for_config()
downloaded = download_models_for_config(config_name="canopy_landscape")
downloaded = download_models_for_config(config_name="flattened_leaves")
downloaded = download_test_images(root="test/images")

# # test models (requires gpu)
# models.test()

# intialize predictor
pred = Predictor(config_name='flattened_leaves',
                 symptoms_seg_params={'model_name': 'tracking_latest',
                                      'use_gpu': False},
                 symptoms_det_params={'model_name': 'tracking_latest',
                                      'use_gpu': False,
                                      'keypoints_thresh': 0.18}
)


# list directories to process
dir_to_process = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/WW40/1260/test")
# dir_to_process = Path("/agroscope/Data-Work-CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/WW40/1260/test")

# predict
pred.predict(images_src=f'{dir_to_process}/inference_crops', export_dst=f'{dir_to_process}/predictions')

# visualize
vis = FlattenedVisualizer(
    src_root=f'{dir_to_process}/predictions', 
    rgb_root=f'{dir_to_process}/inference_crops', 
    export_root=f'{dir_to_process}/predictions')
vis.visualize()

# # loop over directories
# # predict and visualize
# for d in dirs_to_process:
#     print(d)
#     pred.predict(images_src=f'{d}/crop', export_dst=f'{d}/predictions')
    # vis = FlattenedVisualizer(src_root=f'{d}/predictions', rgb_root=f'{d}/crop', export_root=f'{d}/predictions')
    # vis.visualize()