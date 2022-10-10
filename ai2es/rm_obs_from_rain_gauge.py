"""
remove images from rain gauge dataset that are obstructed based on hand labeled dataset
check both precip and no precip subfolders in rain_gauge var - hardcoded and need to manually switch
"""

import os

if __name__ == "__main__":
    hand_lableled = "/ai2es/codebook_dataset/combined_extra/obstructed"
    rain_gauge = "/ai2es/rain_gauge_labeled_no_obs/no_precip"
    for filename in os.listdir(rain_gauge):
        if filename in os.listdir(hand_lableled):
            os.remove(os.path.join(rain_gauge, filename))
