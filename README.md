# Label-free segmentation and tracking of single stem cells using an efficient deep learning model

## Abstract


<p align="center">
  <img src="docs/Fig1.png" width="550" title="hover text">
</p>


# Datasets

To download our datasets go to https://deepseas.org/datasets/ or:

Link to [dataset_for_cell_segmentation](http://google.com)

Link to [dataset_for_nucleus_segmentation](http://google.com)

Link to [dataset_for_cell_tracking](http://google.com)

Link to [dataset_for_cell_cycle_tracking](http://google.com)


# Results

## Segmentation Results

              | IOU         | Precision     | Recall     |
| :----:      |    :----:   |        :----: |  :----:    |
| Cell        | 90%         | 95%           | 94%        | 
| Nucleus     | 73%         | 84%           | 85%        |


## Detection Results

              | IOU         | Precision     | Recall     |
| :----:      |    :----:   |        :----: |  :----:    |
| Cell        | 93%         | 90%           | 92%        | 
| Nucleus     | 73%         | 68%           | 71%        |

## Frame-by-frame cell tracking Results

              | Overal      | Hard          | Easy       | single cell | birth    |
| :----:      |    :----:   |        :----: |  :----:    |:----:       |:----:    |
| Precision   | 89%         | 97%           | 88%        | 89%      |89%          |
| Recall      | 100%         | 99%           | 99%        |100%       | 100%      |

## Cell cycle tracking Results

|      MOTA   | MT          | ML            | Precision  | Recall      | Frag     |IDS       |
| :----:      |    :----:   |        :----: |  :----:    |:----:       |:----:    |:----:    |
| 96%         | 92%         | 0.0           | 99%        | 98%         |11        |5         |

