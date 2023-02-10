## Passage-Level Reading Behavior Model for Mobile Search 

Thanks for visiting.

This repository contains the code and data of our paper "A Passage-Level Reading Behavior Model for Mobile Search".

The code is based on the click model project by THUIR. Please refer to **[this repo](https://github.com/THUIR/click_model_for_mobile_search)** for detailed information on all arguments.

If you find our paper and resources useful, please kindly leave a star and cite our papers. Thanks!

```bibtex
@inproceedings{wu2023passage,
	author = {Wu, Zhijing and Mao, Jiaxin and Xu, Kedi and Song, Dandan and Huang, Heyan},
	title = {A Passage-Level Reading Behavior Model for Mobile Search},
	year = {2023},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	doi = {10.1145/3543507.3583343},
	booktitle = {Proceedings of the ACM Web Conference 2023},
	series = {WWW '23}
}
```
## How to use

1. Clone or download this repo.
2. Train and test the PRM with the given data or other datasets.

```
python test_reading_model.py ../data ../data -m MER-VPT-V3 -o ../output/lognormal --ignore_no_clicks --ignore_no_viewport --viewport_time -V 8
```
  - ``-m``: The model that you would like to run.
  - ``-o``: The path to output dictionary.
  - ``-V``: The viewport time model used in PRM: 
    - ``8``: PRM with log-normal
    - ``9``: PRM with gamma
    - ``10``: PRM with Weibull 

3. Run ``evaluate_reading_model.py`` to evaluate the PRM on passage ranking and document-level relevance estimation tasks.


