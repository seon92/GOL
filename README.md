# GOL
[NeurIPS 2022] Geometric order learning for rank estimation [[paper]](https://openreview.net/pdf?id=agNTJU1QNw)

[Seon-Ho Lee](https://scholar.google.co.kr/citations?user=_LtQ4TcAAAAJ&hl=en), Nyeong-Ho Shin, and Chang-Su Kim

---
## Dependencies
* Python 3.8
* Pytorch 1.7.1
---
## Datasets
- [MORPH II](https://ebill.uncw.edu/C20231_ustores/web/classic/product_detail.jsp?PRODUCTID=8) 
* For MORPH II experiments, we follow the same fold settings in this [OL](https://github.com/changsukim-ku/order-learning/tree/master/index) repo.
- [Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html)
* For Adience experiments, we follow the official splits.
- [CACD]
- [UTK] 
---
## Usage
```
    $ python train.py
```    
* Modify 'cfg.dataset' and 'cfg.setting' for training on other/custom dataset
* You may need to change 'cfg.ref_point_num' and 'cfg.margin' to obtain decent results.



