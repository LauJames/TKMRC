# Indexing the passage's paragraph using Elasticsearch

##  data process
- Step1 配置es configuration
修改config.py

- Step2 为数据建立索引
python ir/index.py

- Step3 设置检索策略并根据question检索paragraph
python ir/search.py

