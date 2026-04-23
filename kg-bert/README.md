# KG-BERT

用于消融实验，即使用KG-BERT作为弱判别器，验证强判别器(LLM判别器)的有效性

- convert_data.py 用于合成 kg-bert 需要的数据：从 data 与 LLM_Discriminator/data 中合成，因此运行前请获得这两个数据
- 然后就可以使用 scripts/ 中和 kg-bert 相关的脚本运行训练与测试
