# factor_generator

Generate low-frequency alpha and risk factors.

## File Digestion

The following modules are available. Most are hidden for confidential purposes.

- [raw_factor](raw_factor): the main portion of factor generation, including alpha factors, risk factors, and support factors that help generate alpha factors.
  - [factor_gen.py](raw_factor/factor_gen.py): for alpha factors
  - [support_factor_gen.py](raw_factor/support_factor_gen.py): for support factors
  - [StyleFactorGenerator.py](raw_factor/StyleFactorGenerator.py): for risk factors
- [process_raw](process_raw): hidden
- [factor_stats](factor_stats): hidden
- [factor_validation](factor_validation): hidden

## Environment Requirement 

This project heavily relies on the SharedMemory module. Only python >= 3.8.5 is adaptable.

## Implementation Details (TO BE ARCHIVED)

### 1. [raw_factor](raw_factor)

原始因子生成部分需要使用[batch_factor_geneartor.py](bin/batch_factor_generator.py)(该文件调用了[FactorGenerator.py](raw_factor/FactorGenerator.py)), 且每次生成需只生成统一类型的因子. 目前支持单次批量生成下列其中一类因子: 设置[config.py](configuration/config.py)中的required_data_types即可.

- 'trade'(交易信息)
- 'trade_dev'(交易衍生信息, 与'trade'同时生成)
- 'depth'(tick信息)
- 'mins'(分钟数据)
- '5_mins' (5分钟频的数据)
- 'eod'(股票日频数据)
- 'fund'(基本面日频数据)
- ''(无需读入任何数据, 只读辅助因子的因子)

其中, 'trade','trade_dev', 'depth', 和'mins'的因子都是分股票开多进程; 'eod'数据则是整体读入后存入共享内存, 再在每个子进程重构eod数据来计算因子.

因子编写需符合下列规格:

```Python
def foo_factor(stock_data_dict, param_list):
    """
    因子编写方式示例
    :param stock_data_dict: required_data_types中有哪一项, dict中就有哪一项.
        例如选择eod, 则stock_data_dict结构为: {'eod': eod_data_dict}
    :param param_list: 将因子涉及参数放入一个list, 无参数则放入空list
    :return eod_df
    """
    factor_df = ...
    return factor_df 
```

以mom动量因子为例, 动量因子需读入eod数据, 则stock_data_dict中只有一个key('eod')和value(eod_data_dict).

```Python
def mom(stock_data_dict, param_list):
    """ 过去N天收益率 """
    # 提取数据
    eod_data_dict, window = stock_data_dict['eod'], param_list[0] 
    price_df = eod_data_dict['ClosePrice']*eod_data_dict['AdjFactor']
    # 计算因子值
    factor_df = price_df / price_df.shift(window,axis=1)-1 
    # 返回dataframe
    return factor_df
```

### 2. [process_raw](process_raw)

因子处理部分则是用共享内存加异步IO的模式批量对原始因子值进行行业市值中性化.

TODO: add more details to README
