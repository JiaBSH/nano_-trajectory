# Raw-frame analysis 代码结构

## 运行链路

```text
analyze-rawframe.py
    -> cli.py              解析配置文件路径
    -> config.py           加载、校验并解析所有路径
    -> pipeline.py         按配置选择并执行处理步骤
    -> tracker.py          初始化单次运行状态并组合领域能力
         -> inputs.py      输入文件与尺度信息
         -> geometry.py    多边形和液滴几何计算
         -> processing.py  逐帧读取和目标特征提取
         -> tracking.py    跨帧关联、ID 和速度序列
         -> exporting.py   CSV 输出
         -> annotations.py 标注能力组合入口
         -> plot_style.py  公共绘图样式
         -> trajectory_plots.py 轨迹图能力组合入口
         -> summary_plots.py    帧统计和演化图
```

## 标注模块

| 模块 | 唯一职责 |
| --- | --- |
| `annotation_support.py` | 标注参数的公共校验 |
| `annotated_images.py` | 在标注尺寸图像上绘制结果 |
| `rawframe_annotations.py` | 在原始帧上绘制当前分析类别 |
| `all_category_annotations.py` | 在原始帧上绘制全部类别 |
| `annotations.py` | 组合以上能力，维持原公开 API |

## 轨迹绘图模块

| 模块 | 唯一职责 |
| --- | --- |
| `area_trajectory_plots.py` | 面积轨迹 |
| `velocity_trajectory_plots.py` | 速度轨迹 |
| `centroid_trajectory_plots.py` | 质心轨迹 |
| `trajectory_plots.py` | 组合以上能力，维持原公开 API |

## 设计约束

- `analysis.target_category` 是类别名称的唯一来源；默认输出名称由它自动派生。
- 配置层不知道具体分析算法；它只负责配置的序列化与校验。
- 编排层依赖 `TrackerProtocol`，可在测试中注入替代实现。
- `GasTracker` 保留旧公开方法，但具体方法由单一职责模块提供。
- 科学计算方法迁移时保持原 AST 不变；构造函数只负责分阶段初始化。
- 新增输出步骤时在 `AnalysisPipeline._build_steps` 注册，不在入口脚本中堆叠条件分支。
