# Dataset Details

We utilize three widely recognized public benchmark datasets in our anomaly detection experiments:

1. **Server Machine Dataset (SMD)**
   - **Description**: Collected from an Internet company, comprising data from 38 sensors across 28 server machines. The training and test sets are of equal size. Labels are provided to indicate anomalous points, and every dimension contributes to the anomaly parts.

2. **Mars Science Laboratory (MSL) Rover**
   - **Description**: An expert-labeled dataset of 55 dimensions, collected from NASA's Incident Surprise Anomaly (ISA) reports.

3. **Pooled Server Metrics (PSM)**
   - **Description**: Collected internally from multiple application server nodes at eBay. The data consists of 25 features representing server machine metrics, such as CPU usage and memory. The anomaly labels are manually labeled by experts.

## Dataset Statistics

| Benchmarks | Dimension | # Train Samples | # Test Samples | Anomaly Rate |
|------------|-----------|------------|----------------|--------------|
| SMD        | 38        | 7084       | 7084           | 4.2%         |
| MSL        | 55        | 583        | 737            | 10.5%        |
| PSM        | 25        | 1324       | 878            | 27.8%        |

Note: \# indicates the number of windows (samples) after the windowing process. Also, the Anomaly Rate is based on the original data points.

---

You can easily download these datasets from the a cloud provided in the official GitHub repository of the AnomalyTransformer study. For access, visit [this link](https://github.com/thuml/Anomaly-Transformer).
