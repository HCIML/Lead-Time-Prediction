# Lead-Time-Prediction 
## Descriptive Lead Time Analysis

Disclaimer: this is the leading repository for Python source code of the project. If you wish to work with the standalone Windows executable (no Python knowledge required) please contact: kiryl.batvinyeu@sap.com.

The hosted project is a prototype which performs **Descriptive Lead Time Analysis**. It consists in total of three major blocks: 

1) Data transformation to generate the time series format with a defined history horizon and the date marking 'today' as the start date of the lead time prediction or the first date after the history horizon. The time series are defined by an aggregation of the data to the specified granularity thus performing a grouping. The output consists of the time series Excel file 'Timeseries.xlsx' in the cache directory.  


2) Outlier detection and removal by one of the four methods: Maximum Cap, Sigma Confidence, Box Plot and Isolation Forest. The Maximum Map can be used together with one of the other three methods. The output consists of the Excel file 'Outliers.xlsx' with the outliers in the cache directory and the visualizations.

<img src="https://media.github.tools.sap/user/87988/files/3d70db86-5f6a-4eae-a3c9-9e5fa9da436d"  alt="BoxPlot" width="800" height="400">  

<img src="https://media.github.tools.sap/user/87988/files/ffdf959f-4984-45a6-ac70-ee58e2d5b894"  alt="Density" width="800" height="400">

3) Descriptive lead time analytics on data with the outliers removed. The output consists of the Excel file 'Statistics.xlsx' with the analysis results and the chart visualization. The analysis includes the following characteristics: Mean, Standard Deviation, Coefficient of Variation, Quantiles, Median, Lower and Upper Bounds.

<img src="https://media.github.tools.sap/user/87988/files/b6fe6865-bc16-41f5-8183-2913fbb61293"  alt="Chart" width="800" height="400"> 

## Prerequisites
The folder /IBP_ML/Inventory needs to be created in root directory. In the folder, the data file 'data.pcl/.txt/.csv' contains the data to be processed.

## Parameters
The parameters can be maintained in the Config.txt file. In the following, the major parameters are listed. 

### I. Meta Parameters
**features**: *['str', 'str', ..].* The features used for the label prediction are specified here  

**granularity**: *['str', 'str', ..].* The granularity of the time series, a subset of the 'features'  

**time level**: *'DAYS' or 'WEEKS'.* Defines the time level of the time series  

**label**: *'str'.* Defines the label column

**date**: *'str'.* The column containing the date is specified here  

**today**: *'str'.* The first date of the future lead time prediction  

**date format**: *'str'.* Needs to be specified according to the date format of the data, specifically the date column

**history**: *integer.* Depending on the time level, how many days or weeks define the time series history used for the descriptive analysis or prediction  

**confidence**: *float.* Defines the quantile for the prediction error upper and lower bound  

**outliers**: *True or False.* Enables outlier analysis 

**analyze**: *True or False.* Enables descriptive analysis  

**chart**: *False or integer.* Enables the analytic charts and specifies the width  

**predict**: *True or False.* Only basic ARIMA prediction and visualization is supported in the descriptive analytics prototype as a PoC   

**minimum group size**: *None or integer.* If a time series has less data points than that, it is excluded from the calculation and marked as outliers

**demo**: *True or False.* Preview mode in which the output is shortened to one figure per step  

### II. General Outlier Detection Parameters

**remove outliers**: *True or False.* Enables automatic outlier removal  

**cap**: *None or integer.* Defines the maximum value cap as outlier removal method  

**outlier removal method**: *'confidence', 'forest' or 'box_plot'*  

**visualize outlier flag**: *True or False.* Enables outlier visualization  

**visualizations sorted by count**: *True or False.* If True, all visualizations that display groups are sorted by number of data points in each group in descending order  

### III. Specific Outlier Detection Parameters

### III.a Box Plot

**group**: *None, meta_params["granularity"]*. Outlier removal based on total data volume or per granularity unit, typically lane  

**minimum group sample**: *None or integer.* Box plot outlier detection method is performed only for groups or keys with more than this number of time series data points. Smaller groups are marked as Inliers  

**remove small groups**: *None or integer.* Smaller groups are generally marked as outliers, the integer should be set to the value that is reasonable for the specific outlier removal method  

**visualize**: *None or integer.* Enables and specifies the width of the outlier visualization, an overlay of a grouped scatter plot and a histogram

### III.b Isolation Forest

**group**: *None, meta_params["granularity"]*. Outlier removal based on total data volume or per granularity unit, typically lane  

**minimum group sample**: *None or integer.* Isolation Forest outlier detection method is performed only for groups or keys with more than this number of time series data points. Smaller groups are marked as Inliers  

**remove small groups**: *None or integer.* Smaller groups are generally marked as outliers, the integer should be set to the value that is reasonable for the specific outlier removal method  

**contamination**:  *float or 'auto'.* Percentage of outliers per group as single leading parameter of the outlier removal method

**maximum outliers per group**: *None or integer.* Outliers number limit per group as single leading parameter of the outlier removal method in combination with an 'auto' for contamination  

**visualize**: *None or integer.* Enables and specifies the width of the outlier visualization, an overlay of a grouped scatter plot and a histogram  

**density**: *True or False.* Enables detailed density visualization via violin plot

**scale**: *“area”, “count”, “width”.* Defines how the widths of the violins in the violin plot are scaled for an intuitive density representation 

**smoothing kernel**: *float.* Defines the smoothness or precision of the density visualization

**scatter**:  *None, 'o', '.' etc.* Enables display of individual data points and specifies the symbol by which they are displayed

### III.c Confidence

**group**: *None, meta_params["granularity"]*. Outlier removal based on total data volume or per granularity unit, for some of the business use cases it is the Transportation Lane  

**significance**: *integer, 3 to 5.* Specifies if the outliers are defined as being beyond 3 to 5 Sigma from the Mean Value, assuming that the data distribution is a Gaussian








