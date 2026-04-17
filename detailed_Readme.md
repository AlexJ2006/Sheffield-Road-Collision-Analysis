# This file is a detailed ReadMe explaining the results that I have found from each section.

# Preprocessing

* local_highway_authority_current

For the cleaning of this column, I decided to fill the column with the same, continuous value. The results of this are displayed within the histogram below

![local_highway_authority_current Cleaning Results Histogram](Results/local_highway_authority_current/HISTOGRAM-local_highway_before_after.png)

Furthermore, below is an image of the cleaning results displayed to the user

![local_highway_authority_current Cleaning Results TEXT](Results/local_highway_authority_current/CLEANING_RESULTS_local_highway.png)

Results & Justification:

It is evident from the images provided that my cleaning method has been effectve for the local_highway_authority_current column. When I first recieved the dataset (pre-cleaning), the total number of valid rows was just over 2500. We can see this from the before histogram. As depicted in the second image (the text image), this meant that there were 5,314 N/A values present within the column. After this column had been cleaned, the histogram shows that the values present reached nearly 8000. This is accurate for the dataset as there were 7,933 total records present.

Therefore meaning, I had filled every row with the same value. This makes sense for this dataset for two reasons.

Firstly, the dataset initially contained roughly 2600 of one value. This meant that it was extremely likely that the rest of the values were going to be the same. Secondly, the dataset is for Sheffield ONLY. Therefore, the local authority that looks after the highways will be the authority within the local area of Sheffield, meaning that it will be the same throughout the entire dataset. If it wasn't the same area code, the data present wouldn't be for sheffield and it would need to be dealt with accordingly as it would present innacuracies within the model, further down the line.



* latitude & longitude

The way in which I decided to clean this column and prepare it for use was with mean imputation. Below is the image of the graphs containing the before and after results for the longitude and latitude columns.

![local_highway_authority_current Cleaning Results Histogram](Results/latitude_&_longitude/HISTOGRAM-latitude_longitude_before_after.png)

Furthermore, below is an image of the cleaning results displayed to the user

![local_highway_authority_current Cleaning Results TEXT](Results/latitude_&_longitude/CLEANING_RESULTS-latitude_&_longitude.png)

Results & Justification:

From both of the images provided above, it is clear that the mean imputation for both of the columns has been extremely effective. For the latitude column, the before image shows that the highest number of one single value for the column reached just over 1200. In the second (after) image, this same value reaches just over 5000. The same applies for the longitudinal results.

Looking at the initial data that I was provided with, I believed that it was the best option to use mean imputation for each of these columns as the data was clustered around certain points. There werent any outliers within the columns for latitude and longitude. Therefore, I was unlikely to draw any innacuracies into the dataset by entering the mean values for each column. I believe this was the right decision for the datatype and the circumstances.

* urban_or_rural_area

For this column, the cleaning was slightly more complicated. I decided to remove values from this column that I deemed innacurate and unnecessary and keep the values 1 and 2. I then re-mapped 1 and 2 to 0 and 1. I will justify this and explain my methodology in the results and justification section below, after the image.

The image below shows the output of the successful cleaning process for this column.

![Cleaning Results for Urban or Rural Area Column](Results/urban_or_rural_area/CLEANING_RESULTS-urban_or_rural_area.png)

Results & Justification:

As I mentioned earlier, for this column I decided to remove some of the values that were initially present. These values were -1 and 3. My reasoning behind this decision was that I could safely deduct that the values -1 and 3 signified inaccurate values of values that hadn't been recorded. These values didn't make sense to be within the datset. However, it was logical that the values 1 and 2 signified urban and rural areas.

One of the things that I was preoccupied with during this process was how low the number of accidents that occured within the rural area was. This value can be seen within the image above in the final section. It shows the value of rural accidents to be 410. Initially, I thought this value was much too low. However, upon reflection, I decided that this value was likely to be accurate as there would be much less traffic on the rural roads within the Sheffield area and therefore fewer accidents (likely only around 8.6% of accidents overall). Therefore, I decided to leave this value as it is.

One of the main reasons behind cleaning the column in this way and mapping the new values to 1 and 0 was for the binary classification. Binary Classification wouldn't have worked on this data if I had kept it as it previously was (as they weren't binary values). I will mention this further on in this file.

* location_northing_osgr & location_easting_osgr

For the location_easting_osgr and location_northing_osgr columns, I imputed the N/A values with the mean.

![Cleaning Results and mean value or easting & northing locations](Results/location_northing_easting_osgr/CLEANING_RESULTS-easting_northing.png)

Results & Justification:

As I emphasised within the comments during this section of the code file, these two columns were actually near-perfect to begin with. Neither of them required much cleaning with only 65 rows each (a total of 130 over both columns), presenting as N/A. Over 7,933 rows in the location_easting_osgr, having just 65 N/A values meant that only 0.8% of the column needed replacing with the mean. This is the same for the location_easting_osgr column too.

I believe that mean imputation was the right choice to fill thesee N/A values as it reduces variance in the dataset. By this, I mean that it keeps the data within the same grographical area by using the mean and avoids accidentally creating any outliers. The data that I am imputing will follow the trend that has already been set by the other data points. I am avoiding creating any unnecessary inaccuracies here as mean imputation allows the new data to stay within the safe boundaries that currently exist within the dataset.

Furthermore, given the fact that I was only imputing 0.8% of each column, if I was to have imputed these values inaccurarately, it is unlikely to actually have much of an effect on the accuracy of the final model itself.

* collision_adjusted_severity_serious & collision_adjusted_severity_slight

For this section, I have tried to use the mode imputation again. I also ran an initial N/A value count and a final N/A value count. I used the same logic as I did for the section above.

Evidence of the output can be found in the image below:

![Cleaning Results for Collision Severity Serious & Slight](Results/collision_adjusted_severity_serious_&_slight/CLEANING_RESULTS-serious_&_severe.png)

Results & Justification:

Once again, as for the section above, the columns showed that they had very few N/A values. However, when using the mode for these columns, the mode was found to be 0 for each as the data is binary data and 0 is evidently the most common value in this instance.














* Outlier Detection (IQR Method)

























* Final Null Value Count

At the end of the preprocessing section, I performed another null value count. This came to 0 over all of the 44 columns within the dataset. The Image of this is below. This image depicts exactly what is displayed to the user on runtime at the end of this section.

![Final N/A Value Count](Results/Final_NA_Value_Count/Final_NA_Value_Count.png)

