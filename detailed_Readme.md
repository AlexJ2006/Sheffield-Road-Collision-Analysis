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

* Final Preprocessing Result Charts (Graphs)

Below is an image containing the results of the final preprocessing checks.

![Final Preprocessing Results](Results/Final_preprocessing_Results_Graphs/Final_preprocessing_results_graphs.png)

The graph on the top left depicts the final distribution for the Serious and Not Serious collisions within the data set. From the graph, we can see that there is a significantly larger number of Not Serious collisions than Serious collisions. This means that the model is likely to predict a collision to be Not Serious. However, in the real world, not serious collisions are more frequent. Therefore, the model should still be accurate.

Moving to the top middle image, this shows the geographical clustering of the accidents. It is clear from this image that the majority of these are tightly packed. This again makes sense as it depicts the collisions within the Sheffield Urban area, which only spans a certain distance. Therefore, the data depicted here is likely to be accurate. It is aslo consistent which is a positive.

The top right graph shows a KDE Plot for the latitudinal distribution. The result is that there is a very sharp peak at the top of the graph, this shows that most of the collisions occur within a narow latitudinal range. This is also a good thing for me to see as it once again reinforces that the data is accurately within the specific region of the city of Sheffield. 

The bottom left image shows a KDE plot for the longitudinal distribution. This shows the same as the latitudinal graph that I have written about above. However, this image shows there is a wider spread of data longitudinally than there was latitudinally.

The bottom middle graph shows the same type of clustering as the latitude/longitude graph above it. However, using the eating/northing data. There are a few clear outliers within this image. This confirms that the coordinate transformation between the latitude and longitude data and the easting and northing data is consistent as we see very similar results within both graphs.

Finally, the bottom right graph shows that one category dominates entirely within the column. This is accurate as the collisions recorded will be recorded through the authority based within the city of Sheffield. This was data that as I mentioned earlier at the start of the file, I imputed myself.

# Feature Engineering

* Adding the Engineered Features to the dataset

The first task that I undertook was engineering my features. In order to do this, I needed to create new features using existing columns within the dataset.

Once I had done this, I displayed to the user which new features would be added to the dataset. An image of the new features that I added is below. I will then explain where each of the features came from. I will also add some code snippets from where I created the new features from the existing features.

![New Feature List](Results/Feature-Engineering-Results/New_Feature_List.png)

The first new feature that I added was "is_weekend". This was very simple. I did this by taking the current "day_of_week" column and using the logic "if the integer is equal to 6 or 7, it belongs to this feature meaning it is a weekend". This meant that any day of the week that then wasn't taken into the new is_weekend feature was left as a normal weekday. This could then be represented within the graph which I will show later within the visualisation section of feature engineering.

The second new feature that I added was "time_of_day". This was slightly more complicated. I already had a column that was called "time". This data within this column was presented in the format of a 24-hour clock. Therefore, all I needed to do was extrapolate the hour from the column (the first each piece of data), and then map it to either Night, Morning, Afternoon or Evening. I also defined "bins" for this section. These helped to place the correct time within the correct category (morning, afternoon etc)

As I stated within the comments in my code, the bins were defined as 0-6 being Night (12am to 6am), 6-12 being morning (6am to 12pm), 12-18 being afternoon (12pm - 6pm) and finally, 18-24 being evening (6pm - midnight). 

The Third new feature that I added was "risk_score". The aim of this was to calculate a total risk score that combined the number of vehicles involved in the collision with the number of casualties from the collision. It also aimed to give more weight to the number of casualties involved in the accident as they are a direct indicator or severity meaning the higher the number of casualties, the more severe the accident.

The weighting was simple, I did the following:

(number_of_vehicles x 0.4) + (number of casualties x 0.6)

The Fourth new feature that I added was "casualties_per_vehicle". This was done by using another simple calculation. I will show this calculation below:

(number_of_casualties) / (number_of_vehicles)

This could then return an estimated number of casualties per vehicle. However, this figure is not guarenteed to be accurate 100% of the time and may vary depending on the circumstances of the accident.

The Fifth new feature that I added was "speed_urban_interaction". This feature aimed to capture how speed limits may have different effects on the accidents within urban areas vs rural areas. This simply flagged the speed limit and linked it to either a rural or urban area.

The Sixth new feature that I added was "high_speed_zone". Simply, this checks whether the "speed_limit" was > (greater than) 60. If it was, it is classed as a high speed zone. Collisions within a high speed zone are more likely to be serious collisions.

The Seventh and final new feature that I added was "collision_age". This aimed to see whetehr there was a difference in the number of collisions in previous years comapred to nowadays. Theoretically, this could then tell me whether road safety has improved or not over the years. However, there are also other factors that come into play here. For example, there may be significantly more drivers on the roads than there has been before. This would mean that the road safety might well have improved but we may not be able to see this statistically represented here.

* Feature Engineering - Distributions

Once I had added all of my new features, I decided to display my findings on graphs, much like I did earlier for the final preprocessing results. The graphs are pictured below and I will now explain my findings.

![Feature Engineering - Graphs](Results/Feature-Engineering-Results/Engineered_features_Visualisation.png)

Beginning at the top left, it was found that the majority of crashed happened on a weekday with just over 5000 on a weekday and around 2500 occuring on the weekend. This seems likely to me as most people will be driving throughout the week and commuting. Furthermore, rush hours from monday-friday mean that there is a high density of road traffic at certain times. This is likely to lead to a lot of collisions. The majority of which are minor (as we saw earlier in the severity chart).

The graph at the top middle of the image shows which time of day it was when the collisions occured. Once again, we can see that the charts somewhat confirm my beliefs as there are minimal accidents throughout the night and the majority happen in the afternoon between the hours of 12pm and 6pm (when the final rush hour of the day occurs). This is closely followed by the evening and then morning accidents trail slightly behind. I believe there could be a few different reasons behind the trends that we can see here.

During the final rish hour of the day (usually between 5pm and 6:30pm), most people will be leaving work. There will be a high volume of traffic on the road and people will be tired after being at work all day. This could therefore lead to mistakes and evidently accidents. This could also explain why the accidents occuring in the evening section of the graph is so high. In the morning, this is likely to be slightly lower (and this is backed up by the visual evidence within the graph), as although there is a high volume of traffic on the road. People are usually going to work so they are unlikely to be as tired as when they are coming home and clearly much less likely to be involved in an accident.

Moving to the top right of the image, we can see the risk score distribution. This once again confirms what we saw earlier with teh majority of the collisions being classed as "not serious", as the majoroity of the cases are between 1 and 2. This measn there weren't many vehicles involved and therefore there weren't many casualties.

Moving to the bottom left graph, we can see that the number of casualties per vehicle is mostly towards the lower end of the scale with the majority being at 1 casualty per vehicle. This would lead me to beliee once again that the majority of the accidents that occur aren't serious.

Moving the the bottom middle image, we can see that the majority of collisions occur at normal speed (any speed below 60mph). This also confirms my suspicions that the majority of accidents aren't serious. with over 7000 collisions occuring at a normal speed.

Finally, moving to the bottom right image, we can see that the number of collisions has in fact reduced over the last 40 years. This is almost certainly due to the new road safety measures and modern technolgy within cars that actually prevent them from crashing. For example, asissted/automated driving. This could also be due to stricter driving tests making for a higher standard driving across the country. This was a very interesting and surprising result for me as I didn't believe this would be the case. This was something that I hilighted earlier..

# Supervised Learning

* Correlation Matrix (all features)

* Confusion Matrix (MultiClass Severity)

* Top 10 Featrure Importances - Gradient Boosting

* Multiclass Model Comparison


