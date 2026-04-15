#DATA PREPROCESSING

#Need to check for blank values as well as n/a vlaues.
#Impute the data using the mean/median/mode of the columns (where applciable datatypes)

#Need to show how many things I can do, will do one of each graph for each column adn run through everything as best as possible.

#NEED TO CORRECT THE NORTHING AND EASTING OSGR HISTOGRAMS as they are stretched because some of the data is high. This makes it difficult to see the differnec in the before and after.








# Notes whilst working through the final version

Whilst working through the final version of the code, there were a few errors that I came across. The first error was that I needed to fix the section that extrapolated the hour from the time_of_day column within the dataset.

Secondly, I also had to ensure that the urban_or_rural_area was a binary value for the binary classification. This is something that I ran into an error with. On the dataset itself, it was hard to read. Therefore, what I did was print the first 50 values from the urban_or_rural_area column to see if they were all binary.

The image displaying the results of the test, along with the file that I conducted the test within is in the "Testing" file. The results showed that the number 1,2 and 3 were used within the column. Therefore, this is not suitable for binary classification and shoudld be multiclass classification instead. This is something that I initially missed as the dataset was very difficult to read.

However, for this type of data (whether the accident occured on a rural road or an urban road), the data SHOULD be binary. The third classification should not be there. Therefore, I chose to go back and remove these during the daa pre-processing stages.


