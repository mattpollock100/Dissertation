library(rcarbon)
library(sf)

#dataset <- read.csv("C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/RadioCarbonDatesPeru.csv")

#dataset <- read.csv("C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/sites_far_from_peruvian_coast.csv")

dataset <- read.csv("C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/sites_close_to_peruvian_coast.csv")

dataset$SiteCode <- paste("S",as.numeric(dataset$SiteName),sep="")

timerange <- c(6000, 0) # define analysis time range

#binsense() indicates 200 is a good compromise (see below)
bins <- binPrep(sites = dataset$SiteCode, ages = dataset$Age, 200) 

#I've set normalised to TRUE, not entirely sure what it does
calDates <- calibrate(dataset$Age, errors = dataset$Error, normalised = FALSE, calCurves = "intcal20") #,
                      #resOffsets = dataset$DeltaR, resErrors = dataset$DeltaRErr, ncores=3)

#I've set datenormalised and spdnormalised to true, to make
#different SPDs comparable. 
spd_result <- spd(calDates, timeRange = timerange, bins=bins, 
                    datenormalised = FALSE, spdnormalised = TRUE, runm = 100)

plot(spd_result, calendars = "BP", type = "standard")

# Convert SPD result to data frame
spd_df <- as.data.frame(spd_result$grid)
# Save the data frame to a CSV file
write.csv(spd_df, "C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/spd_result_close.csv", row.names = FALSE)
