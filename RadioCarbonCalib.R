library(rcarbon)
#radiocarbon_dates <- read.csv("C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/sites_far_from_peruvian_coast.csv")
#calibrated_dates <- calibrate(radiocarbon_dates$Age, radiocarbon_dates$Error, calCurves="intcal20")
#spd_result_far <- spd(calibrated_dates, timeRange=c(6000,0))

#radiocarbon_dates <- read.csv("C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/sites_close_to_peruvian_coast.csv")
#calibrated_dates <- calibrate(radiocarbon_dates$Age, radiocarbon_dates$Error, calCurves="intcal20")
#spd_result_close <- spd(calibrated_dates, timeRange=c(6000,0))

radiocarbon_dates <- read.csv("C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/RadioCarbonDatesPeru.csv")
calibrated_dates <- calibrate(radiocarbon_dates$Age, radiocarbon_dates$Error, calCurves="intcal20")
spd_result <- spd(calibrated_dates, timeRange=c(6000,0))

# Convert SPD result to data frame
#spd_df_far <- as.data.frame(spd_result_far$grid)
# Save the data frame to a CSV file
#write.csv(spd_df_far, "C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/spd_result_far.csv", row.names = FALSE)

# Convert SPD result to data frame
#spd_df_close <- as.data.frame(spd_result_close$grid)
# Save the data frame to a CSV file
#write.csv(spd_df_close, "C:/Users/mattp/OneDrive/Desktop/Climate Change MSc/Dissertation/Data/spd_result_close.csv", row.names = FALSE)


# Plot the first SPD
#plot(spd_result_far, calendar="BP", type="standard", col.fill="lightblue", col.line="blue")

# Add the second SPD to the same plot
#plot(spd_result_close, calendar="BP", type="standard", col.fill=rgb(1, 0, 0, 0.5), col.line="red", add=TRUE)

# Add a legend
#legend("topright", legend=c("Dataset 1", "Dataset 2"), fill=c("lightblue", rgb(1, 0, 0, 0.5)), border=c("blue", "red"))

# plot the overall numbers
plot(spd_result, calendars="BP", type="standard")