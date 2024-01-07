library(lme4)

# Read the data (Assuming the file is named 'raw_df.csv')
input_file_path <- '/home/hyruuk/GitHub/cocolab/cc_saflow/tmp/raw_df_allchans.csv'
data <- read.csv(input_file_path)

# Extract the base name of the input file (without extension)
input_base_name <- tools::file_path_sans_ext(basename(input_file_path))

# Create a directory to store the results
results_dir <- paste0('/home/hyruuk/GitHub/cocolab/cc_saflow/tmp/', input_base_name)
dir.create(results_dir, recursive = TRUE)

# Initialize a list to store the results
results <- list()

sink(paste0(results_dir, '/raw_summaries.txt'))
# Loop through each frequency and region
for(freq in unique(data$Frequency)) {
  for(region in unique(data$Region)) {
    # Subset data for the current frequency and region
    subset_data <- subset(data, Frequency == freq & Region == region)
    
    # Normalize the 'Data' predictor using z-score within the subset
    subset_data$Data <- scale(subset_data$Data)
    
    # Define the model with both linear and quadratic terms for 'Data'
    # Task ~ Data + I(Data^2) + (1 | Subject)
    model <- glmer(Task ~ Data + I(Data^2) + (1 | Subject), 
                   data = subset_data, family = binomial)
    
    # Print and store the summary of the model
    cat("Summary for Frequency:", freq, "Region:", region, "\n")
    print(summary(model))
    cat("\n\n") # Adding space between summaries
    
    # Store the model in a file
    model_file_name <- paste0(results_dir, '/model_Freq', freq, '_Region', region, '.RDS')
    saveRDS(model, file = model_file_name)
    
    # Store the model in the results list as well
    results[[paste("Freq", freq, "Region", region, sep="_")]] <- model
  }
}
sink()