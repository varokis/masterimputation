library(arrow)
library(tidyverse)
library(janitor)
library(mice)
library(foreach)
library(doParallel)

registerDoParallel(cores=8)

# entire pipeline:
# IN: dir path,  parameters
# OUT: parquet files of imputed data
dirpath <- "F:/MasterImputation/Code/outdata/datasets/missing/"
outpath <- "F:/MasterImputation/Code/outdata/datasets/imputed/limdata/"
files <- list.files(dirpath)
# Reduce the list of files to only 2h and only missing rate of 0.5
files <- files[grepl(".*2h_0\\.5.*",files)]


# methods used with this script:
# 2l.pan, 2l.norm ... ?

imp <- "mice"
meth <- "2l.norm"

r <- list.files(outpath)
r <- r[grepl(".*2l.norm.*", r)]
r <- sub("mice_2l.norm_","",r)
files <- setdiff(files, r)

start <- Sys.time()

foreach(file=files, .packages = c("arrow","tidyverse", "mice", "janitor")) %dopar% {
  df <- read_parquet(paste0(dirpath, file)) %>%  # read data
    select(-hospital_expire_flag) %>%
    clean_names()                                # remove spaces and prefix cols starting with numbers
  
  methvec <- c(rep(meth, 5), rep("",6))
  
  pm <- make.predictorMatrix(df)
  pm[,"subject_id"] <- -2
  pm["subject_id","subject_id"] <- 0
  
  mice(df, method = methvec, pred=pm, print = TRUE) %>%      # impute
  complete("broad", printFlag=FALSE) %>%                            # create df containing all the imputed data sets
  write_parquet(paste0(outpath, paste(imp, meth, file, sep = "_")))
}

end <- Sys.time()
dur <- end -start

log <- paste0(end, "-- Imputation: ", imp, ", Method: ", meth, ", Files: ", dirpath, ", Start: ", start, ", End: ", end, "\n")
cat(log, file = "F:/MasterImputation/Code/outdata/datasets/log.txt", append = TRUE)

# 2l.pan -----------------------------------------------------------------------------------------------------------------------------
imp <- "mice"
meth <- "2l.pan"

files <- list.files(dirpath)
# Reduce the list of files to only 2h and only missing rate of 0.5
files <- files[grepl(".*2h_0\\.5.*",files)]

r <- list.files(outpath)
r <- r[grepl(".*2l.pan.*", r)]
r <- sub("mice_2l.pan_","",r)
files <- setdiff(files, r)


start <- Sys.time()

foreach(file=files, .packages = c("arrow","tidyverse", "mice", "janitor")) %dopar% {
  df <- read_parquet(paste0(dirpath, file)) %>%  # read data
    select(-hospital_expire_flag) %>%
    clean_names()                                # remove spaces and prefix cols starting with numbers
  
  methvec <- c(rep(meth, 5), rep("",6))
  
  pm <- make.predictorMatrix(df)
  pm[,"subject_id"] <- -2
  pm["subject_id","subject_id"] <- 0
  
  mice(df, method = methvec, pred=pm, print = TRUE) %>%      # impute
    complete("broad", printFlag=FALSE) %>%                            # create df containing all the imputed data sets
    write_parquet(paste0(outpath, paste(imp, meth, file, sep = "_")))
}

end <- Sys.time()
dur <- end -start

log <- paste0(end, "-- Imputation: ", imp, ", Method: ", meth, ", Files: ", dirpath, ", Start: ", start, ", End: ", end, "\n")
cat(log, file = "F:/MasterImputation/Code/outdata/datasets/log.txt", append = TRUE)




