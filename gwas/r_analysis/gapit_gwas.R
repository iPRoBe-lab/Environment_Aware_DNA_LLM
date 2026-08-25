library(GAPIT)

args       <- commandArgs(trailingOnly=TRUE)
geno_path  <- args[1]
map_path   <- args[2]
pheno_path <- args[3]
model      <- args[4]
pca_total  <- as.integer(args[5])

myGD              <- read.csv(geno_path, check.names=FALSE)
colnames(myGD)[1] <- "Taxa"

myGM           <- read.csv(map_path)
colnames(myGM) <- c("Chromosome", "Position", "Name")
myGM           <- myGM[, c("Name", "Chromosome", "Position")]

myY <- read.csv(pheno_path)

tryCatch(
  GAPIT(Y=myY, GD=myGD, GM=myGM, PCA.total=pca_total, model=model),
  error=function(e) message("Post-processing error (results saved): ", e$message)
)
